from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import pandas as pd
from asyncio_mqtt import Client as MQTTClient
from confluent_kafka import Consumer, Producer

from neurosynth.core.logging import get_logger
from neurosynth.data.iceberg_catalog import IcebergDomainCatalog

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

MODALITIES = {"eeg", "accel", "heart_rate"}


@dataclass
class WearableEvent:
    patient_id: str
    patient_cohort: str
    modality: str
    timestamp: datetime
    value: float


class WearableWindowAggregator:
    """5-minute sliding window aggregator for wearable telemetry."""

    def __init__(self, window_minutes: int = 5) -> None:
        self.window = timedelta(minutes=window_minutes)
        self.buffers: dict[tuple[str, str], deque[WearableEvent]] = defaultdict(deque)

    def add(self, event: WearableEvent) -> dict[str, Any] | None:
        key = (event.patient_id, event.modality)
        q = self.buffers[key]
        q.append(event)

        cutoff = event.timestamp - self.window
        while q and q[0].timestamp < cutoff:
            q.popleft()

        if len(q) < 2:
            return None

        values = [x.value for x in q]
        return {
            "timeseries_id": str(uuid4()),
            "patient_id": event.patient_id,
            "patient_cohort": event.patient_cohort,
            "ingestion_date": date.today(),
            "modality": event.modality,
            "window_start": q[0].timestamp,
            "window_end": q[-1].timestamp,
            "metric_mean": float(sum(values) / len(values)),
            "metric_std": float(pd.Series(values).std(ddof=0)),
            "sample_count": len(values),
        }


class WearableKafkaBridge:
    def __init__(
        self,
        mqtt_host: str,
        mqtt_port: int,
        kafka_bootstrap_servers: str,
        iceberg: IcebergDomainCatalog,
    ) -> None:
        self.log = get_logger(__name__)
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        self.producer = Producer({"bootstrap.servers": kafka_bootstrap_servers})
        self.consumer = Consumer(
            {
                "bootstrap.servers": kafka_bootstrap_servers,
                "group.id": "neurosynth-wearable-aggregator",
                "auto.offset.reset": "earliest",
            }
        )
        self.iceberg = iceberg
        self.aggregator = WearableWindowAggregator(window_minutes=5)

    async def mqtt_to_kafka(self, topic_prefix: str = "wearable") -> None:
        async with MQTTClient(self.mqtt_host, self.mqtt_port) as mqtt:
            await mqtt.subscribe(f"{topic_prefix}/#")
            async for message in mqtt.messages:
                payload = json.loads(message.payload.decode("utf-8"))
                modality = str(payload.get("modality", "")).lower()
                if modality not in MODALITIES:
                    continue
                kafka_topic = f"neurosynth.{modality}"
                self.producer.produce(kafka_topic, json.dumps(payload).encode("utf-8"))
                self.producer.poll(0)

    def consume_and_aggregate(self, max_messages: int = 500) -> int:
        topics = [f"neurosynth.{m}" for m in sorted(MODALITIES)]
        self.consumer.subscribe(topics)

        emitted = 0
        for _ in range(max_messages):
            msg = self.consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                self.log.warning("kafka.consume_error", error=str(msg.error()))
                continue

            payload = json.loads(msg.value().decode("utf-8"))
            event = WearableEvent(
                patient_id=str(payload["patient_id"]),
                patient_cohort=str(payload.get("patient_cohort", "unknown")),
                modality=str(payload["modality"]),
                timestamp=datetime.fromisoformat(payload["timestamp"]).astimezone(timezone.utc),
                value=float(payload["value"]),
            )
            row = self.aggregator.add(event)
            if row is not None:
                self.iceberg.append_dataframe("biomarker_timeseries", pd.DataFrame([row]))
                emitted += 1

        self.consumer.close()
        self.log.info("kafka.aggregate_complete", rows=emitted)
        return emitted
