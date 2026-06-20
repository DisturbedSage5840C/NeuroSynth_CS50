from __future__ import annotations

import struct
from uuid import uuid4

from neurosynth.connectors.wearable_stream import WearableStreamConnector
from neurosynth.core.config import NeuroSynthSettings


def test_parse_bin_packet() -> None:
    connector = WearableStreamConnector(NeuroSynthSettings())
    patient_id = uuid4()
    payload = struct.pack("<qfffff", 1000, 1.0, 2.0, 3.0, 31.5, 100.0)

    sample = connector._parse_bin_packet(payload, patient_id, "dev1")
    assert sample.patient_id == patient_id
    assert sample.temperature_c == 31.5
    assert sample.light_lux == 100.0
