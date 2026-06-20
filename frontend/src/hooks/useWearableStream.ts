// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { useEffect, useMemo, useState } from "react";
import { streamUrl } from "@/lib/api";
import { WearablePoint } from "@/types/api";

const WINDOW_MS = 5 * 60 * 1000;

export function useWearableStream(): WearablePoint[] {
  const [points, setPoints] = useState<WearablePoint[]>([]);

  useEffect(() => {
    const source = new EventSource(streamUrl("/stream/biomarkers"));

    source.onmessage = (event) => {
      const payload = JSON.parse(event.data) as WearablePoint;
      setPoints((prev) => {
        const merged = [...prev, payload];
        const now = Date.now();
        return merged.filter((p) => now - new Date(p.timestamp).getTime() <= WINDOW_MS);
      });
    };

    source.onerror = () => {
      source.close();
    };

    return () => source.close();
  }, []);

  return useMemo(() => points.sort((a, b) => +new Date(a.timestamp) - +new Date(b.timestamp)), [points]);
}
