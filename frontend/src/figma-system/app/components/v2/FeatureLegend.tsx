// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
/**
 * FeatureLegend (Gap 7) — explains what encoded feature values mean
 * (e.g. Gender 0 = Female), so clinicians never see opaque 0/1/2 codes.
 */
import { useMemo, useState } from "react";
import { FEATURE_MAP, FEATURE_SCHEMA, type FeatureMeta } from "@/lib/featureSchema";

interface FeatureLegendProps {
  /** Restrict the legend to these feature keys; omit to show all encoded features. */
  visibleFeatures?: string[];
  /** Render collapsed by default with a toggle. */
  collapsible?: boolean;
  title?: string;
}

function legendEntries(visibleFeatures?: string[]): FeatureMeta[] {
  const source = visibleFeatures
    ? visibleFeatures.map((k) => FEATURE_MAP[k]).filter(Boolean)
    : FEATURE_SCHEMA;
  // Only features that carry a categorical/boolean encoding need a legend.
  return source.filter((f) => f.values && Object.keys(f.values).length > 0);
}

export function FeatureLegend({
  visibleFeatures,
  collapsible = true,
  title = "Feature encoding reference",
}: FeatureLegendProps) {
  const entries = useMemo(() => legendEntries(visibleFeatures), [visibleFeatures]);
  const [open, setOpen] = useState(!collapsible);

  if (entries.length === 0) return null;

  return (
    <div className="rounded-lg border border-border bg-card/60 mt-3">
      <button
        type="button"
        onClick={() => collapsible && setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-3 py-2 text-left"
        aria-expanded={open}
      >
        <span className="text-xs font-medium text-muted-foreground">{title}</span>
        {collapsible && (
          <span className="text-xs text-muted-foreground font-mono">{open ? "−" : "+"}</span>
        )}
      </button>

      {open && (
        <div className="px-3 pb-3 space-y-2">
          {entries.map((f) => (
            <div key={f.key} className="flex flex-col gap-0.5">
              <span className="text-xs font-medium text-foreground" title={f.full_name}>
                {f.label}
              </span>
              <span className="text-xs text-muted-foreground font-mono">
                {Object.entries(f.values!).map(([k, v]) => `${k} = ${v}`).join("  ·  ")}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
