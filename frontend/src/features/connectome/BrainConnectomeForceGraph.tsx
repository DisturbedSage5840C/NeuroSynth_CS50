// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";
import { CausalEdge } from "@/types/api";

export interface BrainRegionNode {
  id: string;
  risk: number;
  structuralMetric: number;
}

export interface BrainConnectomeForceGraphProps {
  width?: number;
  height?: number;
  nodes: BrainRegionNode[];
  edges: CausalEdge[];
  onNodeSelect?: (node: BrainRegionNode) => void;
}

const riskColor = d3.scaleLinear<string>().domain([0, 0.5, 1]).range(["#14b8a6", "#f59e0b", "#f97373"]);

export function BrainConnectomeForceGraph({
  width = 640,
  height = 420,
  nodes,
  edges,
  onNodeSelect,
}: BrainConnectomeForceGraphProps): JSX.Element {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [selected, setSelected] = useState<BrainRegionNode | null>(null);

  const graphData = useMemo(() => ({
    nodes: nodes.map((n) => ({ ...n })),
    links: edges.map((e) => ({ source: e.from, target: e.to, strength: e.strength })),
  }), [nodes, edges]);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const simulation = d3
      .forceSimulation(graphData.nodes as d3.SimulationNodeDatum[])
      .force("link", d3.forceLink(graphData.links).id((d: any) => d.id).distance(80))
      .force("charge", d3.forceManyBody().strength(-180))
      .force("center", d3.forceCenter(width / 2, height / 2));

    const link = svg
      .append("g")
      .selectAll("line")
      .data(graphData.links)
      .enter()
      .append("line")
      .attr("stroke", "#7dd3fc")
      .attr("stroke-opacity", (d: any) => Math.min(0.9, Math.max(0.15, d.strength)))
      .attr("stroke-width", (d: any) => 1 + d.strength * 4);

    const node = svg
      .append("g")
      .selectAll("circle")
      .data(graphData.nodes)
      .enter()
      .append("circle")
      .attr("r", 8)
      .attr("fill", (d: any) => riskColor(d.risk))
      .attr("tabindex", 0)
      .attr("role", "button")
      .on("click", (_event, d: any) => {
        setSelected(d);
        onNodeSelect?.(d);
      })
      .call(
        d3
          .drag<any, any>()
          .on("start", (event: any) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
          })
          .on("drag", (event: any) => {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
          })
          .on("end", (event: any) => {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
          })
      );

    simulation.on("tick", () => {
      link
        .attr("x1", (d: any) => d.source.x)
        .attr("y1", (d: any) => d.source.y)
        .attr("x2", (d: any) => d.target.x)
        .attr("y2", (d: any) => d.target.y);
      node.attr("cx", (d: any) => d.x).attr("cy", (d: any) => d.y);
    });

    return () => simulation.stop();
  }, [graphData, height, onNodeSelect, width]);

  return (
    <div className="grid gap-3 tablet:grid-cols-[2fr_1fr]">
      <svg ref={svgRef} width="100%" height={height} viewBox={`0 0 ${width} ${height}`} className="rounded-panel border border-line bg-surface" aria-label="Brain connectome force graph" />
      <aside className="rounded-panel border border-line bg-elevated p-3 text-sm">
        <h3 className="font-semibold">Region detail</h3>
        {selected ? (
          <div className="mt-2 space-y-1 text-muted">
            <p>ID: {selected.id}</p>
            <p>Risk: {(selected.risk * 100).toFixed(1)}%</p>
            <p>Structural metric: {selected.structuralMetric.toFixed(3)}</p>
          </div>
        ) : (
          <p className="mt-2 text-muted">Select a node for details.</p>
        )}
      </aside>
    </div>
  );
}
