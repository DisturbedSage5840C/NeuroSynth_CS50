// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { connectomeData } from '../data/mock-data';
import { Network } from 'lucide-react';
import type { AnalysisResult } from '../types/analysis';

interface ConnectomeGraphProps {
  analysisResult?: AnalysisResult | null;
}

export function ConnectomeGraph({ analysisResult }: ConnectomeGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;
    if (!analysisResult) {
      d3.select(svgRef.current).selectAll('*').remove();
      return;
    }
    const causalEdges = analysisResult?.causal_graph?.edges;
    const fromCausal = Array.isArray(causalEdges) && causalEdges.length > 0;
    const nodes = fromCausal
      ? Array.from(
          new Set(causalEdges.flatMap((e) => [String(e.from), String(e.to)]))
        ).map((id, idx) => ({
          id,
          label: id,
          region: 'causal',
          activity: 0.3 + ((idx % 7) / 10),
        }))
      : connectomeData.nodes;
    const edges = fromCausal
      ? causalEdges.map((e) => ({
          source: String(e.from),
          target: String(e.to),
          weight: Number(e.strength || 0.2),
        }))
      : connectomeData.edges;
    const width = svgRef.current.clientWidth || 400;
    const height = 320;

    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    const colorScale = d3.scaleSequential(d3.interpolateRdYlGn)
      .domain([1, 0]);

    const simulation = d3.forceSimulation(nodes as d3.SimulationNodeDatum[])
      .force('link', d3.forceLink(edges)
        .id((d: any) => d.id)
        .strength((d: any) => d.weight * 0.3))
      .force('charge', d3.forceManyBody().strength(-80))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide(18));

    const link = svg.append('g')
      .selectAll('line')
      .data(edges)
      .join('line')
      .attr('stroke', 'rgba(129,140,248,0.25)')
      .attr('stroke-width', (d: any) => d.weight * 2);

    const node = svg.append('g')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .attr('cursor', 'pointer')
      .call(d3.drag<any, any>()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        }));

    node.append('circle')
      .attr('r', (d: any) => 8 + d.activity * 6)
      .attr('fill', (d: any) => colorScale(d.activity))
      .attr('stroke', 'rgba(255,255,255,0.2)')
      .attr('stroke-width', 1);

    node.append('text')
      .text((d: any) => d.label)
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('fill', '#e4e4e7')
      .attr('font-size', '9px')
      .attr('pointer-events', 'none');

    node.append('title').text((d: any) => `${d.label}\nRegion: ${d.region}\nActivity: ${d.activity.toFixed(2)}`);

    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);
      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });

    return () => simulation.stop();
  }, [analysisResult]);

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-foreground">Brain Connectome</h3>
        <span className="text-xs text-muted-foreground">D3 Force Graph · Drag nodes</span>
      </div>
      {!analysisResult ? (
        <div className="flex h-[320px] flex-col items-center justify-center gap-3">
          <div className="w-12 h-12 rounded-full border-2 border-dashed border-border flex items-center justify-center">
            <Network size={20} className="text-muted-foreground" />
          </div>
          <div className="text-center">
            <div className="text-sm font-medium text-foreground">No connectome data yet</div>
            <div className="text-xs text-muted-foreground mt-0.5">Run analysis to infer brain network patterns</div>
          </div>
        </div>
      ) : (
        <>
          <svg ref={svgRef} className="w-full" style={{ height: 320 }} />
          <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
            <span style={{ color: 'var(--risk-low)' }}>● Low activity</span>
            <span style={{ color: 'var(--risk-high)' }}>● High activity</span>
            <span>Edge opacity = connectivity strength</span>
          </div>
        </>
      )}
    </div>
  );
}
