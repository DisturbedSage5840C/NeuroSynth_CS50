// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './v3.css';

export interface CausalNode {
  id: string;
  label: string;
  type?: 'cause' | 'effect' | 'mediator';
}

export interface CausalEdge {
  source: string;
  target: string;
  weight?: number;
  direction?: 'positive' | 'negative';
}

interface CytoscapeGraphProps {
  nodes: CausalNode[];
  edges: CausalEdge[];
  width?: number;
  height?: number;
}

const NODE_COLOR: Record<string, string> = {
  cause:    'var(--accent-primary)',
  effect:   'var(--color-als)',
  mediator: 'var(--color-ep)',
};

export function CytoscapeGraph({ nodes, edges, width = 420, height = 280 }: CytoscapeGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !nodes.length) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const sim = d3.forceSimulation(nodes as d3.SimulationNodeDatum[])
      .force('link', d3.forceLink(edges).id((d: d3.SimulationNodeDatum) => (d as CausalNode).id).distance(90))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collide', d3.forceCollide(32));

    svg.append('defs').append('marker')
      .attr('id', 'arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 22).attr('refY', 0)
      .attr('markerWidth', 6).attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', 'var(--text-tertiary)');

    const link = svg.append('g').selectAll('line')
      .data(edges).enter().append('line')
      .attr('stroke', (d) => d.direction === 'positive' ? 'var(--risk-low)' : d.direction === 'negative' ? 'var(--risk-high)' : 'var(--text-tertiary)')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', (d) => Math.max(1, (d.weight ?? 0.5) * 3))
      .attr('marker-end', 'url(#arrow)');

    const node = svg.append('g').selectAll('g')
      .data(nodes).enter().append('g')
      .call(d3.drag<SVGGElement, CausalNode>()
        .on('start', (event, d) => { if (!event.active) sim.alphaTarget(0.3).restart(); (d as d3.SimulationNodeDatum).fx = d3.pointer(event, svgRef.current)[0]; (d as d3.SimulationNodeDatum).fy = d3.pointer(event, svgRef.current)[1]; })
        .on('drag', (event, d) => { (d as d3.SimulationNodeDatum).fx = event.x; (d as d3.SimulationNodeDatum).fy = event.y; })
        .on('end', (event, d) => { if (!event.active) sim.alphaTarget(0); (d as d3.SimulationNodeDatum).fx = null; (d as d3.SimulationNodeDatum).fy = null; })
      );

    node.append('circle')
      .attr('r', 18)
      .attr('fill', (d) => NODE_COLOR[d.type ?? 'cause'] + '22')
      .attr('stroke', (d) => NODE_COLOR[d.type ?? 'cause'])
      .attr('stroke-width', 1.5);

    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('font-size', 9)
      .attr('fill', 'var(--text-secondary)')
      .attr('font-family', 'var(--font-mono)')
      .text((d) => d.label.length > 12 ? d.label.slice(0, 11) + '…' : d.label);

    sim.on('tick', () => {
      link
        .attr('x1', (d) => (d.source as d3.SimulationNodeDatum).x ?? 0)
        .attr('y1', (d) => (d.source as d3.SimulationNodeDatum).y ?? 0)
        .attr('x2', (d) => (d.target as d3.SimulationNodeDatum).x ?? 0)
        .attr('y2', (d) => (d.target as d3.SimulationNodeDatum).y ?? 0);
      node.attr('transform', (d) => `translate(${(d as d3.SimulationNodeDatum).x ?? 0},${(d as d3.SimulationNodeDatum).y ?? 0})`);
    });

    return () => { sim.stop(); };
  }, [nodes, edges, width, height]);

  if (!nodes.length) {
    return (
      <div className="cyto-empty">No causal graph available</div>
    );
  }

  return (
    <div className="cyto-root">
      <svg ref={svgRef} width={width} height={height} className="cyto-svg" />
      <div className="cyto-legend">
        <span className="cyto-leg-item cyto-leg-positive">positive effect</span>
        <span className="cyto-leg-item cyto-leg-negative">negative effect</span>
      </div>
    </div>
  );
}
