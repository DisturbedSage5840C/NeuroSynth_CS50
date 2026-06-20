// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import './v3.css';

const API = import.meta.env.VITE_API_BASE_URL ?? '';

export interface LiteratureResult {
  pmid: string;
  title: string;
  abstract: string;
  journal?: string;
  pub_year?: number;
  similarity?: number;
}

interface LiteraturePanelProps {
  /** Pre-fill the query from current patient context (e.g. top disease + top SHAP feature). */
  initialQuery?: string;
  /** Max results to show. */
  topK?: number;
}

async function searchLiterature(query: string, topK: number): Promise<LiteratureResult[]> {
  const res = await fetch(`${API}/v3/literature/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: topK }),
  });
  if (!res.ok) throw new Error('search failed');
  const data = await res.json();
  // Normalise: backend returns {results:[...]} or [{...}]
  return Array.isArray(data) ? data : (data.results ?? []);
}

export function LiteraturePanel({ initialQuery = '', topK = 5 }: LiteraturePanelProps) {
  const [query, setQuery] = useState(initialQuery);
  const [submitted, setSubmitted] = useState(initialQuery.length > 0);

  const { data: results, isFetching, isError } = useQuery<LiteratureResult[]>({
    queryKey: ['literature', query, topK],
    queryFn: () => searchLiterature(query, topK),
    enabled: submitted && query.trim().length > 0,
    staleTime: 10 * 60 * 1000,
  });

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (query.trim()) setSubmitted(true);
  }

  return (
    <div className="lit-panel">
      <form className="lit-search-row" onSubmit={handleSubmit}>
        <input
          className="lit-input"
          type="text"
          value={query}
          onChange={(e) => { setQuery(e.target.value); setSubmitted(false); }}
          placeholder="e.g. amyloid beta Alzheimer treatment"
          aria-label="Literature search query"
        />
        <button type="submit" className="lit-search-btn" disabled={isFetching}>
          {isFetching ? '…' : 'Search'}
        </button>
      </form>

      {isError && (
        <p className="lit-error">RAG search unavailable — OpenAI key or pgvector not configured.</p>
      )}

      {results && results.length === 0 && !isFetching && (
        <p className="lit-empty">No results. Try different keywords.</p>
      )}

      <div className="lit-results">
        {(results ?? []).map((r) => (
          <article key={r.pmid} className="lit-card">
            <div className="lit-card-header">
              <a
                className="lit-card-title"
                href={`https://pubmed.ncbi.nlm.nih.gov/${r.pmid}`}
                target="_blank"
                rel="noopener noreferrer"
              >
                {r.title}
              </a>
              {r.similarity != null && (
                <span className="lit-sim-badge">
                  {Math.round(r.similarity * 100)}% match
                </span>
              )}
            </div>
            {r.journal && (
              <div className="lit-card-meta">
                {r.journal}{r.pub_year ? ` · ${r.pub_year}` : ''} · PMID {r.pmid}
              </div>
            )}
            <p className="lit-card-abstract">
              {r.abstract.length > 280 ? `${r.abstract.slice(0, 280)}…` : r.abstract}
            </p>
          </article>
        ))}
      </div>
    </div>
  );
}
