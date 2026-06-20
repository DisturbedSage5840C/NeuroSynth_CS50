// AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import { useState } from 'react';
import { Search, BookOpen, ExternalLink, Loader2 } from 'lucide-react';
import { GlassCard } from './v3/GlassCard';

const API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

interface AbstractResult {
  pmid: string;
  title: string;
  abstract: string;
  journal: string;
  pub_year: number | null;
  diseases: string[];
  similarity: number | null;
}

const DISEASE_COLORS: Record<string, string> = {
  alzheimer: '#818cf8', parkinson: '#34d399',
  multiple_sclerosis: '#fb923c', epilepsy: '#a78bfa',
  als: '#f87171', huntington: '#fbbf24',
};

const EXAMPLE_QUERIES = [
  'APOE4 Alzheimer risk prediction biomarker',
  'UPDRS motor score Parkinson progression',
  'neurofilament light chain ALS prognosis',
  'hippocampal atrophy MRI cognitive decline',
];

export function LiteratureSearch() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<AbstractResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [ragEnabled, setRagEnabled] = useState<boolean | null>(null);

  const search = async (q: string) => {
    if (!q.trim()) return;
    setLoading(true);
    setError('');
    try {
      const res = await fetch(`${API}/v3/literature/search`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, top_k: 10 }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setResults(data.results ?? []);
      setRagEnabled(data.rag_enabled ?? false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') search(query);
  };

  return (
    <div className="page-root">
      <div className="page-header">
        <h1 className="page-title">Literature Search</h1>
        <p className="page-subtitle">
          pgvector similarity search over 10,000 PubMed neurology abstracts
        </p>
      </div>

      {/* Search bar */}
      <GlassCard className="mb-6">
        <div className="flex gap-3">
          <div className="relative flex-1">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
            <input
              type="text"
              className="w-full bg-transparent border border-border rounded-lg pl-9 pr-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-primary transition-colors"
              placeholder="Search neurology literature…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={onKey}
            />
          </div>
          <button
            type="button"
            onClick={() => search(query)}
            disabled={loading || !query.trim()}
            className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-primary text-primary-foreground text-sm font-semibold disabled:opacity-50 hover:opacity-90 transition-opacity"
          >
            {loading ? <Loader2 size={14} className="animate-spin" /> : <Search size={14} />}
            Search
          </button>
        </div>

        {/* Example queries */}
        <div className="mt-3 flex flex-wrap gap-2">
          {EXAMPLE_QUERIES.map((q) => (
            <button
              key={q}
              type="button"
              onClick={() => { setQuery(q); search(q); }}
              className="text-xs px-2.5 py-1 rounded-md text-muted-foreground border border-border hover:border-primary hover:text-primary transition-colors"
            >
              {q}
            </button>
          ))}
        </div>
      </GlassCard>

      {/* Status */}
      {ragEnabled === false && (
        <div className="mb-4 px-4 py-3 rounded-lg bg-yellow-500/10 border border-yellow-500/20 text-sm text-yellow-300">
          RAG pipeline not active — set OPENAI_API_KEY and run embed_corpus.py to enable search.
        </div>
      )}
      {error && (
        <div className="mb-4 px-4 py-3 rounded-lg bg-red-500/10 border border-red-500/20 text-sm text-red-300">
          {error}
        </div>
      )}

      {/* Results */}
      {results.length > 0 && (
        <div className="space-y-3">
          <div className="text-xs text-muted-foreground mb-2">{results.length} abstracts found</div>
          {results.map((r, i) => (
            <GlassCard key={r.pmid}>
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2 mb-1 flex-wrap">
                    <span className="text-xs font-mono text-primary">PMID{r.pmid}</span>
                    {r.pub_year && (
                      <span className="text-xs text-muted-foreground">{r.pub_year}</span>
                    )}
                    {r.journal && (
                      <span className="text-xs text-muted-foreground italic">{r.journal}</span>
                    )}
                    {r.diseases.map((d) => (
                      <span
                        key={d}
                        className="text-xs px-1.5 py-0.5 rounded font-medium"
                        style={{
                          background: `${DISEASE_COLORS[d] ?? '#64748b'}18`,
                          color: DISEASE_COLORS[d] ?? '#64748b',
                        }}
                      >
                        {d}
                      </span>
                    ))}
                    {r.similarity != null && (
                      <span className="text-xs font-mono text-muted-foreground ml-auto">
                        {(r.similarity * 100).toFixed(1)}% match
                      </span>
                    )}
                  </div>
                  <h3 className="text-sm font-semibold text-foreground mb-2 leading-snug">{r.title}</h3>
                  <p className="text-xs text-muted-foreground leading-relaxed line-clamp-3">{r.abstract}</p>
                </div>
                <a
                  href={`https://pubmed.ncbi.nlm.nih.gov/${r.pmid}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex-shrink-0 p-1.5 rounded-lg text-muted-foreground hover:text-primary hover:bg-primary/10 transition-colors"
                  title="Open in PubMed"
                >
                  <ExternalLink size={14} />
                </a>
              </div>
              <div className="mt-2 flex items-center gap-1 text-xs text-muted-foreground">
                <BookOpen size={11} />
                <span>#{i + 1} · cite as [PMID{r.pmid}]</span>
              </div>
            </GlassCard>
          ))}
        </div>
      )}

      {!loading && results.length === 0 && query && !error && (
        <div className="text-center py-16 text-sm text-muted-foreground">No results found.</div>
      )}
    </div>
  );
}
