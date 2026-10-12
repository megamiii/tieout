import { useEffect, useMemo, useState } from 'react'
import { traces, type Trace, type TraceSpan } from '../api'

const TYPE_COLOR: Record<string, string> = {
  agent: 'var(--muted)',
  workflow: 'var(--muted)',
  step: 'var(--border)',
  llm: 'var(--accent)',
  tool: 'var(--ok)',
  retrieval: '#8b5cf6',
  validation: 'var(--warn)',
  adjudication: 'var(--warn)',
  explain: 'var(--warn)',
}

function depthOf(spans: TraceSpan[]): Map<string, number> {
  const byId = new Map(spans.map((s) => [s.span_id, s]))
  const depth = new Map<string, number>()
  for (const s of spans) {
    let d = 0
    let p = s.parent_id
    while (p && byId.has(p)) {
      d += 1
      p = byId.get(p)?.parent_id ?? null
    }
    depth.set(s.span_id, d)
  }
  return depth
}

function Json({ value }: { value: unknown }) {
  if (value === null || value === undefined) return <span className="muted">—</span>
  return <pre className="passage" style={{ maxHeight: 260 }}>{JSON.stringify(value, null, 2)}</pre>
}

function LlmInput({ input }: { input: unknown }) {
  const data = input as { new_messages?: { role: string; content?: string }[]; conversation_offset?: number } | null
  if (!data?.new_messages) return <Json value={input} />
  return (
    <div className="small stack">
      <div className="muted">
        {data.conversation_offset ? `${data.conversation_offset} earlier messages not repeated · ` : ''}
        {data.new_messages.length} new message(s):
      </div>
      {data.new_messages.map((m, i) => (
        <div key={i}>
          <span className="badge pending">{m.role}</span>{' '}
          <span>{(m.content ?? '').slice(0, 600) || <span className="muted">(tool call)</span>}</span>
        </div>
      ))}
    </div>
  )
}

export function TraceView({ traceId }: { traceId: string }) {
  const [trace, setTrace] = useState<Trace | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [open, setOpen] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    traces.get(traceId).then(
      (t) => {
        if (!cancelled) setTrace(t)
      },
      (e: unknown) => setError(e instanceof Error ? e.message : String(e)),
    )
    return () => {
      cancelled = true
    }
  }, [traceId])

  const depth = useMemo(() => (trace ? depthOf(trace.spans) : new Map<string, number>()), [trace])

  if (error) return <div className="error">{error}</div>
  if (!trace) return <p className="muted small">Loading trace…</p>
  const total = Math.max(trace.duration_ms, ...trace.spans.map((s) => s.start_ms + s.duration_ms), 1)
  const t = trace.totals
  return (
    <div className="stack">
      <p className="small muted" style={{ margin: 0 }}>
        {(trace.duration_ms / 1000).toFixed(1)} s · {t.llm_requests} LLM requests
        {t.cached ? ` (+${t.cached} from cache)` : ''} · {(t.prompt_tokens + t.completion_tokens).toLocaleString()} tokens ·{' '}
        {t.tool_calls} tool calls{t.errors ? ` · ${t.errors} errors` : ''} · model {trace.model} · free tier: $0
      </p>
      <div className="trace-grid">
        {trace.spans.map((s) => {
          const tokens = s.prompt_tokens + s.completion_tokens
          const isOpen = open === s.span_id
          return (
            <div key={s.span_id}>
              <button
                className="trace-row"
                aria-expanded={isOpen}
                onClick={() => setOpen(isOpen ? null : s.span_id)}
              >
                <span className="trace-name" style={{ paddingLeft: (depth.get(s.span_id) ?? 0) * 14 }}>
                  <span className="trace-dot" style={{ background: TYPE_COLOR[s.type] ?? 'var(--muted)' }} />
                  <span className="trace-label" title={s.name}>
                    <span className="muted">{s.type}</span> {s.name}
                  </span>
                  {s.error && <span className="badge needs_review">error</span>}
                  {(s.cached || s.meta.cached === true) && <span className="badge accept">cached</span>}
                </span>
                <span className="trace-track">
                  <span
                    className="trace-bar"
                    style={{
                      left: `${(s.start_ms / total) * 100}%`,
                      width: `${Math.max((s.duration_ms / total) * 100, 0.4)}%`,
                      background: TYPE_COLOR[s.type] ?? 'var(--muted)',
                    }}
                  />
                </span>
                <span className="trace-meta mono">
                  {s.duration_ms >= 1000 ? `${(s.duration_ms / 1000).toFixed(1)} s` : `${Math.round(s.duration_ms)} ms`}
                  {tokens > 0 && ` · ${tokens.toLocaleString()} tok`}
                </span>
              </button>
              {isOpen && (
                <div className="trace-detail stack">
                  {s.error && <div className="error small">{s.error}</div>}
                  <div className="small">
                    <strong>Input</strong>
                    {s.type === 'llm' ? <LlmInput input={s.input} /> : <Json value={s.input} />}
                  </div>
                  <div className="small">
                    <strong>Output</strong>
                    <Json value={s.output} />
                  </div>
                  {Object.keys(s.meta).length > 0 && (
                    <div className="small">
                      <strong>Metadata</strong>
                      <Json value={s.meta} />
                    </div>
                  )}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
