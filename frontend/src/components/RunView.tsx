import { useEffect, useRef } from 'react'
import Markdown from 'react-markdown'
import type { Citation, RunDetail, Segment, TraceStep } from '../api'

export function StatusBadge({ status }: { status: string }) {
  const label = status === 'needs_review' ? 'needs review' : status
  return <span className={`badge ${status}`}>{label}</span>
}

function Passage({ segments }: { segments: Segment[] }) {
  const ref = useRef<HTMLParagraphElement>(null)
  // Scroll the passage (not the page) so the first highlight is visible.
  useEffect(() => {
    const box = ref.current
    const first = box?.querySelector('mark')
    if (box && first instanceof HTMLElement) box.scrollTop = Math.max(0, first.offsetTop - box.offsetTop - 40)
  }, [segments])
  return (
    <p className="passage" ref={ref}>
      {segments.map((s, i) => (s.mark ? <mark key={i}>{s.text}</mark> : <span key={i}>{s.text}</span>))}
    </p>
  )
}

function Quotes({ quotes }: { quotes: string[] }) {
  return (
    <ul className="quotes">
      {quotes.map((q, i) => (
        <li key={i}>“{q}”</li>
      ))}
    </ul>
  )
}

function CitationCard({ c }: { c: Citation }) {
  if (!c.found) {
    return (
      <div className="citation">
        <header>
          <span className="mono">{c.source_id}</span>
          <span className="muted">source not found</span>
        </header>
        <Quotes quotes={c.quotes} />
      </div>
    )
  }
  if (c.kind === 'fact') {
    const period = c.period_start ? `${c.period_start} – ${c.period_end}` : `as of ${c.period_end}`
    return (
      <div className="citation">
        <header>
          <span>
            <strong>XBRL fact</strong> · {c.ticker} · {c.label}
          </span>
          <span className="mono muted">{c.source_id}</span>
        </header>
        <div className="fact">
          <span className="muted">Concept</span>
          <span className="mono">{c.concept}</span>
          <span className="muted">Value</span>
          <span className="mono">
            {Number(c.value).toLocaleString()} {c.unit}
          </span>
          <span className="muted">Period</span>
          <span>{period}</span>
          <span className="muted">Reported in</span>
          <span>
            10-K filed {c.filed} <span className="mono muted">({c.accession})</span>
          </span>
        </div>
      </div>
    )
  }
  const unmatched = c.quotes.filter((_, i) => c.matches[i] === 'none')
  return (
    <div className="citation">
      <header>
        <span>
          <strong>
            {c.ticker} 10-K FY{c.fiscal_year}
          </strong>{' '}
          · Item {c.section} · page {c.page}
          {c.quotes.length > 1 && <span className="muted"> · {c.quotes.length} quotes</span>}
        </span>
        <a href={c.edgar_url} target="_blank" rel="noreferrer" className="mono">
          {c.source_id}
        </a>
      </header>
      <Passage segments={c.segments} />
      {unmatched.length > 0 && (
        <div className="small muted" style={{ padding: '0 12px 8px' }}>
          Not found in this passage: <Quotes quotes={unmatched} />
        </div>
      )}
    </div>
  )
}

function Trace({ steps }: { steps: TraceStep[] }) {
  if (steps.length === 0) return <p className="muted small">No agent trace (single-call baseline).</p>
  return (
    <ol className="trace">
      {steps.map((s) => {
        const tokens = s.tokens ?? (s.prompt_tokens ?? 0) + (s.completion_tokens ?? 0)
        const flags = [s.forced_submit && 'forced submit', s.nudged && 'plain text, nudged', s.cached && 'cached']
          .filter(Boolean)
          .join(' · ')
        return (
          <li key={s.step}>
            <div className="small muted">
              Step {s.step + 1} · {tokens.toLocaleString()} tokens
              {s.latency_ms !== undefined && !s.cached && ` · ${(s.latency_ms / 1000).toFixed(1)} s`}
              {flags && ` · ${flags}`}
            </div>
            {s.tools.length === 0 && <div className="small">(no tool call)</div>}
            {s.tools.map((t, i) => (
              <div key={i} className={`tool ${t.error ? 'err' : ''}`}>
                {t.name}({t.arguments.length > 160 ? `${t.arguments.slice(0, 160)}…` : t.arguments})
                {t.error ? ` → ${t.error}` : t.sources ? ` → ${t.sources.length} sources` : ''}
              </div>
            ))}
          </li>
        )
      })}
    </ol>
  )
}

export function RunView({ run }: { run: RunDetail }) {
  const u = run.usage
  return (
    <div className="stack">
      <section className="card">
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <StatusBadge status={run.status} />
          <span className="small muted">
            {run.confidence !== null && `confidence ${run.confidence.toFixed(2)} · `}
            {u.requests} LLM requests · {(u.prompt_tokens + u.completion_tokens).toLocaleString()} tokens
            {u.latency_ms ? ` · ${(u.latency_ms / 1000).toFixed(1)} s` : ''}
          </span>
        </div>
        <p className="small muted" style={{ marginTop: 12 }}>
          {run.question}
        </p>
        {/* Model output is Markdown; react-markdown never renders raw HTML. */}
        <div className="answer">{run.answer ? <Markdown>{run.answer}</Markdown> : <p className="muted">No answer.</p>}</div>
        {run.review_reasons.length > 0 && (
          <ul className="reasons">
            {[...new Set(run.review_reasons)].map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        )}
        {run.warnings.length > 0 && (
          <ul className="reasons">
            {run.warnings.map((w, i) => (
              <li key={i}>Warning: {w}</li>
            ))}
          </ul>
        )}
        {run.rationale && (
          <div className="small">
            <strong>Rationale</strong>
            <Markdown>{run.rationale}</Markdown>
          </div>
        )}
      </section>

      <section className="card stack">
        <h3>Citations ({run.citations.length})</h3>
        {run.citations.length === 0 && <p className="muted small">No citations.</p>}
        {run.citations.map((c, i) => (
          <CitationCard key={`${c.source_id}-${i}`} c={c} />
        ))}
      </section>

      <section className="card">
        <h3>Agent trace</h3>
        <Trace steps={run.trace} />
        <p className="small muted">
          Model {run.model} · stop reason: {run.stop_reason}
        </p>
      </section>
    </div>
  )
}
