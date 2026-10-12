import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  api,
  workflows,
  type Company,
  type ReviewItem,
  type TieoutItem,
  type WorkflowKind,
  type WorkflowRun,
} from '../api'
import { TraceView } from '../components/TraceView'
import { useConfig } from '../config'
import { hashParams, replaceHash } from '../route'

const money = (v: string) => {
  const n = Math.abs(Number(v))
  return n >= 1e9 ? `$${(n / 1e9).toFixed(2)}B` : `$${(n / 1e6).toLocaleString(undefined, { maximumFractionDigits: 1 })}M`
}

const STATUS_ORDER: TieoutItem['status'][] = ['mismatched', 'matched', 'unmatched', 'out_of_scope']
const STATUS_LABEL: Record<string, string> = {
  matched: 'matched',
  mismatched: 'exception',
  unmatched: 'not in XBRL',
  out_of_scope: 'out of scope',
  explained: 'explained',
  not_explained: 'no explanation found',
  unverified: 'quote not verified',
}

function Sentence({ item }: { item: TieoutItem }) {
  const i = item.sentence.indexOf(item.amount)
  if (i < 0) return <>{item.sentence}</>
  return (
    <>
      {item.sentence.slice(0, i)}
      <mark>{item.amount}</mark>
      {item.sentence.slice(i + item.amount.length)}
    </>
  )
}

function TieoutTable({ items }: { items: TieoutItem[] }) {
  const [show, setShow] = useState<TieoutItem['status']>('mismatched')
  const counts = Object.fromEntries(STATUS_ORDER.map((s) => [s, items.filter((i) => i.status === s).length]))
  const rows = items.filter((i) => i.status === show)
  return (
    <div className="stack">
      <div className="row">
        {STATUS_ORDER.map((s) => (
          <button key={s} className={`btn small ${s === show ? 'primary' : ''}`} onClick={() => setShow(s)}>
            {STATUS_LABEL[s]} ({counts[s]})
          </button>
        ))}
      </div>
      {show === 'mismatched' && (
        <p className="small muted">
          Exceptions are claims whose figure differs from the same XBRL line item (identified by a value-blind LLM check).
          They need a reviewer: on five audited 10-Ks, every exception reviewed was a scope or definition difference
          (segment vs total, net vs gross), not an error.
        </p>
      )}
      {rows.length === 0 && <p className="muted">None.</p>}
      {rows.map((it, k) => (
        <div className="citation" key={`${it.chunk_id}-${k}`}>
          <header>
            <span>
              <strong>{it.amount}</strong> · FY{it.year ?? '?'} · page {it.page}{' '}
              <span className={`badge ${it.status === 'matched' ? 'answered' : 'needs_review'}`}>
                {STATUS_LABEL[it.status]}
              </span>
            </span>
            <span className="mono muted">{it.chunk_id}</span>
          </header>
          <div style={{ padding: '10px 12px' }} className="stack">
            <div className="small">
              <Sentence item={it} />
            </div>
            {it.fact && (
              <div className="small">
                XBRL: <span className="mono">{it.fact.concept}</span> = <strong>{money(it.fact.value)}</strong> (
                {it.fact.period}) <span className="mono muted">{it.fact.fact_id}</span>
              </div>
            )}
            {(it.note || it.adjudication) && (
              <div className="small muted">
                {it.note}
                {it.adjudication && <> · LLM: {it.adjudication}</>}
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}

function ReviewTable({ items }: { items: ReviewItem[] }) {
  return (
    <div className="stack">
      {items.map((it) => (
        <div className="citation" key={it.concept}>
          <header>
            <span>
              <strong>{it.label}</strong>{' '}
              <span className="mono muted">{it.concept}</span>
            </span>
            <span className={`badge ${it.explanation.status === 'explained' ? 'answered' : 'needs_review'}`}>
              {STATUS_LABEL[it.explanation.status]}
            </span>
          </header>
          <div style={{ padding: '10px 12px' }} className="stack small">
            <div>
              FY{it.prior.fiscal_year} {money(it.prior.value)} → FY{it.current.fiscal_year} {money(it.current.value)} (
              <strong>
                {Number(it.pct) > 0 ? '+' : ''}
                {it.pct}%
              </strong>
              )
            </div>
            {it.explanation.quote && (
              <blockquote style={{ margin: 0 }}>
                <mark>“{it.explanation.quote}”</mark>{' '}
                <span className="muted">
                  (p.{it.explanation.page ?? '?'}, <span className="mono">{it.explanation.chunk_id}</span>)
                </span>
              </blockquote>
            )}
            <div className="muted">{it.explanation.summary}</div>
          </div>
        </div>
      ))}
    </div>
  )
}

export function WorkflowView({ kind }: { kind: WorkflowKind }) {
  const [companies, setCompanies] = useState<Company[]>([])
  const [ticker, setTicker] = useState('')
  const [year, setYear] = useState<number | ''>('')
  const [runs, setRuns] = useState<WorkflowRun[]>([])
  const [current, setCurrent] = useState<WorkflowRun | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const loadRuns = useCallback(() => {
    workflows.list(kind).then(setRuns, (e: unknown) => setError(String(e)))
  }, [kind])

  useEffect(() => {
    api.companies().then(setCompanies, () => setCompanies([]))
    loadRuns()
  }, [loadRuns])

  const years = useMemo(() => companies.find((c) => c.ticker === ticker)?.fiscal_years ?? [], [companies, ticker])

  // Linked run (#/tieout?ticker=MMM&year=2018): open the latest stored run for that filing.
  useEffect(() => {
    const params = hashParams()
    const hit = runs.find(
      (r) => r.ticker === params.get('ticker') && r.fiscal_year === Number(params.get('year')),
    )
    if (hit && !current) workflows.get(hit.id).then(setCurrent, (e: unknown) => setError(String(e)))
  }, [runs, current])

  // Keep the address shareable for the run on screen.
  useEffect(() => {
    if (current) {
      replaceHash(kind === 'tieout' ? 'tieout' : 'analytical', {
        ticker: current.ticker,
        year: String(current.fiscal_year),
      })
    }
  }, [current, kind])

  async function run() {
    if (!ticker || !year) return
    setBusy(true)
    setError(null)
    try {
      setCurrent(await workflows.run(kind, ticker, year))
      loadRuns()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const s = current?.summary
  const demo = useConfig()?.demo_mode ?? false
  const recorded = [...new Set(runs.map((r) => `${r.ticker} FY${r.fiscal_year}`))].sort()
  return (
    <div className="stack">
      <div>
        <h1>{kind === 'tieout' ? 'Tie-out' : 'Analytical review'}</h1>
        <p className="muted">
          {kind === 'tieout'
            ? 'Checks every dollar figure stated in the MD&A against the XBRL financial data the company filed in the same 10-K.'
            : 'Finds line items with large year-over-year changes in XBRL and looks for management’s explanation in the MD&A.'}
        </p>
      </div>
      <div className="card row">
        <select className="field" style={{ width: 'auto' }} value={ticker} onChange={(e) => { setTicker(e.target.value); setYear('') }} aria-label="Company">
          <option value="">Company…</option>
          {companies.map((c) => (
            <option key={c.ticker} value={c.ticker}>
              {c.ticker} · {c.name}
            </option>
          ))}
        </select>
        <select className="field" style={{ width: 'auto' }} value={year} onChange={(e) => setYear(Number(e.target.value))} aria-label="Fiscal year" disabled={!ticker}>
          <option value="">Fiscal year…</option>
          {years.map((y) => (
            <option key={y} value={y}>
              FY{y}
            </option>
          ))}
        </select>
        <button className="btn primary" disabled={busy || !ticker || !year} onClick={() => void run()}>
          {busy ? 'Running… (up to a few minutes)' : 'Run'}
        </button>
        {runs.length > 0 && (
          <select
            className="field"
            style={{ width: 'auto', marginLeft: 'auto' }}
            value={current?.id ?? ''}
            onChange={(e) => e.target.value && workflows.get(e.target.value).then(setCurrent, (err: unknown) => setError(String(err)))}
            aria-label="Previous runs"
          >
            <option value="">Previous runs…</option>
            {runs.map((r) => (
              <option key={r.id} value={r.id}>
                {r.ticker} FY{r.fiscal_year} · {new Date(r.created_at).toLocaleDateString()}
              </option>
            ))}
          </select>
        )}
      </div>
      {demo && recorded.length > 0 && (
        <p className="small muted" style={{ margin: 0 }}>
          Recorded in this demo: {recorded.join(' · ')}. Other filings need a local install with a free API key.
        </p>
      )}
      {error && <div className="error">{error}</div>}
      {current && s && (
        <section className="card stack">
          <h2>
            {current.ticker} 10-K FY{current.fiscal_year}
          </h2>
          <p className="small muted">
            {kind === 'tieout'
              ? `${s.claims} dollar claims: ${s.matched} matched, ${s.mismatched} exceptions, ${s.unmatched} not in XBRL, ${s.out_of_scope} out of scope (quarterly or coarsely rounded).`
              : `${s.material_changes} material changes (≥20% and ≥1% of ${String(s.materiality_base)}): ${s.explained} explained, ${s.not_explained} without an explanation, ${s.unverified} unverified.`}{' '}
            {String(s.llm_requests)} LLM requests · model {current.model}
          </p>
          {kind === 'tieout' ? (
            <TieoutTable items={(current.items ?? []) as TieoutItem[]} />
          ) : (
            <ReviewTable items={(current.items ?? []) as ReviewItem[]} />
          )}
        </section>
      )}
      {current?.trace_id && (
        <details className="card">
          <summary>
            <strong>Trace</strong> <span className="small muted">every LLM call, search and check in this run</span>
          </summary>
          <div style={{ marginTop: 12 }}>
            <TraceView traceId={current.trace_id} />
          </div>
        </details>
      )}
    </div>
  )
}
