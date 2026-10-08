import { useCallback, useEffect, useState } from 'react'
import { api, type Decision, type RunDetail, type RunSummary } from '../api'
import { RunView, StatusBadge } from '../components/RunView'

type Queue = 'pending' | 'reviewed' | 'all'

function ReviewForm({ run, onSaved }: { run: RunDetail; onSaved: (r: RunDetail) => void }) {
  const [decision, setDecision] = useState<Decision>('accept')
  const [override, setOverride] = useState(run.answer ?? '')
  const [note, setNote] = useState('')
  const [reviewer, setReviewer] = useState(() => localStorage.getItem('tieout.reviewer') ?? '')
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  async function save() {
    setSaving(true)
    setError(null)
    try {
      localStorage.setItem('tieout.reviewer', reviewer)
      const updated = await api.review(run.id, {
        decision,
        reviewer,
        note,
        override_answer: decision === 'override' ? override : undefined,
      })
      onSaved(updated)
      setNote('')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  return (
    <section className="card stack">
      <h3>Reviewer decision</h3>
      <div className="row" role="radiogroup" aria-label="Decision">
        <label className="row">
          <input type="radio" checked={decision === 'accept'} onChange={() => setDecision('accept')} />
          Accept the agent's answer
        </label>
        <label className="row">
          <input type="radio" checked={decision === 'override'} onChange={() => setDecision('override')} />
          Override with a corrected answer
        </label>
      </div>
      {decision === 'override' && (
        <textarea
          className="field"
          rows={3}
          value={override}
          onChange={(e) => setOverride(e.target.value)}
          aria-label="Corrected answer"
        />
      )}
      <textarea
        className="field"
        rows={2}
        value={note}
        onChange={(e) => setNote(e.target.value)}
        placeholder="Note (what you checked, why you overrode)"
        aria-label="Note"
      />
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <input
          className="field"
          style={{ maxWidth: 240 }}
          value={reviewer}
          onChange={(e) => setReviewer(e.target.value)}
          placeholder="Your name"
          aria-label="Reviewer"
        />
        <button
          className="btn primary"
          disabled={saving || !reviewer.trim() || (decision === 'override' && !override.trim())}
          onClick={() => void save()}
        >
          {saving ? 'Saving…' : 'Save decision'}
        </button>
      </div>
      {error && <div className="error">{error}</div>}
      {run.decisions.length > 0 && (
        <div className="small">
          <h3 style={{ marginTop: 8 }}>History</h3>
          {run.decisions
            .slice()
            .reverse()
            .map((d) => (
              <p key={d.id} style={{ margin: '4px 0' }}>
                <span className={`badge ${d.decision}`}>{d.decision}</span> by {d.reviewer} ·{' '}
                {new Date(d.created_at).toLocaleString()}
                {d.decision === 'override' && <> · final answer: “{d.final_answer}”</>}
                {d.note && <span className="muted"> · {d.note}</span>}
              </p>
            ))}
        </div>
      )}
    </section>
  )
}

export function ReviewView() {
  const [queue, setQueue] = useState<Queue>('pending')
  const [items, setItems] = useState<RunSummary[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [run, setRun] = useState<RunDetail | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(() => {
    api.runs(queue).then(
      (rows) => {
        setItems(rows)
        setError(null)
      },
      (e: unknown) => setError(e instanceof Error ? e.message : String(e)),
    )
  }, [queue])

  useEffect(load, [load])

  useEffect(() => {
    if (!selected) return
    let cancelled = false
    api.run(selected).then(
      (r) => {
        if (!cancelled) setRun(r)
      },
      (e: unknown) => setError(e instanceof Error ? e.message : String(e)),
    )
    return () => {
      cancelled = true
    }
  }, [selected])

  return (
    <div className="stack">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <div>
          <h1>Review queue</h1>
          <p className="muted">Answers the agent routed to a human: low confidence, conflicts or failed checks.</p>
        </div>
        <select className="field" style={{ width: 'auto' }} value={queue} onChange={(e) => setQueue(e.target.value as Queue)} aria-label="Queue">
          <option value="pending">Pending ({queue === 'pending' ? items.length : '…'})</option>
          <option value="reviewed">Reviewed</option>
          <option value="all">All needing review</option>
        </select>
      </div>
      {error && <div className="error">{error}</div>}
      <div className="split">
        <div className="card" style={{ padding: 0 }}>
          {items.length === 0 ? (
            <p className="muted" style={{ padding: 16 }}>
              Nothing here.
            </p>
          ) : (
            <ul className="queue">
              {items.map((r) => (
                <li key={r.id}>
                  <button aria-current={r.id === selected} onClick={() => setSelected(r.id)}>
                    <div className="row small" style={{ justifyContent: 'space-between' }}>
                      <span className="muted">{r.source}</span>
                      {r.review ? <span className={`badge ${r.review.decision}`}>{r.review.decision}</span> : <StatusBadge status="pending" />}
                    </div>
                    <div className="q">{r.question}</div>
                    <div className="small muted">{r.review_reasons[0] ?? ''}</div>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className="stack">
          {run ? (
            <>
              <RunView run={run} />
              <ReviewForm
                key={run.id}
                run={run}
                onSaved={(updated) => {
                  setRun(updated)
                  load()
                }}
              />
            </>
          ) : (
            <p className="muted card">Select a run to see its answer, evidence and trace.</p>
          )}
        </div>
      </div>
    </div>
  )
}
