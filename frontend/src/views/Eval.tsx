import { useEffect, useState } from 'react'
import { api, type EvalMetrics, type EvalReport } from '../api'

// Archived experiments are kept for comparison and never resumed.
const isArchived = (labels: Record<string, string>, experiment: string) =>
  (labels[experiment] ?? '').includes('(archived)')

const pct = (x: number | null) => (x === null ? '–' : `${(x * 100).toFixed(1)}%`)
const num = (x: number | null, digits = 0) => (x === null ? '–' : x.toLocaleString(undefined, { maximumFractionDigits: digits }))

const SET_TITLES: Record<string, string> = {
  financebench: 'Set A · FinanceBench (70 questions on supported 10-Ks)',
  xbrl_numeric: 'Set B · XBRL numeric (40 generated questions, exact ground truth)',
}

function Table({ rows, labels }: { rows: EvalMetrics[]; labels: Record<string, string> }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Configuration</th>
            <th>Graded</th>
            <th>Accuracy</th>
            <th>Answered</th>
            <th>Acc. when answered</th>
            <th>Citation precision</th>
            <th>Abstention P / R</th>
            <th>Requests / q</th>
            <th>Tokens / q</th>
            <th>Latency p50 / p95</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((m) => (
            <tr key={m.experiment} className={m.graded < m.n ? 'partial' : ''}>
              <td>
                {labels[m.experiment] ?? m.experiment}
                {m.graded < m.n && (
                  <span className="badge pending" style={{ marginLeft: 8 }}>
                    {isArchived(labels, m.experiment) ? 'incomplete' : 'in progress'}
                  </span>
                )}
              </td>
              <td>
                {m.graded}/{m.n}
              </td>
              <td>{pct(m.accuracy)}</td>
              <td>{pct(m.answered_rate)}</td>
              <td>{pct(m.accuracy_answered)}</td>
              <td>
                {pct(m.citation_precision)} <span className="muted">({m.citations})</span>
              </td>
              <td>
                {pct(m.abstention_precision)} / {pct(m.abstention_recall)}
              </td>
              <td>{num(m.requests_per_item, 1)}</td>
              <td>{num(m.tokens_per_item)}</td>
              <td>
                {m.latency_p50_s === null ? '–' : `${m.latency_p50_s.toFixed(1)} s / ${(m.latency_p95_s ?? 0).toFixed(1)} s`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function EvalView() {
  const [report, setReport] = useState<EvalReport | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.evalReport().then(setReport, (e: unknown) => setError(e instanceof Error ? e.message : String(e)))
  }, [])

  if (error) return <div className="error">{error}</div>
  if (!report) return <p className="muted">Loading…</p>

  const sets = [...new Set(report.metrics.map((m) => m.eval_set))]
  const partial = report.metrics.some(
    (m) => m.graded < m.n && !isArchived(report.experiments, m.experiment),
  )
  return (
    <div className="stack">
      <div>
        <h1>Evaluation</h1>
        <p className="muted">
          Agent model {report.meta.agent_model} · judge {report.meta.judge_model} · report of {report.meta.date}.
          {partial
            ? ' Rows marked “in progress” are partial: the free tier allows 500 agent requests a day, so runs continue across days.'
            : ' Every question was answered and graded; CI replays all answers from the recordings on each push.'}
        </p>
      </div>
      {sets.map((s) => (
        <section className="card" key={s}>
          <h2>{SET_TITLES[s] ?? s}</h2>
          <Table rows={report.metrics.filter((m) => m.eval_set === s)} labels={report.experiments} />
        </section>
      ))}
      <section className="card small muted">
        <p style={{ margin: 0 }}>
          Accuracy counts abstentions as not correct. Citation precision: share of citations that point at the gold
          evidence. Abstention P / R: P(wrong | sent to review) and P(sent to review | wrong). Definitions and caveats are
          in the README.
        </p>
      </section>
    </div>
  )
}
