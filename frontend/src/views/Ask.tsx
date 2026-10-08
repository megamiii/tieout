import { useEffect, useState } from 'react'
import { api, type Company, type RunDetail } from '../api'
import { RunView } from '../components/RunView'

const EXAMPLES = [
  "What was 3M's FY2018 capital expenditure in USD millions?",
  'Does Corning have positive working capital based on FY2022 data?',
  'By what percentage did Nike’s revenue change from FY2022 to FY2023?',
]

export function AskView() {
  const [question, setQuestion] = useState('')
  const [run, setRun] = useState<RunDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [companies, setCompanies] = useState<Company[]>([])

  useEffect(() => {
    api.companies().then(setCompanies, () => setCompanies([]))
  }, [])

  async function submit(q: string) {
    setLoading(true)
    setError(null)
    setRun(null)
    try {
      setRun(await api.ask(q))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="stack">
      <div>
        <h1>Ask about a 10-K</h1>
        <p className="muted">
          The agent searches the filing text, reads XBRL financial data, computes with a calculator, and
          must cite every source. Uncertain answers go to the review queue.
        </p>
      </div>

      <form
        className="card stack"
        onSubmit={(e) => {
          e.preventDefault()
          if (question.trim().length >= 5) void submit(question.trim())
        }}
      >
        <textarea
          className="field"
          rows={3}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. What was 3M's FY2018 capital expenditure in USD millions?"
          aria-label="Question"
        />
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <div className="row small">
            {EXAMPLES.map((ex) => (
              <button type="button" key={ex} className="btn small" onClick={() => setQuestion(ex)}>
                {ex.length > 48 ? `${ex.slice(0, 48)}…` : ex}
              </button>
            ))}
          </div>
          <button className="btn primary" type="submit" disabled={loading || question.trim().length < 5}>
            {loading ? 'Working… (up to a minute)' : 'Ask'}
          </button>
        </div>
        {companies.length > 0 && (
          <p className="small muted">
            Supported: {companies.map((c) => `${c.ticker} (FY${c.fiscal_years.join(', ')})`).join(' · ')}
          </p>
        )}
      </form>

      {error && <div className="error">{error}</div>}
      {run && <RunView run={run} />}
    </div>
  )
}
