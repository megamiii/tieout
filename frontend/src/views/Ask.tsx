import { useEffect, useState } from 'react'
import { api, type Company, type RunDetail } from '../api'
import { RunView } from '../components/RunView'
import { useConfig } from '../config'
import { hashParams, replaceHash } from '../route'

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
  const config = useConfig()
  const demo = config?.demo_mode ?? false

  useEffect(() => {
    api.companies().then(setCompanies, () => setCompanies([]))
  }, [])

  // Open a linked recorded question (#/ask?q=<id>) once the config is known.
  const linked = config?.demo_questions.find((d) => d.id === hashParams().get('q'))
  useEffect(() => {
    if (linked) void submit(linked.question)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- run once per linked question
  }, [linked?.id])

  async function submit(q: string) {
    setQuestion(q)
    setLoading(true)
    setError(null)
    setRun(null)
    // A recorded question gets a shareable address: #/ask?q=<id>.
    const recorded = config?.demo_questions.find((d) => d.question === q)
    replaceHash('ask', recorded ? { q: recorded.id } : {})
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

      {demo && config && config.demo_questions.length > 0 && (
        <section className="card stack">
          <h3 style={{ margin: 0 }}>Recorded questions</h3>
          <div className="demo-questions">
            {config.demo_questions.map((q) => (
              <button
                key={q.id}
                type="button"
                className="demo-question"
                disabled={loading}
                onClick={() => void submit(q.question)}
              >
                <span>{q.question}</span>
                <span className="small muted">{q.shows}</span>
              </button>
            ))}
          </div>
        </section>
      )}

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
            {(demo ? [] : EXAMPLES).map((ex) => (
              <button type="button" key={ex} className="btn small" onClick={() => setQuestion(ex)}>
                {ex.length > 48 ? `${ex.slice(0, 48)}…` : ex}
              </button>
            ))}
          </div>
          <button className="btn primary" type="submit" disabled={loading || question.trim().length < 5}>
            {loading ? (demo ? 'Replaying…' : 'Working… (up to a minute)') : 'Ask'}
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
