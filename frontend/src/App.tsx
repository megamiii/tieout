import { useEffect, useState } from 'react'
import { useConfig } from './config'
import { AskView } from './views/Ask'
import { EvalView } from './views/Eval'
import { ReviewView } from './views/Review'
import { WorkflowView } from './views/Workflows'

const TieoutView = () => <WorkflowView kind="tieout" />
const AnalyticalView = () => <WorkflowView kind="analytical_review" />

const ROUTES = {
  ask: { label: 'Ask', view: AskView },
  tieout: { label: 'Tie-out', view: TieoutView },
  analytical: { label: 'Analytical review', view: AnalyticalView },
  review: { label: 'Review queue', view: ReviewView },
  eval: { label: 'Evaluation', view: EvalView },
} as const

type Route = keyof typeof ROUTES

function currentRoute(): Route {
  const hash = window.location.hash.replace('#/', '')
  return hash in ROUTES ? (hash as Route) : 'ask'
}

export default function App() {
  const [route, setRoute] = useState<Route>(currentRoute)

  useEffect(() => {
    const onHash = () => setRoute(currentRoute())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const config = useConfig()
  const View = ROUTES[route].view
  return (
    <>
      <nav className="topbar">
        <div className="brand">
          tieout<span>audit agent for 10-K filings</span>
        </div>
        <div className="tabs">
          {(Object.keys(ROUTES) as Route[]).map((r) => (
            <a key={r} href={`#/${r}`} aria-current={r === route ? 'page' : undefined}>
              {ROUTES[r].label}
            </a>
          ))}
        </div>
      </nav>
      {config?.demo_mode && (
        <div className="demo-banner small">
          <strong>Demo mode.</strong> Answers, tie-outs and reviews here are recorded runs replayed
          without an API key, so the demo is free and always works. To ask your own questions, run
          tieout locally with a free Gemini or Groq key.
        </div>
      )}
      <main>
        <View />
      </main>
    </>
  )
}
