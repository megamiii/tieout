import { useEffect, useState } from 'react'
import { AskView } from './views/Ask'
import { EvalView } from './views/Eval'
import { ReviewView } from './views/Review'

const ROUTES = {
  ask: { label: 'Ask', view: AskView },
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
      <main>
        <View />
      </main>
    </>
  )
}
