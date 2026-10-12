// The static demo (GitHub Pages): no server. GET requests read JSON files that
// `tieout demo export-static` wrote from the real demo API; "Ask" and "Run"
// look up the recorded question or filing; review decisions are kept in this
// browser's localStorage, since there is no shared database to write to.

import type { Decision, ReviewRecord, RunDetail, RunSummary, UiConfig, WorkflowRun } from './api'
import { ApiError } from './errors'

const DATA = `${import.meta.env.BASE_URL}demo-data/`
const DECISIONS_KEY = 'tieout-demo-decisions'

type LocalDecision = ReviewRecord & { run_id: string }

async function file<T>(rel: string): Promise<T> {
  const res = await fetch(DATA + rel)
  if (!res.ok) throw new ApiError(404, 'This page is not part of the static demo.')
  return (await res.json()) as T
}

function loadDecisions(): LocalDecision[] {
  try {
    const raw = localStorage.getItem(DECISIONS_KEY)
    return raw ? (JSON.parse(raw) as LocalDecision[]) : []
  } catch {
    return [] // storage blocked (private mode, previews): decisions just don't persist
  }
}

function saveDecisions(decisions: LocalDecision[]): void {
  try {
    localStorage.setItem(DECISIONS_KEY, JSON.stringify(decisions))
  } catch {
    // ignore: see loadDecisions
  }
}

function localFor(runId: string): ReviewRecord[] {
  return loadDecisions()
    .filter((d) => d.run_id === runId)
    .map((d) => ({
      id: d.id,
      created_at: d.created_at,
      decision: d.decision,
      final_answer: d.final_answer,
      note: d.note,
      reviewer: d.reviewer,
    }))
}

function withLocal<T extends RunSummary>(run: T): T {
  const local = localFor(run.id)
  if (local.length === 0) return run
  const merged = { ...run, review: local[local.length - 1] }
  return 'decisions' in run
    ? { ...merged, decisions: [...(run as unknown as RunDetail).decisions, ...local] }
    : merged
}

async function runDetail(id: string): Promise<RunDetail> {
  return withLocal(await file<RunDetail>(`runs/${encodeURIComponent(id)}.json`))
}

async function queue(name: string): Promise<RunSummary[]> {
  const reviewedHere = new Set(loadDecisions().map((d) => d.run_id))
  if (name === 'pending') {
    const pending = await file<RunSummary[]>('runs/queue-pending.json')
    return pending.filter((r) => !reviewedHere.has(r.id))
  }
  if (name === 'reviewed') {
    const [reviewed, pending] = await Promise.all([
      file<RunSummary[]>('runs/queue-reviewed.json'),
      file<RunSummary[]>('runs/queue-pending.json'),
    ])
    return [...pending.filter((r) => reviewedHere.has(r.id)), ...reviewed].map(withLocal)
  }
  return (await file<RunSummary[]>('runs/queue-all.json')).map(withLocal)
}

async function notRecorded(): Promise<never> {
  const config = await file<UiConfig>('config.json')
  throw new ApiError(403, config.not_recorded ?? 'Not recorded in this demo.')
}

interface ReviewBody {
  decision: Decision
  reviewer: string
  note: string
  override_answer?: string
}

async function review(id: string, body: ReviewBody): Promise<RunDetail> {
  const run = await runDetail(id)
  const override = body.override_answer?.trim() ?? ''
  if (body.decision === 'override' && !override) {
    throw new ApiError(422, 'an override needs the corrected answer')
  }
  const record: LocalDecision = {
    run_id: id,
    id: Date.now(),
    created_at: new Date().toISOString(),
    decision: body.decision,
    final_answer: body.decision === 'override' ? override : (run.answer ?? ''),
    note: body.note,
    reviewer: body.reviewer,
  }
  saveDecisions([...loadDecisions(), record])
  return runDetail(id)
}

export async function staticRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const url = new URL(path, 'http://static.invalid')
  const p = url.pathname
  const body = init?.body ? (JSON.parse(String(init.body)) as Record<string, unknown>) : {}
  let m: RegExpMatchArray | null

  if ((init?.method ?? 'GET') === 'POST') {
    if (p === '/api/ask') {
      const asks = await file<Record<string, string>>('asks.json')
      const id = asks[String(body.question ?? '').trim()]
      return (id ? runDetail(id) : notRecorded()) as Promise<T>
    }
    if (p === '/api/workflows/tieout' || p === '/api/workflows/review') {
      const kind = p.endsWith('tieout') ? 'tieout' : 'analytical_review'
      const runs = await file<WorkflowRun[]>(`workflows/list-${kind}.json`)
      const ticker = String(body.ticker ?? '').toUpperCase()
      const hit = runs.find((r) => r.ticker === ticker && r.fiscal_year === Number(body.fiscal_year))
      return (hit ? file(`workflows/${hit.id}.json`) : notRecorded()) as Promise<T>
    }
    if ((m = p.match(/^\/api\/runs\/([^/]+)\/review$/))) {
      return review(decodeURIComponent(m[1]), body as unknown as ReviewBody) as Promise<T>
    }
    throw new ApiError(405, 'Not available in the static demo.')
  }

  if (p === '/api/config') return file<T>('config.json')
  if (p === '/api/companies') return file<T>('companies.json')
  if (p === '/api/eval') return file<T>('eval.json')
  if (p === '/api/runs') return queue(url.searchParams.get('queue') ?? 'pending') as Promise<T>
  if (p === '/api/workflows') return file<T>(`workflows/list-${url.searchParams.get('kind') ?? 'tieout'}.json`)
  if ((m = p.match(/^\/api\/runs\/([^/]+)$/))) return runDetail(decodeURIComponent(m[1])) as Promise<T>
  if ((m = p.match(/^\/api\/workflows\/([^/]+)$/))) return file<T>(`workflows/${m[1]}.json`)
  if ((m = p.match(/^\/api\/traces\/([^/]+)$/))) return file<T>(`traces/${m[1]}.json`)
  throw new ApiError(404, 'This page is not part of the static demo.')
}
