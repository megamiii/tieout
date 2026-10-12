// Typed client for the tieout API. Types mirror backend/tieout/runs.py and api/app.py.

export type RunStatus = 'answered' | 'needs_review'
export type Decision = 'accept' | 'override'

export interface Segment {
  text: string
  mark: boolean
}

interface CitationBase {
  source_id: string
  quotes: string[] // every quote cited from this source
}

export interface ChunkCitation extends CitationBase {
  found: true
  kind: 'chunk'
  ticker: string
  company: string
  fiscal_year: number
  section: string
  section_title: string
  page: number
  doc_name: string
  edgar_url: string
  segments: Segment[]
  matches: ('quote' | 'figures' | 'none')[] // per quote
}

export interface FactCitation extends CitationBase {
  found: true
  kind: 'fact'
  ticker: string | null
  concept: string
  label: string
  unit: string
  value: string
  period_start: string | null
  period_end: string
  accession: string
  filed: string
}

export interface MissingCitation extends CitationBase {
  found: false
}

export type Citation = ChunkCitation | FactCitation | MissingCitation

export interface TraceTool {
  name: string
  arguments: string
  error: string | null
  sources?: string[]
  latency_ms?: number
}

export interface TraceStep {
  step: number
  tokens?: number
  prompt_tokens?: number
  completion_tokens?: number
  latency_ms?: number
  cached?: boolean
  forced_submit?: boolean
  nudged?: boolean
  tools: TraceTool[]
}

export interface ReviewRecord {
  id: number
  created_at: string
  decision: Decision
  final_answer: string
  note: string
  reviewer: string
}

export interface RunSummary {
  id: string
  created_at: string
  source: string
  question: string
  status: RunStatus
  answer: string | null
  confidence: number | null
  review_reasons: string[]
  review: ReviewRecord | null
}

export interface RunDetail extends RunSummary {
  rationale: string | null
  warnings: string[]
  stop_reason: string
  model: string
  usage: {
    requests: number
    prompt_tokens: number
    completion_tokens: number
    latency_ms: number | null
  }
  citations: Citation[]
  trace: TraceStep[]
  decisions: ReviewRecord[]
  trace_id?: string | null
}

export interface Company {
  ticker: string
  name: string
  sector: string
  fiscal_years: number[]
}

export interface EvalMetrics {
  experiment: string
  eval_set: string
  n: number
  graded: number
  accuracy: number | null
  answered_rate: number | null
  accuracy_answered: number | null
  numeric_accuracy: number | null
  text_accuracy: number | null
  citation_precision: number | null
  citations: number
  abstention_precision: number | null
  abstention_recall: number | null
  requests_per_item: number | null
  tokens_per_item: number | null
  latency_p50_s: number | null
  latency_p95_s: number | null
  errors: number
}

export interface EvalReport {
  meta: Record<string, string>
  metrics: EvalMetrics[]
  experiments: Record<string, string>
}

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!res.ok) {
    let message = res.statusText
    try {
      const body = (await res.json()) as { detail?: unknown }
      if (typeof body.detail === 'string') message = body.detail
      else if (body.detail) message = JSON.stringify(body.detail)
    } catch {
      // not JSON: keep the status text
    }
    throw new ApiError(res.status, message)
  }
  return (await res.json()) as T
}

export const api = {
  companies: () => request<Company[]>('/api/companies'),
  ask: (question: string) =>
    request<RunDetail>('/api/ask', { method: 'POST', body: JSON.stringify({ question }) }),
  runs: (queue: 'pending' | 'reviewed' | 'all') =>
    request<RunSummary[]>(`/api/runs?queue=${queue}`),
  run: (id: string) => request<RunDetail>(`/api/runs/${encodeURIComponent(id)}`),
  review: (
    id: string,
    body: { decision: Decision; reviewer: string; note: string; override_answer?: string },
  ) =>
    request<RunDetail>(`/api/runs/${encodeURIComponent(id)}/review`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  evalReport: () => request<EvalReport>('/api/eval'),
}

// --- audit workflows ------------------------------------------------------------

export type WorkflowKind = 'tieout' | 'analytical_review'

export interface FactRef {
  fact_id: string
  concept: string
  label: string
  value: string
  period: string
  fiscal_year: number
}

export interface TieoutItem {
  status: 'matched' | 'mismatched' | 'unmatched' | 'out_of_scope'
  amount: string
  value: string
  year: number | null
  chunk_id: string
  page: number
  sentence: string
  quarterly: boolean
  non_gaap: boolean
  note: string
  fact: FactRef | null
  concept_rank: number | null
  adjudication: string
}

export interface ReviewItem {
  concept: string
  label: string
  current: FactRef
  prior: FactRef
  delta: string
  pct: string
  explanation: {
    status: 'explained' | 'not_explained' | 'unverified'
    summary: string
    chunk_id: string | null
    quote: string | null
    page: number | null
  }
}

export interface WorkflowRun<T = TieoutItem | ReviewItem> {
  id: string
  created_at: string
  kind: WorkflowKind
  ticker: string
  fiscal_year: number
  doc_name: string
  model: string
  summary: Record<string, number | string | null | Record<string, number>>
  items: T[] | null
  trace_id?: string | null
}

export const workflows = {
  list: (kind: WorkflowKind) => request<WorkflowRun[]>(`/api/workflows?kind=${kind}`),
  get: (id: string) => request<WorkflowRun>(`/api/workflows/${encodeURIComponent(id)}`),
  run: (kind: WorkflowKind, ticker: string, fiscal_year: number) =>
    request<WorkflowRun>(kind === 'tieout' ? '/api/workflows/tieout' : '/api/workflows/review', {
      method: 'POST',
      body: JSON.stringify({ ticker, fiscal_year }),
    }),
}

// --- traces -----------------------------------------------------------------------

export interface TraceSpan {
  span_id: string
  parent_id: string | null
  seq: number
  type: string
  name: string
  start_ms: number
  duration_ms: number
  input: unknown
  output: unknown
  error: string | null
  prompt_tokens: number
  completion_tokens: number
  cached: boolean
  meta: Record<string, unknown>
}

export interface Trace {
  trace_id: string
  kind: string
  name: string
  ref_id: string | null
  started_at: string
  duration_ms: number
  model: string
  totals: Record<string, number>
  spans: TraceSpan[]
}

export const traces = {
  get: (id: string) => request<Trace>(`/api/traces/${encodeURIComponent(id)}`),
}
