// Query parameters inside the hash route (#/ask?q=...), so a recorded answer
// or a workflow run can be linked to directly.

export function hashParams(): URLSearchParams {
  const hash = window.location.hash
  const i = hash.indexOf('?')
  return new URLSearchParams(i >= 0 ? hash.slice(i + 1) : '')
}

/** Update the address bar without a navigation (no hashchange event). */
export function replaceHash(route: string, params: Record<string, string> = {}): void {
  const query = new URLSearchParams(params).toString()
  window.history.replaceState(null, '', `#/${route}${query ? `?${query}` : ''}`)
}
