import { useEffect, useState } from 'react'
import { api, type UiConfig } from './api'

const FALLBACK: UiConfig = { demo_mode: false, demo_questions: [], model: '' }
let pending: Promise<UiConfig> | null = null

/** Server settings the UI depends on (demo mode), fetched once per page load. */
export function useConfig(): UiConfig | null {
  const [config, setConfig] = useState<UiConfig | null>(null)
  useEffect(() => {
    let active = true
    pending ??= api.config().catch(() => FALLBACK)
    void pending.then((c) => {
      if (active) setConfig(c)
    })
    return () => {
      active = false
    }
  }, [])
  return config
}
