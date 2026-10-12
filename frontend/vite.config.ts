import { cpSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import react from '@vitejs/plugin-react'
import { defineConfig, type Plugin } from 'vite'

const root = fileURLToPath(new URL('.', import.meta.url))

// The static demo (GitHub Pages) reads JSON exported by `tieout demo export-static`.
function demoData(): Plugin {
  return {
    name: 'tieout-demo-data',
    apply: 'build',
    closeBundle() {
      cpSync(`${root}demo-data`, `${root}dist/demo-data`, { recursive: true })
    },
  }
}

// In development the API runs separately (`tieout serve`, port 8000).
// `vite build --mode static` builds the server-less demo; VITE_BASE sets the
// path it is served under (e.g. /tieout/ on GitHub Pages).
export default defineConfig(({ mode }) => ({
  base: process.env.VITE_BASE ?? '/',
  plugins: [react(), ...(mode === 'static' ? [demoData()] : [])],
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
}))
