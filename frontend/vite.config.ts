import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

// Generate build timestamp
const buildTime = new Date().toISOString().replace(/[-:]/g, '').split('.')[0] + 'Z'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  // Load .env from parent directory (project root)
  envDir: '../',
  define: {
    // Inject build time as a global constant
    '__BUILD_TIME__': JSON.stringify(buildTime),
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        ws: true,  // Enable WebSocket proxying
      },
    },
  },
})
