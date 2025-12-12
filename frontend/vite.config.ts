import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

// Generate build timestamp
const buildTime = new Date().toISOString().replace(/[-:]/g, '').split('.')[0] + 'Z'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // Load env from parent directory (project root)
  const env = loadEnv(mode, path.resolve(__dirname, '..'), '')

  // Port configuration with environment variable overrides
  // Defaults: frontend=5173, backend=8000 (same as before for backward compatibility)
  const frontendPort = parseInt(env.FRONTEND_PORT || '5173', 10)
  const backendPort = parseInt(env.SERVER_PORT || '8000', 10)

  return {
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
      port: frontendPort,
      allowedHosts: ['.ppinfra.com', '.gpu-instance.ppinfra.com', 'localhost'],
      proxy: {
        '/api': {
          target: `http://localhost:${backendPort}`,
          changeOrigin: true,
          ws: true,  // Enable WebSocket proxying
        },
      },
    },
  }
})
