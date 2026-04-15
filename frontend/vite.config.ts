import path from "path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// WHY: In Docker, the backend hostname is "api" (the compose service name).
//      On the host, it's "localhost". The env var lets both work.
const API_TARGET = process.env.VITE_API_TARGET ?? "http://localhost:8001"
const WS_TARGET = API_TARGET.replace("http", "ws")

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 3000,
    proxy: {
      "/api/chat": {
        target: WS_TARGET,
        ws: true,
      },
      "/api": {
        target: API_TARGET,
        changeOrigin: true,
      },
      "/health": {
        target: API_TARGET,
        changeOrigin: true,
      },
    },
  },
})
