import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: Number(process.env.VITE_DEV_PORT || 5173),
    strictPort: true,
    watch: {
      ignored: ["**/venv/**", "**/.venv/**", "**/__pycache__/**", "**/outputs/**"]
    },
    proxy: {
      "/api": "http://127.0.0.1:8000"
    }
  },
  optimizeDeps: {
    entries: ["index.html"]
  },
  build: {
    outDir: "frontend/dist"
  }
});
