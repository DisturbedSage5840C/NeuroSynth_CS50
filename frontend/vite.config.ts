import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/auth":        { target: "http://localhost:8000", changeOrigin: true },
      "/v3":          { target: "http://localhost:8000", changeOrigin: true },
      "/v2":          { target: "http://localhost:8000", changeOrigin: true },
      "/predictions": { target: "http://localhost:8000", changeOrigin: true },
      "/patients":    { target: "http://localhost:8000", changeOrigin: true },
      "/biomarkers":  { target: "http://localhost:8000", changeOrigin: true },
      "/reports":     { target: "http://localhost:8000", changeOrigin: true },
      "/features":    { target: "http://localhost:8000", changeOrigin: true },
      "/causal":      { target: "http://localhost:8000", changeOrigin: true },
      "/pipelines":   { target: "http://localhost:8000", changeOrigin: true },
      "/literature":  { target: "http://localhost:8000", changeOrigin: true },
      "/admin":       { target: "http://localhost:8000", changeOrigin: true },
      "/health":      { target: "http://localhost:8000", changeOrigin: true },
      "/ready":       { target: "http://localhost:8000", changeOrigin: true },
      "/metrics":     { target: "http://localhost:8000", changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
