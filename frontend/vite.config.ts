import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build straight into the FastAPI static dir so `python -m app.cli serve` serves
// the API and the UI from one origin on one port. In dev, /api is proxied instead.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../backend/static", emptyOutDir: true },
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
