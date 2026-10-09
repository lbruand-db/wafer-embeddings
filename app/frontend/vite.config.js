import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

// `vite build frontend` -> frontend/dist, served by server.py. In `npm run dev`, /api is
// proxied to a local `python server.py` on :8000.
export default defineConfig({
  plugins: [vue()],
  base: "./",
  build: { outDir: "dist", emptyOutDir: true },
  server: { proxy: { "/api": "http://localhost:8000" } },
});
