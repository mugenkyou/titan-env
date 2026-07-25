import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  root: ".",
  publicDir: "public",
  server: {
    port: 5173,
    open: true,
  },
  build: {
    outDir: "dist",
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        analysis: resolve(__dirname, "analysis.html"),
      },
    },
  },
});
