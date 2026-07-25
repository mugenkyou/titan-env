import { defineConfig } from "vite";
import { resolve } from "path";

function decisionRoutePlugin() {
  return {
    name: "decision-route",
    configureServer(server) {
      server.middlewares.use((req, _res, next) => {
        const raw = req.url;
        if (!raw) return next();
        const path = raw.split("?")[0];
        if (path === "/decision" || path === "/decision/") {
          req.url =
            "/decision.html" + (raw.includes("?") ? "?" + raw.split("?")[1] : "");
        }
        next();
      });
    },
    configurePreviewServer(server) {
      server.middlewares.use((req, _res, next) => {
        const raw = req.url;
        if (!raw) return next();
        const path = raw.split("?")[0];
        if (path === "/decision" || path === "/decision/") {
          req.url =
            "/decision.html" + (raw.includes("?") ? "?" + raw.split("?")[1] : "");
        }
        next();
      });
    },
  };
}

export default defineConfig({
  root: ".",
  publicDir: "public",
  plugins: [decisionRoutePlugin()],
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
        decision: resolve(__dirname, "decision.html"),
      },
    },
  },
});
