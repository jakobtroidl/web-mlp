import { defineConfig } from "vite";

export default defineConfig({
  build: {
    lib: {
      // Could also be a dictionary or array of multiple entry points
      entry: "index.js",
      name: "Web-MLP",
      // the proper extensions will be added
      fileName: "web-mlp",
    },
    rollupOptions: {
      input: ["index.js"],
    },
    outDir: "dist",
  },
  // This is here because of https://github.com/vitejs/vite/issues/7287
  optimizeDeps: {
    exclude: ["@webonnx/wonnx-wasm"],
    // esbuildOptions: {
    //   target: "es2020",
    // },
  },
});
