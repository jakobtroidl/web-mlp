import { defineConfig } from "vite";
import { resolve } from "path";
import { nodePolyfills } from 'vite-plugin-node-polyfills'


export default defineConfig({
  plugins: [
    nodePolyfills(),
  ],
  build: {
    emptyOutDir: false,
    sourcemap: "inline",
    lib: {
      // Could also be a dictionary or array of multiple entry points
      entry: resolve(__dirname, "index.js"),
      name: "web-mlp",
      // the proper extensions will be added
      fileName: "web-mlp",
      formats: ["es"],
    },
  },
  // This is here because of https://github.com/vitejs/vite/issues/7287
  optimizeDeps: {
    exclude: ["@webonnx/wonnx-wasm"],
    esbuildOptions: {
      target: "es2020",
    },
  },
});
