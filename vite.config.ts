import { defineConfig } from "vite";

export default defineConfig({

  plugins: [
    {
      name: 'wgsl-loader',
      resolveId(id) {
        return id.endsWith('.wgsl') ? id : null;
      },
      async load(id) {
        if (id.endsWith('.wgsl')) {
          const fs = require('fs').promises;
          const contents = await fs.readFile(id, 'utf-8');
          return `export default ${JSON.stringify(contents)}`;
        }
        return null;
      },
    },
  ],

  assetsInclude: ["**/*.onnx"],

  // This is here because of https://github.com/vitejs/vite/issues/7287
  optimizeDeps: {
    exclude: ["@webonnx/wonnx-wasm"],
    esbuildOptions: {
      target: "es2020",
    },
  },
});
