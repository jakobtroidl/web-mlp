// const { test, expect } = require("@playwright/test");
// import { gemm } from "../src/gemm.js";
// import { generate_random_matrix } from "../src/utils.js";

// test("Gemm test", async ({ page }) => {
//   // Navigate to a neutral page like 'about:blank'

//   await page.addScriptTag({ path: "../src/utils.js" });

//   // Evaluate WebGPU availability in browser context
//   const matricesEqual = await page.evaluate(async () => {
//     // let A = new Float32Array([
//     //   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
//     // ]);
//     // let B = new Float32Array([
//     //   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
//     // ]);
//     // let Y = new Float32Array([
//     //   90, 100, 110, 116, 202, 228, 254, 272, 314, 356, 398, 428, 413, 470, 527,
//     //   569,
//     // ]);
//     // let batch_size = 4;
//     // let in_features = 4;
//     // let out_features = 4;
//     // let tile_size = 2;
//     // let result = await gemm(
//     //   A,
//     //   B,
//     //   batch_size,
//     //   in_features,
//     //   out_features,
//     //   tile_size
//     // );
//     // if (result == Y) return true;
//     // return false;

//     let a = generate_random_matrix(10, 10);
//     let b = generate_random_matrix(10, 10);
//   });

//   expect(matricesEqual).toBe(true);

//   // You can add more code here to test specific WebGPU functionalities by evaluating custom JS
// });
