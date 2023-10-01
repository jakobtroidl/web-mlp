// import { test, expect } from "@playwright/test";

// test("It should take a snapshot of the GPU Chrome page", async ({ page }) => {
//   await page.goto("chrome://gpu", { waitUntil: "domcontentloaded" });
//   await page.screenshot({ path: "gpu.png" });
//   await expect(
//     page.locator("text=Graphics Feature Status").first()
//   ).toBeVisible();
// });
// test("Multiplies two matrices", async () => {
//   const browser = await puppeteer.launch();
//   const page = await browser.newPage();

//   let A = new Float32Array([
//     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
//   ]);
//   let B = new Float32Array([
//     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
//   ]);

//   let Y = new Float32Array(
//     90,
//     100,
//     110,
//     116,
//     202,
//     228,
//     254,
//     272,
//     314,
//     356,
//     398,
//     428,
//     413,
//     470,
//     527,
//     569
//   );

//   let batch_size = 4;
//   let in_features = 4;
//   let out_features = 4;

//   let tile_size = 2;

//   const success = page.evaluate(async () => {
//     let result = await gemm(
//       A,
//       B,
//       batch_size,
//       in_features,
//       out_features,
//       tile_size
//     );

//     expect(result).toEqual(Y);
//   });
// });
