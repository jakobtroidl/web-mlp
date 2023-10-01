// @ts-check
const { test, expect } = require("@playwright/test");

test("WebGPU should be available", async ({ page }) => {
  // Navigate to a neutral page like 'about:blank'
  await page.goto("https://jakobtroidl.github.io/html-boilerplate/");

  // Evaluate WebGPU availability in browser context
  const isWebGPUAvailable = await page.evaluate(() => {
    return navigator.gpu != undefined;
  });

  // Assert WebGPU availability
  expect(isWebGPUAvailable).toBe(true);

  // You can add more code here to test specific WebGPU functionalities by evaluating custom JS
});
