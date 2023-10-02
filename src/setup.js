import { setTileSize } from "./utils";
import shaderString from "./shaders/tiled_mm.wgsl?raw";

export async function initWebGPU(ts) {
  // Initialize WebGPU
  if (navigator.gpu === undefined) {
    console.error("WebGPU is not supported.");
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();

  if (!adapter) {
    console.error("WebGPU is not supported. Failed to find a GPU adapter.");
    return;
  }

  console.log("tile_size", ts);

  const device = await adapter.requestDevice();

  const wgslCode = setTileSize(shaderString, ts); // Replace this with your actual WGSL code
  const shaderModule = device.createShaderModule({ code: wgslCode });

  return { device, shaderModule };
}
