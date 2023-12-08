export async function initWebGPU() {
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

  const requiredFeatures = [];

  const requiredLimits = {};
  requiredLimits.maxBufferSize = adapter.limits.maxBufferSize;
  requiredLimits.maxComputeInvocationsPerWorkgroup =
    adapter.limits.maxComputeInvocationsPerWorkgroup;
  requiredLimits.maxComputeWorkgroupSizeX =
    adapter.limits.maxComputeWorkgroupSizeX;
  requiredLimits.maxComputeWorkgroupSizeY =
    adapter.limits.maxComputeWorkgroupSizeY;
  requiredLimits.maxComputeWorkgroupSizeZ =
    adapter.limits.maxComputeWorkgroupSizeZ;
  requiredLimits.maxComputeWorkgroupStorageSize =
    adapter.limits.maxComputeWorkgroupStorageSize;
  requiredLimits.maxStorageBufferBindingSize =
    adapter.limits.maxStorageBufferBindingSize;
  requiredLimits.maxComputeWorkgroupsPerDimension =
    adapter.limits.maxComputeWorkgroupsPerDimension;

  const device = await adapter.requestDevice({
    defaultQueue: {
      label: "default",
    },
    requiredFeatures,
    requiredLimits,
  });

  return device;
}
