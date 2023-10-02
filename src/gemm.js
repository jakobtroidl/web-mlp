import { initWebGPU } from "./setup.js";
import { Activation } from "./utils.js";

export async function gemm(
  x,
  weights,
  batchSize,
  in_features,
  out_features,
  ts
) {
  // // Initialize WebGPU
  const { device, shaderModule } = await initWebGPU(ts);

  const biases = new Float32Array(batchSize).fill(0);
  const y = new Float32Array(batchSize * out_features).fill(0);

  // Create params buffer and write data to it
  const paramsBuffer = device.createBuffer({
    size: 4 * 4, // 2 uint32s of 4 bytes each (width, height, activation)
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });

  new Uint32Array(paramsBuffer.getMappedRange()).set([
    batchSize,
    in_features,
    out_features,
    Activation.ReLU,
  ]);
  paramsBuffer.unmap();

  // Create and populate buffers
  const [xBuffer, weightsBuffer, yBuffer, biasBuffer] = [
    x,
    weights,
    y,
    biases,
  ].map((arr) =>
    device.createBuffer({
      size: arr.byteLength,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    })
  );

  // staging buffer to make data accessible to CPU
  const stagingBuffer = device.createBuffer({
    size: y.byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  new Float32Array(xBuffer.getMappedRange()).set(x);
  new Float32Array(weightsBuffer.getMappedRange()).set(weights);
  new Float32Array(yBuffer.getMappedRange()).set(y);
  new Float32Array(biasBuffer.getMappedRange()).set(biases);
  xBuffer.unmap();
  weightsBuffer.unmap();
  yBuffer.unmap();
  biasBuffer.unmap();

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "read-only-storage",
        },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "read-only-storage",
        },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "read-only-storage",
        },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage",
        },
      },
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "uniform",
        },
      },
    ],
  });



  // Bind group
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: xBuffer } },
      { binding: 1, resource: { buffer: weightsBuffer } },
      { binding: 2, resource: { buffer: biasBuffer } },
      { binding: 3, resource: { buffer: yBuffer } },
      { binding: 4, resource: { buffer: paramsBuffer } },
    ],
  });

  // Create pipeline
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    }),
    compute: {
      module: shaderModule,
      entryPoint: "main",
    },
  });

  // Command encoder and pass
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(
    Math.ceil(out_features / ts),
    Math.ceil(batchSize / ts)
  );
  passEncoder.end();

  // Copy output buffer to staging buffer
  commandEncoder.copyBufferToBuffer(
    yBuffer,
    0, // Source offset
    stagingBuffer,
    0, // Destination offset
    y.byteLength
  );

  await device.queue.onSubmittedWorkDone();

  let start = performance.now();
  // Submit and execute
  device.queue.submit([commandEncoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  
  let end = performance.now();

  console.log("GPU Time: ", end - start, "ms");

  // map staging buffer to read results back to JS
  await stagingBuffer.mapAsync(
    GPUMapMode.READ,
    0, // Offset
    y.byteLength // Length
  );

  const copyArrayBuffer = stagingBuffer.getMappedRange(0, y.byteLength);
  const data = copyArrayBuffer.slice();
  stagingBuffer.unmap();

  console.log("GPU result: ", new Float32Array(data));

  return new Float32Array(data);
}
