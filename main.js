import tiled_mm from "./shaders/tiled_mm.wgsl";

async function runMatrixMultiplication() {
  // Initialize WebGPU
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  // Your WGSL code as a string
  const wgslCode = tiled_mm; // Replace this with your actual WGSL code

  // Create shader module
  const shaderModule = device.createShaderModule({ code: wgslCode });

  // Matrix dimensions
  const [width, height] = [256, 256]; // Modify these dimensions as needed

  // Initialize data
  const aData = Float32Array.from(Array(width * height).fill(0), () =>
    Math.random()
  );
  const bData = Float32Array.from(Array(width * height).fill(0), () =>
    Math.random()
  );
  const cData = new Float32Array(width * height).fill(0);

  // Create and populate buffers
  const [aBuffer, bBuffer, cBuffer] = [aData, bData, cData].map((arr) =>
    device.createBuffer({
      size: arr.byteLength,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    })
  );

  new Float32Array(aBuffer.getMappedRange()).set(aData);
  new Float32Array(bBuffer.getMappedRange()).set(bData);
  aBuffer.unmap();
  bBuffer.unmap();

  // Create pipeline
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: shaderModule,
      entryPoint: "main",
    },
  });

  // Bind group
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: aBuffer } },
      { binding: 1, resource: { buffer: bBuffer } },
      { binding: 2, resource: { buffer: cBuffer } },
    ],
  });

  // Command encoder and pass
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatch(Math.ceil(width / 16), Math.ceil(height / 16)); // Assuming TILE_SIZE is 16
  passEncoder.endPass();

  const start = performance.now();

  // Submit and execute
  device.queue.submit([commandEncoder.finish()]);

  const end = performance.now();
  console.log("WGSL inference time: " + (end - start) + " ms");

  // Read back the results
  const readBuffer = device.createBuffer({
    size: cData.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  commandEncoder.copyBufferToBuffer(
    cBuffer,
    0,
    readBuffer,
    0,
    cData.byteLength
  );

  await readBuffer.mapAsync(GPUMapMode.READ);
  const outputData = new Float32Array(readBuffer.getMappedRange());

  // Do something with outputData
  console.log(outputData);

  readBuffer.unmap();
}

runMatrixMultiplication().catch(console.error);
