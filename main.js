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
  const [width, height] = [1024, 1024]; // Modify these dimensions as needed

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
      size: arr.byteLength, // 4 bytes for width, 4 bytes for height
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    })
  );

  // staging buffer to make data accessible to CPU
  const stagingBuffer = device.createBuffer({
    size: cData.byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  new Float32Array(aBuffer.getMappedRange()).set(aData);
  new Float32Array(bBuffer.getMappedRange()).set(bData);
  new Float32Array(cBuffer.getMappedRange()).set(cData);
  aBuffer.unmap();
  bBuffer.unmap();
  cBuffer.unmap();

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
          type: "storage",
        },
      },
    ],
  });

  // Bind group
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: aBuffer } },
      { binding: 1, resource: { buffer: bBuffer } },
      { binding: 2, resource: { buffer: cBuffer } },
    ],
  });

  // Create pipeline
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    }),
    compute: {
      module: shaderModule,
      entryPoint: "main"
    }
  });

  // Command encoder and pass
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16)); // Assuming TILE_SIZE is 16
  passEncoder.end();

  // Copy output buffer to staging buffer
  commandEncoder.copyBufferToBuffer(
    cBuffer,
    0, // Source offset
    stagingBuffer,
    0, // Destination offset
    cData.byteLength
  );

  console.log("Running matrix multiplication...");

  // Submit and execute
  device.queue.submit([commandEncoder.finish()]);

  console.log("Running matrix multiplication...");

  // map staging buffer to read results back to JS
  await stagingBuffer.mapAsync(
    GPUMapMode.READ,
    0, // Offset
    cData.byteLength // Length
  );

  console.log("Running matrix multiplication...");

  const copyArrayBuffer = stagingBuffer.getMappedRange(0, cData.byteLength);
  const data = copyArrayBuffer.slice();
  stagingBuffer.unmap();
  console.log(new Float32Array(data));
}

runMatrixMultiplication().catch(console.error);
