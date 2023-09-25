import tiled_mm from "./src/shaders/tiled_mm.wgsl";
import {
  setTileSize,
  generate_random_matrix,
  Activation,
} from "./src/utils.js";

function createMLP(config) {
  // create a sequential model
}

async function linear(x, weights, batchSize, in_features, out_features, ts) {
  // Initialize WebGPU
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  console.log("Limits: ", device.limits);

  // Your WGSL code as a string
  const wgslCode = setTileSize(tiled_mm, ts); // Replace this with your actual WGSL code

  // Create shader module
  const shaderModule = device.createShaderModule({ code: wgslCode });

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
  const [xBuffer, weightsBuffer, yBuffer] = [x, weights, y].map((arr) =>
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
  xBuffer.unmap();
  weightsBuffer.unmap();
  yBuffer.unmap();

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
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "uniform",
        },
      },
    ],
  });

  let start = performance.now();

  // Bind group
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: xBuffer } },
      { binding: 1, resource: { buffer: weightsBuffer } },
      { binding: 2, resource: { buffer: yBuffer } },
      { binding: 3, resource: { buffer: paramsBuffer } },
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

  // Submit and execute
  device.queue.submit([commandEncoder.finish()]);

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

function gemm_cpu(A, B, rowsA, colsA, colsB) {
  let start = performance.now();
  if (!A || !B || !rowsA || !colsA || !colsB) return null;

  const C = new Float32Array(rowsA * colsB).fill(0.0);

  for (let i = 0; i < rowsA; i++) {
    for (let j = 0; j < colsB; j++) {
      let sum = 0;
      for (let k = 0; k < colsA; k++) {
        sum += A[i * colsA + k] * B[k * colsB + j];
      }
      C[i * colsB + j] = sum;
    }
  }
  let end = performance.now();

  console.log("CPU Time: ", end - start, "ms");
  console.log("CPU result: ", C);

  return C;
}

// all must be divisible by tile_size
let batch_size = 16384;
let input_size = 32;
let output_size = 64;
let tile_size = 16; // must not be bigger than 16

let X = generate_random_matrix(batch_size, input_size);
let W = generate_random_matrix(input_size, output_size);

// let X = new Float32Array([
//   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
//   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
//   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
// ]);

// let W = new Float32Array([
//   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8,
//   9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
//   15, 16,
// ]);

let gpu_gemm = await linear(
  X,
  W,
  batch_size,
  input_size,
  output_size,
  tile_size
);
//let cpu_gemm = await gemm_cpu(A, B, width, height, height);
