import sgemm from "webgpu-blas";
import tiled_mm from "./shaders/tiled_mm.wgsl";

function generate_random_matrix(w, h) {
  return Float32Array.from(Array(w * h).fill(0), () => Math.random());
}

async function gemm_wgpu(aData, bData, w, h) {
  // Initialize WebGPU
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  console.log("Limits: ", device.limits);

  // Your WGSL code as a string
  const wgslCode = tiled_mm; // Replace this with your actual WGSL code

  // Create shader module
  const shaderModule = device.createShaderModule({ code: wgslCode });

  const cData = new Float32Array(w * h).fill(0);

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
      bindGroupLayouts: [bindGroupLayout],
    }),
    compute: {
      module: shaderModule,
      entryPoint: "main",
    },
  });

  let start = performance.now();

  // Command encoder and pass
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(Math.ceil(w / 2), Math.ceil(h / 2)); // Assuming TILE_SIZE is 2
  passEncoder.end();

  let end = performance.now();

  console.log("Time: ", end - start);

  // Copy output buffer to staging buffer
  commandEncoder.copyBufferToBuffer(
    cBuffer,
    0, // Source offset
    stagingBuffer,
    0, // Destination offset
    cData.byteLength
  );

  // Submit and execute
  device.queue.submit([commandEncoder.finish()]);

  // map staging buffer to read results back to JS
  await stagingBuffer.mapAsync(
    GPUMapMode.READ,
    0, // Offset
    cData.byteLength // Length
  );

  const copyArrayBuffer = stagingBuffer.getMappedRange(0, cData.byteLength);
  const data = copyArrayBuffer.slice();
  stagingBuffer.unmap();

  console.log(new Float32Array(data));

  return new Float32Array(data);
}

function gemm_cpu(A, B, rowsA, colsA, colsB) {
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

  console.log(C);
  return C;
}

let width = 4;
let height = 4;

// let A = generate_random_matrix(width, height);
// let B = generate_random_matrix(width, height);

let A = new Float32Array([
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
]);
let B = new Float32Array([
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
]);

let gpu_gemm = await gemm_wgpu(A, B, width, height);
let cpu_gemm = gemm_cpu(A, B, width, height, height);

if (gpu_gemm == cpu_gemm) {
  console.log("success, results match");
} else {
  console.log("fail, results do not match");
}
