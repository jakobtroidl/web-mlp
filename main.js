import tiled_mm from "./src/shaders/tiled_mm.wgsl";
import { mlp_config } from "./example-config.js";
import { MLP, Linear } from "./src/mlp.js";
import { from_tfjs } from "./src/modelLoader.js";

import {
  setTileSize,
  generate_random_matrix,
  Activation,
  getActivation,
} from "./src/utils.js";

async function initWebGPU(ts) {
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

  const device = await adapter.requestDevice();
  const wgslCode = setTileSize(tiled_mm, ts); // Replace this with your actual WGSL code
  const shaderModule = device.createShaderModule({ code: wgslCode });

  return { device, shaderModule };
}

function loadWeights(config) {
  return config.weights.map((w) => {
    return new Float32Array(w.data);
  });
}

function loadBiases(config) {
  return config.biases.map((b) => {
    return new Float32Array(b.data);
  });
}

function loadComputeParams(config, batch_size) {
  let n_layers = config.params.n_layers;
  let params = [];

  for (let i = 0; i < n_layers; i++) {
    let layerParam = [
      batch_size, // batch_size,
      i == 0 ? config.params.n_inputs : config.params.n_neurons, // in_features,
      i == n_layers - 1 ? config.params.n_outputs : config.params.n_neurons, // out_features,
      getActivation(config.params.activation), // activation
    ];

    layerParam = new Uint32Array(layerParam);
    params.push(layerParam);
  }

  return params;
}

function createDataBuffers(device, config, batch_size) {
  let dataBuffers = [];
  let n_buffers = config.params.n_layers + 1;

  console.log("n_buffers: ", n_buffers);
  for (let i = 0; i < n_buffers; i++) {
    let bufferSize = 0.0;
    if (i == 0) {
      // input layer size
      bufferSize = batch_size * config.params.n_inputs;
    } else if (i == n_buffers - 1) {
      // output layer size
      bufferSize = batch_size * config.params.n_outputs;
    } else {
      // hidden layer size
      bufferSize = batch_size * config.params.n_neurons;
    }
    // initialize all data buffers with zeros
    let data = new Float32Array(bufferSize).fill(0);
    let buffer = createGPUBuffer(device, data);

    dataBuffers.push(buffer);
  }
  return dataBuffers;
}

function getPerLayerBindLayout(device) {
  return device.createBindGroupLayout({
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
}

function createGPUBuffer(device, data) {
  let buffer = device.createBuffer({
    size: data.byteLength,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });

  if (data instanceof Uint32Array) {
    // map the data to the buffer
    new Uint32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();
  } else {
    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();
  }
  return buffer;
}

function getComputePipeline(device, shaderModule, bindGroupLayout) {
  return device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    }),
    compute: {
      module: shaderModule,
      entryPoint: "main",
    },
  });
}

function createLayers(
  device,
  computePipeline,
  bindGroupLayout,
  config,
  batch_size,
  tile_size,
  dataBuffers,
  weightBuffers,
  biasBuffers,
  computeParamsBuffers
) {
  let layers = [];
  let n_layers = config.params.n_layers;

  for (let i = 0; i < n_layers; i++) {
    let layer = new Linear(
      device,
      computePipeline,
      bindGroupLayout,
      batch_size,
      config.params.n_outputs,
      tile_size,
      dataBuffers[i],
      weightBuffers[i],
      biasBuffers[i],
      computeParamsBuffers[i],
      dataBuffers[i + 1]
    );
    layers.push(layer);
  }
  return layers;
}

async function createMLP(config, batch_size = 1024, tile_size = 16) {
  const { device, shaderModule } = await initWebGPU();

  let weights = loadWeights(config);
  let biases = loadBiases(config);
  let params = loadComputeParams(config, batch_size);

  // create buffers
  let weightBuffers = weights.map((w) => createGPUBuffer(device, w));
  let biasBuffers = biases.map((b) => createGPUBuffer(device, b));
  let computeParamsBuffers = params.map((p) => createGPUBuffer(device, p));
  let dataBuffers = createDataBuffers(device, config, batch_size);

  // create bind group layout
  let perLayerBindLayout = getPerLayerBindLayout(device);
  let computePipeline = getComputePipeline(
    device,
    shaderModule,
    perLayerBindLayout
  );

  let layers = createLayers(
    device,
    computePipeline,
    perLayerBindLayout,
    config,
    batch_size,
    tile_size,
    dataBuffers,
    weightBuffers,
    biasBuffers,
    computeParamsBuffers
  );

  return new MLP(layers);
}

async function linear(x, weights, batchSize, in_features, out_features, ts) {
  // // Initialize WebGPU
  const { device, shaderModule } = await initWebGPU(ts);

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

// let gpu_gemm = await linear(
//   X,
//   W,
//   batch_size,
//   input_size,
//   output_size,
//   tile_size
// );

const path = "https://jakobtroidl.github.io/data/trainedModel/model.json";
from_tfjs(path).then((res) => {
  console.log(res);
});

// console.log("config", mlp_config);
// let mlp = await createMLP(mlp_config);
//mlp.inference(X);
