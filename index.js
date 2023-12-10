import { MLP, Linear } from "./src/mlp.js";
import { from_json } from "./src/modelLoader.js";
import {
  generate_random_matrix,
  generate_ones_matrix,
  getActivation,
} from "./src/utils.js";
import { setTileSize } from "./src/utils";
import shaderString from "./src/shaders/tiled_mm.wgsl?raw";
import { initWebGPU } from "./src/setup.js";

function loadComputeParams(model, batch_size) {
  let n_layers = model.length;
  let params = [];

  for (let i = 0; i < n_layers; i++) {
    let layerParam = [
      batch_size, // batch_size,
      model[i].weight_shape[1], // in_features,
      model[i].weight_shape[0], // out_features,
      getActivation(model[i].activation), // activation
    ];

    layerParam = new Uint32Array(layerParam);
    params.push(layerParam);
  }

  return params;
}

function createDataBuffers(device, model, batch_size) {
  let dataBuffers = [];
  let n_buffers = model.length + 1;

  for (let i = 0; i < n_buffers; i++) {
    let bufferElements = 0.0;
    if (i == 0) {
      // input layer size
      bufferElements = batch_size * model[i].weight_shape[1];
    } else if (i == n_buffers - 1) {
      // output layer size
      bufferElements = batch_size * model[i - 1].weight_shape[0];
    } else {
      // hidden layer size
      bufferElements = batch_size * model[i].weight_shape[1];
    }
    // initialize all data buffers with zeros
    let bufferSize = bufferElements * 4;
    let data = new Float32Array(bufferSize).fill(0.0);
    let buffer = createGPUBuffer(device, data);

    dataBuffers.push(buffer);
  }
  return dataBuffers;
}

function getBindLayout(device) {
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
}

function createGPUBuffer(device, data, isUniform = false) {
  let buffer = device.createBuffer({
    size: data.byteLength,
    usage: isUniform
      ? GPUBufferUsage.UNIFORM
      : GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });

  if (data instanceof Uint32Array) {
    // map the data to the buffer
    new Uint32Array(buffer.getMappedRange()).set(data);
  } else {
    new Float32Array(buffer.getMappedRange()).set(data);
  }
  buffer.unmap();
  return buffer;
}

function getComputePipeline(device, shaderModule, layout) {
  return device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [layout],
    }),
    compute: {
      module: shaderModule,
      entryPoint: "main",
    },
  });
}

async function createMLP(tf_model, device, batch_size = 1024, tile_size = 16) {
  const wgslCode = setTileSize(shaderString, tile_size);
  const shaderModule = device.createShaderModule({ code: wgslCode });

  let params = loadComputeParams(tf_model, batch_size);

  // create buffers
  let weightBuffers = tf_model.map((layer) => {
    return createGPUBuffer(device, layer.weights);
  });

  let biasBuffers = tf_model.map((layer) =>
    createGPUBuffer(device, layer.biases)
  );
  let isUniform = true;
  let computeParamsBuffers = params.map((p) =>
    createGPUBuffer(device, p, isUniform)
  );
  let dataBuffers = createDataBuffers(device, tf_model, batch_size);

  // create bind group layout
  let layout = getBindLayout(device);
  let computePipeline = getComputePipeline(device, shaderModule, layout);

  let layers = [];

  // create layers
  for (let i = 0; i < tf_model.length; i++) {
    let bindGroup = device.createBindGroup({
      layout: layout,
      entries: [
        { binding: 0, resource: { buffer: dataBuffers[i] } },
        { binding: 1, resource: { buffer: weightBuffers[i] } },
        { binding: 2, resource: { buffer: biasBuffers[i] } },
        { binding: 3, resource: { buffer: dataBuffers[i + 1] } },
        { binding: 4, resource: { buffer: computeParamsBuffers[i] } },
      ],
    });

    layers.push(
      new Linear(
        i,
        device,
        bindGroup,
        dataBuffers[i],
        dataBuffers[i + 1],
        computePipeline,
        tf_model[i].weight_shape[1],
        tf_model[i].weight_shape[0],
        batch_size,
        tile_size
      )
    );
  }

  let outputBuffer = dataBuffers[dataBuffers.length - 1];
  let mlp = new MLP(device, layers);

  return [mlp, outputBuffer];
}

async function testMLP() {
  let batch_size = 20;
  let tile_size = 8; // must not be bigger than 16
  const path = "https://jakobtroidl.github.io/data/mlp-v11.json";

  let device = await initWebGPU();

  let model_data = await from_json(path);
  let [model, outputBuffer] = await createMLP(
    model_data,
    device,
    batch_size,
    tile_size
  );
  console.log("model", model);

  console.log(batch_size, model.inputSize, model.outputSize, model);
  let X = generate_ones_matrix(batch_size, model.inputSize);

  console.log("Starting WebMLP Inference...");
  let commandEncoder = device.createCommandEncoder();

  let start = performance.now();
  model.inference(X, commandEncoder);
  device.queue.submit([commandEncoder.finish()]);

  // transfer output buffer to CPU
  let result = await model.transferToCPU(outputBuffer);
  let end = performance.now();
  console.log("WebMLP Inference time + Data Transfer: ", end - start, "ms");
  console.log("WebMLP result", result);
  console.log("WebMLP result should match dummy_output in model.json");
}

// testMLP();

export { createMLP, from_json, initWebGPU };
