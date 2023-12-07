import { MLP, Linear } from "./src/mlp.js";
import { from_tfjs } from "./src/modelLoader.js";
import { initWebGPU } from "./src/setup.js";
import { gemm } from "./src/gemm.js";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgpu";
import { generate_random_matrix, getActivation } from "./src/utils.js";

function loadComputeParams(model, batch_size) {
  let n_layers = model.length;
  let params = [];

  for (let i = 0; i < n_layers; i++) {
    let layerParam = [
      batch_size, // batch_size,
      model[i].weight_shape[0], // in_features,
      model[i].weight_shape[1], // out_features,
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
      bufferElements = batch_size * model[i].weight_shape[0];
      //console.log("bufferElements in first layer: ", bufferElements);
    } else if (i == n_buffers - 1) {
      // output layer size
      bufferElements = batch_size * model[i - 1].weight_shape[1];
    } else {
      // hidden layer size
      bufferElements = batch_size * model[i].weight_shape[0];
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

async function createMLP(tf_model, batch_size = 1024, tile_size = 16) {
  const { device, shaderModule } = await initWebGPU(tile_size);
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
        tf_model[i].weight_shape[0],
        tf_model[i].weight_shape[1],
        batch_size,
        tile_size
      )
    );
  }

  return new MLP(device, layers);
}

async function testGemm() {
  let batch_size = 70000;
  let in_features = 100;
  let out_features = 2;
  let tile_size = 16;

  let A = generate_random_matrix(batch_size, in_features);
  let B = generate_random_matrix(in_features, out_features);

  // let A = new Float32Array([
  //   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
  // ]);
  // let B = new Float32Array([
  //   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
  // ]);
  // let Y = new Float32Array([
  //   90, 100, 110, 116, 202, 228, 254, 272, 314, 356, 398, 428, 413, 470, 527,
  //   569,
  // ]);

  let result = await gemm(
    A,
    B,
    batch_size,
    in_features,
    out_features,
    tile_size
  );
}

async function testTensorFlowMLP() {
  const path =
    "https://jakobtroidl.github.io/data/model_3layers_256neurons/model.json";

  const loadedModel = await tf.loadLayersModel(path);
  let start = performance.now();
  const result = await loadedModel.predict(
    tf.tensor2d([[0, 0.625, 0.495, 0.165, 1.262, 0.507, 0.318, 0.39]])
  );
  let end = performance.now();
  console.log("tensorflow js Inference time: ", end - start, "ms");
  console.log("Ground Truth: " + result.dataSync());
}

async function testMLP() {
  let batch_size = 3000;
  let tile_size = 8; // must not be bigger than 16
  const path =
    "https://jakobtroidl.github.io/data/model_3layers_256neurons/model.json";

  console.log("tf object", tf);

  await tf.setBackend("webgpu");
  let tfjs_model = await from_tfjs(path);
  let model = await createMLP(tfjs_model, batch_size, tile_size);

  console.log(batch_size, model.inputSize, model.outputSize, model);
  let X = generate_random_matrix(batch_size, model.inputSize);

  let start = performance.now();
  let result = await model.inference(X); // result should be 11.881376266479492
  let end = performance.now();
  console.log("WebMLP Inference time + Data Transfer: ", end - start, "ms");
  console.log("result", result);

  const loadedModel = await tf.loadLayersModel(path);
  start = performance.now();
  result = await loadedModel.predict(
    tf.tensor2d(X, [batch_size, model.inputSize])
  );
  end = performance.now();
  console.log("Tensorflow Inference time + Data Transfer: ", end - start, "ms");
  console.log("Ground Truth: " + result.dataSync());
}

// testGemm();
// testTensorFlowMLP();
// testMLP();

export { createMLP, from_tfjs };
