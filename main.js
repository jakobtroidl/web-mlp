import tiled_mm from "./src/shaders/tiled_mm.wgsl";
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

  console.log("tile_size", ts);

  const device = await adapter.requestDevice();
  const wgslCode = setTileSize(tiled_mm, ts); // Replace this with your actual WGSL code
  const shaderModule = device.createShaderModule({ code: wgslCode });

  return { device, shaderModule };
}

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
  console.log("model: ", model);
  let dataBuffers = [];
  let n_buffers = model.length + 1;

  console.log("n_buffers: ", n_buffers);
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
    let data = new Float32Array(bufferSize).fill(0);
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
    buffer.unmap();
  } else {
    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();
  }
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
  let weightBuffers = tf_model.map((layer) =>
    createGPUBuffer(device, layer.weights)
  );
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

// async function linear(x, weights, batchSize, in_features, out_features, ts) {
//   // // Initialize WebGPU
//   const { device, shaderModule } = await initWebGPU(ts);

//   const y = new Float32Array(batchSize * out_features).fill(0);

//   // Create params buffer and write data to it
//   const paramsBuffer = device.createBuffer({
//     size: 4 * 4, // 2 uint32s of 4 bytes each (width, height, activation)
//     usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
//     mappedAtCreation: true,
//   });

//   new Uint32Array(paramsBuffer.getMappedRange()).set([
//     batchSize,
//     in_features,
//     out_features,
//     Activation.ReLU,
//   ]);
//   paramsBuffer.unmap();

//   // Create and populate buffers
//   const [xBuffer, weightsBuffer, yBuffer] = [x, weights, y].map((arr) =>
//     device.createBuffer({
//       size: arr.byteLength,
//       usage:
//         GPUBufferUsage.STORAGE |
//         GPUBufferUsage.COPY_SRC |
//         GPUBufferUsage.COPY_DST,
//       mappedAtCreation: true,
//     })
//   );

//   // staging buffer to make data accessible to CPU
//   const stagingBuffer = device.createBuffer({
//     size: y.byteLength,
//     usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
//   });

//   new Float32Array(xBuffer.getMappedRange()).set(x);
//   new Float32Array(weightsBuffer.getMappedRange()).set(weights);
//   new Float32Array(yBuffer.getMappedRange()).set(y);
//   xBuffer.unmap();
//   weightsBuffer.unmap();
//   yBuffer.unmap();

//   const bindGroupLayout = device.createBindGroupLayout({
//     entries: [
//       {
//         binding: 0,
//         visibility: GPUShaderStage.COMPUTE,
//         buffer: {
//           type: "read-only-storage",
//         },
//       },
//       {
//         binding: 1,
//         visibility: GPUShaderStage.COMPUTE,
//         buffer: {
//           type: "read-only-storage",
//         },
//       },
//       {
//         binding: 2,
//         visibility: GPUShaderStage.COMPUTE,
//         buffer: {
//           type: "storage",
//         },
//       },
//       {
//         binding: 3,
//         visibility: GPUShaderStage.COMPUTE,
//         buffer: {
//           type: "uniform",
//         },
//       },
//     ],
//   });

//   let start = performance.now();

//   // Bind group
//   const bindGroup = device.createBindGroup({
//     layout: bindGroupLayout,
//     entries: [
//       { binding: 0, resource: { buffer: xBuffer } },
//       { binding: 1, resource: { buffer: weightsBuffer } },
//       { binding: 2, resource: { buffer: yBuffer } },
//       { binding: 3, resource: { buffer: paramsBuffer } },
//     ],
//   });

//   // Create pipeline
//   const pipeline = device.createComputePipeline({
//     layout: device.createPipelineLayout({
//       bindGroupLayouts: [bindGroupLayout],
//     }),
//     compute: {
//       module: shaderModule,
//       entryPoint: "main",
//     },
//   });

//   // Command encoder and pass
//   const commandEncoder = device.createCommandEncoder();
//   const passEncoder = commandEncoder.beginComputePass();
//   passEncoder.setPipeline(pipeline);
//   passEncoder.setBindGroup(0, bindGroup);
//   passEncoder.dispatchWorkgroups(
//     Math.ceil(out_features / ts),
//     Math.ceil(batchSize / ts)
//   );
//   passEncoder.end();

//   // Copy output buffer to staging buffer
//   commandEncoder.copyBufferToBuffer(
//     yBuffer,
//     0, // Source offset
//     stagingBuffer,
//     0, // Destination offset
//     y.byteLength
//   );

//   // Submit and execute
//   device.queue.submit([commandEncoder.finish()]);

//   let end = performance.now();

//   console.log("GPU Time: ", end - start, "ms");

//   // map staging buffer to read results back to JS
//   await stagingBuffer.mapAsync(
//     GPUMapMode.READ,
//     0, // Offset
//     y.byteLength // Length
//   );

// const copyArrayBuffer = stagingBuffer.getMappedRange(0, y.byteLength);
// const data = copyArrayBuffer.slice();
// stagingBuffer.unmap();

// console.log("GPU result: ", new Float32Array(data));

// return new Float32Array(data);
// }

async function testMLP() {
  let batch_size = 30000;
  let tile_size = 16; // must not be bigger than 16
  const path = "https://jakobtroidl.github.io/data/trainedModel/model.json";
  let tfjs_model = await from_tfjs(path);
  let model = await createMLP(tfjs_model, batch_size, tile_size);

  console.log(batch_size, model.inputSize, model.outputSize);
  let X = generate_random_matrix(batch_size, model.inputSize);
  let result = await model.inference(X);
  console.log("result", result);
}

testMLP();
