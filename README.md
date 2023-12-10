[![npm version](https://img.shields.io/npm/v/web-mlp.svg?color=1a8cff)](https://www.npmjs.com/package/web-mlp)

# WebGPU accelerated fast MLP inference

Hardware accelerated inference of multi-layer perceptions (MLPs) in the broswer. Works in [browsers supporting WebGPU](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status).

## Install

```
npm i web-mlp
```

## Quick Start

```javascript
import {
    initWebGPU,
    createMLP,
    from_json
} from "web-mlp";

async function testMLP() {
    let batch_size = 70000;
    let tile_size = 8;
    const path = "https://jakobtroidl.github.io/data/mlp-v8.json"; // path to example model's JSON file

    let device = await initWebGPU();
    let model_data = await from_json(path); // load tensorflow.js model
    let [model, outputBuffer] = await createMLP(model_data, batch_size, tile_size); // convert to web-mlp model for fast inference
    let X = Float32Array.from(Array(batch_size * model.inputSize).fill(0), () => Math.random()); // generate random a input

    let commandEncoder = device.createCommandEncoder();

    let start = performance.now();
    model.inference(X, commandEncoder); // inference the model
    device.queue.submit([commandEncoder.finish()]);
    let result = await model.transferToCPU(outputBuffer);
    let end = performance.now();

    console.log("WebMLP Inference time + Data Transfer: ", end - start, "ms");
    console.log("result", result);
}

testMLP();
```

Depending on your computing hardware, you can increase `batch_size` and `tile_size` . Tested on a MacBook Pro 2021 w/ Intel GPU, which supports up to `batch_size=800000` and `tile_size=32`. Check out [this website](https://webgpureport.org/) to view WebGPU limits for your own device. Also, [here's](https://github.com/jakobtroidl/webmlp-test) an example repository that uses `web-mlp`.

## PyTorch to WebMLP
WebMLP is purely designed for model inference. That means you can't train an MLP using this repository. We recommend training the MLP in PyTorch and exporting the model to work with WebMLP. Here, we describe how that process works. 

### WebMLP Input Format
WebMLP takes a JSON file as input. The file format is described below. [Here's](https://jakobtroidl.github.io/data/mlp-v8.json) an example of a small 3-hidden 64-neuron layer MLP in that file format. 
``` json5
{
    "input_shape": [
        null, 19
    ],
    "output_shape": [
        null, 1
    ],
    "activations": "Relu", // applied to all layers except the last layer. Options are [Relu, Sigmoid, Tanh, Linear]
    "layers": [
        {
            "weight": [ /* Linear Layer 1 weights as Float32Array. Ordered row-major. */ ],
            "weight_shape": [ 64, 19], /* output dimension, input dimension */
            "bias": [ /* Linear Layer 1 biases as Float32Array. Ordered row-major. */ ],
            "bias_shape": [ 64 ]
        }, 
        { ... }, // Layer 2
        { ... }  // Layer N
    ]
}
```
## Development

```sh
git clone https://github.com/jakobtroidl/web-mlp.git
cd web-mlp
npm install
npm run dev
```

## Publish to npm

```
// ensure all changes are comitted
npm run release-patch // release new patch version
npm run release-minor // release new minor version
npm run release-major // release new major version
```

## License

MIT license ([LICENSE-MIT](LICENSE-MIT)).
