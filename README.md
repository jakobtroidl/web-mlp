[![npm version](https://img.shields.io/npm/v/web-mlp.svg?color=1a8cff)](https://www.npmjs.com/package/web-mlp)

# WebGPU accelerated fast MLP inference

Hardware accelerated inference of multi-layer perceptions (MLPs) in the browser. Works in [browsers supporting WebGPU](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status).

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
    let batch_size = 20;
    let tile_size = 8;
    const path = "https://jakobtroidl.github.io/data/mlp-v11.json"; // path to example model's JSON file

    let device = await initWebGPU();
    let model_data = await from_json(path); // load model
    let [model, outputBuffer] = await createMLP(model_data, batch_size, tile_size); // convert to WebMLP model for fast inference
    let X = Float32Array.from(Array(batch_size * model.inputSize).fill(0), () => Math.random()); // generate random a input

    let commandEncoder = device.createCommandEncoder();

    let start = performance.now();
    model.inference(X, commandEncoder); // inference the model
    device.queue.submit([commandEncoder.finish()]);
    let result = await model.transferToCPU(outputBuffer);
    let end = performance.now();

    console.log("WebMLP Inference time + Data Transfer: ", end - start, "ms");
    console.log("WebMLP result", result);
    console.log("WebMLP result should match dummy_output in model.json");
}

testMLP();
```

Depending on your computing hardware, you can increase `batch_size` and `tile_size` . Tested on a MacBook Pro 2021 w/ Intel GPU, which supports up to `batch_size=800000` and `tile_size=32` . Check out [this website](https://webgpureport.org/) to view WebGPU limits for your own device.

## PyTorch to WebMLP

WebMLP is purely designed for model inference. That means you can't use it to train an MLP. We recommend training the MLP in PyTorch and exporting the model to JSON for WebMLP inference. Here, we describe how that process works. 

### PyTorch export 

```python
import torch.nn as nn
import torch
import json

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, n_hidden, n_neurons):
        super().__init__()
        layers = []
        self.in_dim = in_dim
        self.out_dim = out_dim
        lastv = in_dim
        for i in range(n_hidden):
            layers.append(nn.Linear(lastv, n_neurons))
            layers.append(nn.ReLU())
            lastv = n_neurons
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

    def export(self, filename):
        # export model
        self.eval()

        dummy_input = torch.ones(1, self.in_dim).cuda()
        dummy_out = self(dummy_input)

        activation = "Relu"
        weights_and_biases = {}
        weights_and_biases['input_shape'] = [None, self.in_dim]
        weights_and_biases['output_shape'] = [None, self.out_dim]
        weights_and_biases['activations'] = activation
        weights_and_biases['dummy_input'] = dummy_input.cpu().detach().numpy().tolist()
        weights_and_biases['dummy_output'] = dummy_out.cpu().detach().numpy().tolist()

        layers = {}
        for name, param in self.named_parameters():
            name_parts = name.split('.')
            key = name_parts[0] + "." + name_parts[1]
            if key not in layers:
                layers[key] = {}
            param_np = param.cpu().detach().numpy()
            layers[key][name_parts[2]] = param_np.flatten(order="F").tolist()
            layers[key][name_parts[2] + '_shape'] = list(param_np.shape)

        sorted_keys = sorted(layers.keys())
        weights_and_biases['layers'] = [layers[key] for key in sorted_keys]

        # safe weights and biases as json
        with open(filename, 'w') as outfile:
            json.dump(weights_and_biases, outfile)
```

### WebMLP Input Format

WebMLP takes a JSON file as input. The file format is described below. [Here's](https://jakobtroidl.github.io/data/mlp-v11.json) an example of a small 3-hidden 64-neuron layer MLP in that file format. 

```json5
{
    "input_shape": [
        null, 19
    ],
    "output_shape": [
        null, 1
    ],
    "activations": "Relu", // applied to all layers except the last layer. Options are [Relu, Sigmoid, Tanh, Linear]
    "dummy_input": [[ /* Example Input */ ]],
    "dummy_output": [[ /* Expected Output for dummy input. Can be used to verify if inference works */ ]]
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
