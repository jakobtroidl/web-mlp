[![npm version](https://img.shields.io/npm/v/web-mlp.svg?color=1a8cff)](https://www.npmjs.com/package/web-mlp)

# WebGPU accelerated fast MLP inference
Hardware accelerated inference of [tensorflow.js](https://www.tensorflow.org/js) multi-layer perceptrons. Works in [browsers supporting WebGPU](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status).

## Install

```
npm i web-mlp
```

## Quick Start

```javascript
import { createMLP, from_tfjs } from "web-mlp";

async function testMLP() {
  let batch_size = 70000;
  let tile_size = 8; // must not be bigger than 16

  const path = "https://jakobtroidl.github.io/data/trainedModelOriginal/model.json"; // path to tensorflow.js model
  let tfjs_model = await from_tfjs(path); // load tensorflow.js model
  let model = await createMLP(tfjs_model, batch_size, tile_size); // convert to web-mlp model for fast inference
  let X = Float32Array.from(Array(batch_size * model.inputSize).fill(0), () => Math.random()); // generate random a input

  let result = await model.inference(X); // inference the model
  console.log("result", result); // print result
}

testMLP();
```
Depending on your computing hardware, you can increase `batch_size` and `tile_size`. Tested on a MacBook Pro 2021 w/ Intel GPU, which supports up to `batch_size=800000` and `tile_size=32`. Check out [this website](https://webgpureport.org/) to view WebGPU limits for your own device. Also, [here's](https://github.com/jakobtroidl/webmlp-test) an example repository that uses `web-mlp`.

## Running

```sh
npm install
npm run dev
```

## Building

```sh
npm run build
```

## Pubish to npm

```
npm run build
npm pack
npm login
npm publish
```

## License

Licensed under either of
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
at your option.
