[![npm version](https://img.shields.io/npm/v/web-mlp.svg?color=1a8cff)](https://www.npmjs.com/package/web-mlp)




# WebGPU accelerated fast MLP inference

## Install
```
npm i web-mlp
```

## Quick Start
```javascript
let batch_size = 70000;
let tile_size = 8; // must not be bigger than 16
const path = "https://jakobtroidl.github.io/data/trainedModel/model.json";

let tfjs_model = await from_tfjs(path);
let model = await createMLP(tfjs_model, batch_size, tile_size); 
let X = Float32Array.from(Array(batch_size * model.inputSize).fill(0), () => Math.random());

let result = await model.inference(X);
console.log("result", result);
```

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
