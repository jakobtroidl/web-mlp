import * as tf from "@tensorflow/tfjs";
import { linearizeRowMajor } from "./utils";

// must be a tfjs based model, only including linear/dense layers and activations
export async function from_tfjs(path) {
  const model = await tf.loadLayersModel(path);
  let layers = [];

  try {
    model.layers.forEach((layer) => {
      if (layer.getClassName() != "Dense") {
        throw new Error("Only dense layers are supported in WebMLP");
      }
      const w = layer.getWeights();

      // collect weights
      let w_tensor = w[0];
      let w_array = new Float32Array(linearizeRowMajor(w_tensor.dataSync()));
      // collect biases
      let b_tensor = w[1];
      let b_array = new Float32Array(linearizeRowMajor(b_tensor.dataSync()));

      const layerObject = {
        weights: w_array,
        weight_shape: w_tensor.shape,
        biases: b_array,
        bias_shape: b_tensor.shape,
        activation: layer.activation.constructor.name,
      };

      layers.push(layerObject);
    });
    return layers;
  } catch (e) {
    console.error(e);
  }

  return null;
}
