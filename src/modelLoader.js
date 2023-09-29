import * as tf from "@tensorflow/tfjs";
import { linearizeRowMajor } from "./utils";

// must be a tfjs based model, only including linear/dense layers and activations
export async function from_tfjs(path) {
  const model = await tf.loadLayersModel(path);

  let weights = [];
  let biases = [];
  let activations = [];
  let weight_shapes = [];
  let bias_shapes = [];

  try {
    model.layers.forEach((layer) => {
      if (layer.getClassName() != "Dense") {
        throw new Error("Only dense layers are supported in WebMLP");
      }

      const w = layer.getWeights();

      // collect weights
      let w_tensor = w[0];
      let w_array = new Float32Array(linearizeRowMajor(w_tensor.dataSync()));
      weights.push(w_array);
      weight_shapes.push(w_tensor.shape);

      // collect biases
      let b_tensor = w[1];
      let b_array = new Float32Array(linearizeRowMajor(b_tensor.dataSync()));
      biases.push(b_array);
      bias_shapes.push(b_tensor.shape);

      // collect activations
      activations.push(layer.activation);
    });
    return [weights, weight_shapes, biases, bias_shapes, activations];
  } catch (e) {
    console.error(e);
  }

  return null;
}
