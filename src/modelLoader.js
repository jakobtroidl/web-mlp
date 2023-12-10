export async function from_json(path) {
  /**
   * @param {string} path is a url to a JSON file. See documentation on how the JSON file needs to be structured
   * @returns {Array} of layer objects describing the MLP
   */
  const response = await fetch(path);
  const mlp_data = await response.json();

  let out_layers = [];
  try {
    let i = 0;
    mlp_data.layers.forEach((layer) => {
      let act = mlp_data.activations;
      if (i == mlp_data.layers.length - 1) {
        // last layer has no activations
        act = "Linear";
      }
      const layerObject = {
        weights: new Float32Array(layer.weight),
        weight_shape: layer.weight_shape,
        biases: new Float32Array(layer.bias),
        bias_shape: layer.bias_shape,
        activation: act,
      };

      out_layers.push(layerObject);
      i++;
    });
    return out_layers;
  } catch (e) {
    console.error(e);
  }
  return null;
}

// // must be a tfjs based model, only including linear/dense layers and activations
// export async function from_tfjs(path) {
//   // download JSON file from path
//   const response = await fetch(path);
//   const mlp_json = await response.json();
//   console.log("model_json", mlp_json);

//   const model = await tf.loadLayersModel(path);
//   let layers = [];

//   console.log("Layers: ", model.layers);

//   try {
//     model.layers.forEach((layer) => {
//       if (layer.getClassName() != "Dense") {
//         throw new Error("Only dense layers are supported in WebMLP");
//       }
//       const w = layer.getWeights();

//       // collect weights
//       let w_tensor = w[0];
//       let w_array = new Float32Array(linearizeRowMajor(w_tensor.dataSync()));
//       // collect biases
//       let b_tensor = w[1];
//       if (b_tensor == null) {
//         b_tensor = tf.zeros([w_tensor.shape[1]]);
//       }
//       let b_array = new Float32Array(linearizeRowMajor(b_tensor.dataSync()));

//       const layerObject = {
//         weights: w_array,
//         weight_shape: w_tensor.shape,
//         biases: b_array,
//         bias_shape: b_tensor.shape,
//         activation: layer.activation.constructor.name,
//       };

//       layers.push(layerObject);
//     });
//     return layers;
//   } catch (e) {
//     console.error(e);
//   }

//   return null;
// }
