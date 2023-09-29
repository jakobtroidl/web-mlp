// config file that defines an example multi-layer perceptron
export const mlp_config = {
  params: {
    n_inputs: 64,
    n_outputs: 4,
    activation: "ReLU", // "ReLU", "Sigmoid", "Tanh"
    output_activation: "None",
    n_neurons: 256, // 64, 128, 256, 512
    n_layers: 4, // 1, 2, 3, 4
  },
  weights: [
    [0.0], // add weights here as Float32Array
    [0.0], // add weights here as Float32Array
    [0.0], // add weights here as Float32Array
    [0.0], // add weights here as Float32Array
  ],
  biases: [
    [0.0], // add biases here as Float32Array
    [0.0], // add biases here as Float32Array
    [0.0], // add biases here as Float32Array
    [0.0], // add biases here as Float32Array
  ],
};
