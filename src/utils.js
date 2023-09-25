// Sets the tile size in the shader
// Shader --> String
export function setTileSize(shader, size) {
  return shader.split("TILE_SIZE").join(size);
}

// Generates a random matrix of size w x h and returns it as a Float32Array
// values are in [0, 1]
export function generate_random_matrix(w, h) {
  return Float32Array.from(Array(w * h).fill(0), () => Math.random());
}

export const Activation = {
  None: 0,
  ReLU: 1,
  Sigmoid: 2,
  Tanh: 3,
};
