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




export function mlp_test_data(batch_size)
{
  return new Float32Array([
    0, 0.625, 0.495, 0.165, 1.262, 0.507, 0.318, 0.39,
    // 0, 0.625, 0.495, 0.165, 1.262, 0.507, 0.318, 0.39,
    // 0, 0.625, 0.495, 0.165, 1.262, 0.507, 0.318, 0.39,
    // 0, 0.625, 0.495, 0.165, 1.262, 0.507, 0.318, 0.39
  ]);
}

export function linearizeRowMajor(twoDArray) {
  return twoDArray.reduce((acc, row) => acc.concat(row), []);
}

function gemm_cpu(A, B, rowsA, colsA, colsB) {
  let start = performance.now();
  if (!A || !B || !rowsA || !colsA || !colsB) return null;

  const C = new Float32Array(rowsA * colsB).fill(0.0);

  for (let i = 0; i < rowsA; i++) {
    for (let j = 0; j < colsB; j++) {
      let sum = 0;
      for (let k = 0; k < colsA; k++) {
        sum += A[i * colsA + k] * B[k * colsB + j];
      }
      C[i * colsB + j] = sum;
    }
  }
  let end = performance.now();

  console.log("CPU Time: ", end - start, "ms");
  console.log("CPU result: ", C);

  return C;
}

export const Activation = {
  None: 0,
  ReLU: 1,
  Sigmoid: 2,
  Tanh: 3,
};

export function getActivation(act_string) {
  switch (act_string) {
    case "ReLU":
      return Activation.ReLU;
    case "Sigmoid2":
      return Activation.Sigmoid;
    case "Tanh":
      return Activation.Tanh;
    case "Linear":
      return Activation.None;
    default:
      return console.error("Activation not supported");
  }
}
