// Define the size of the tile
struct Params {
    batch_size: u32,
    in_features: u32,
    out_features: u32,
    activation: u32, // None: 0, ReLU: 1, Sigmoid: 2, Tanh: 3
}

fn activation(x: f32, activation: u32) -> f32 {
    if activation == 0u {          // No activation
        return x;
    } else if activation == 1u {   // ReLU activation
        return max(0.0, x);
    } else if activation == 2u {   // Sigmoid activation
        return 1.0 / (1.0 + exp(-x));
    } else if activation == 3u {   // Tanh activation
        return tanh(x);
    }
    return x;
}

@group(0) @binding(0) var<storage> X: array<f32>; // input matrix
@group(0) @binding(1) var<storage> W: array<f32>; // weight matrix
@group(0) @binding(2) var<storage> B: array<f32>; // bias matrix
@group(0) @binding(3) var<storage, read_write> Y: array<f32>; // output matrix
@group(0) @binding(4) var<uniform> params : Params;

var<workgroup> tileX : array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> tileW : array<array<f32, TILE_SIZE>, TILE_SIZE>;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>, 
    @builtin(local_invocation_id) local_id: vec3<u32>
    ) {
    var sum: f32 = 0.0;

    //let group_y: u32 = global_id.y; let group_x: u32 = global_id.x;
    let group_y: u32 = group_id.y; let group_x: u32 = group_id.x;
    let local_y: u32 = local_id.y; let local_x: u32 = local_id.x;

    let row: u32 = group_y * TILE_SIZE + local_y;
    let col: u32 = group_x * TILE_SIZE + local_x;

    //Loop over tiles
    for (var t = 0u; t < ( TILE_SIZE + params.in_features - 1) / TILE_SIZE; t = t + 1u) {
        // Load data into shared memory
        
        let checkXCols: bool = t * TILE_SIZE + local_x < params.in_features;
        let checkXRows: bool = row < params.batch_size;

        if (checkXCols && checkXRows) {
            tileX[local_y][local_x] = X[row * params.in_features + (t * TILE_SIZE + local_x)];
        } else {
             tileX[local_y][local_x] = 0.0;
        }

        let checkWRows: bool = t * TILE_SIZE + local_y < params.in_features;
        let checkWCols: bool = col < params.out_features;

        if (checkWRows && checkWCols) {
            tileW[local_y][local_x] = W[(t * TILE_SIZE + local_y) * params.out_features + col];
        } else {
             tileW[local_y][local_x] = 0.0;
        }
        workgroupBarrier();

        // Compute partial results
        for (var k = 0u; k < TILE_SIZE; k = k + 1u) {
            sum += tileX[local_y][k] * tileW[k][local_x];
        }
        workgroupBarrier();
    }

    let checkYRows: bool = row < params.batch_size;
    let checkYCols: bool = col < params.out_features;

    if (checkYRows && checkYCols) {
         Y[row * params.out_features + col] = sum;   
    }
}