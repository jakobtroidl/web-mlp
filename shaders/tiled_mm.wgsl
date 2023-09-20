// Define the size of the tile
const TILE_SIZE: u32 = 16;
const width: u32 = 1024;
const height: u32 = 1024;

// struct Matrix {
//     width: u32,
//     height: u32,
//     data: array<f32>,
// };


@group(0) @binding(0) var<storage> A: array<f32>;
@group(0) @binding(1) var<storage> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>, 
    @builtin(local_invocation_id) local_id: vec3<u32>
    ) {
    var sum: f32 = 0.0;

    // Shared memory for the tile
    var tileA: array<f32, TILE_SIZE * TILE_SIZE>;
    var tileB: array<f32, TILE_SIZE * TILE_SIZE>;

    // int bx = blockIdx.x; int by = blockIdx.y;
    // int tx = threadIdx.x; int ty = threadIdx.y;

    let group_y: u32 = group_id.y; let group_x: u32 = group_id.x;
    let local_y: u32 = local_id.y; let local_x: u32 = local_id.x;

    let row: u32 = group_y * TILE_SIZE + local_y;
    let col: u32 = group_x * TILE_SIZE + local_x;

    //Loop over tiles
    for (var t = 0u; t < width / TILE_SIZE; t = t + 1u) {
        // Load data into shared memory
        tileA[local_y * local_x] = A[row * width + (t * TILE_SIZE + local_y)];
        tileB[local_y * local_x] = B[(t * TILE_SIZE + local_y) * width + col];
        workgroupBarrier();

        // Compute partial results
        for (var k = 0u; k < TILE_SIZE; k = k + 1u) {
            sum += tileA[local_y + k] * tileB[k * local_x];
        }
        workgroupBarrier();
    }

    // Write results to the output matrix
    C[row * width + col] = sum;
}
