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
    //@builtin(workgroup_id) group_id: vec3<u32>, 
    @builtin(local_invocation_id) local_id: vec3<u32>
    ) {
    var sum: f32 = 0.0;

    // Shared memory for the tile
    var tileA: array<f32, TILE_SIZE>;
    var tileB: array<f32, TILE_SIZE>;

    let row: u32 = global_id.y;
    let col: u32 = global_id.x;
    let localRow: u32 = local_id.y;
    let localCol: u32 = local_id.x;

    // Loop over tiles
    for (var t = 0u; t < width; t = t + TILE_SIZE) {
        // Load data into shared memory
        tileA[localRow * TILE_SIZE + localCol] = A[row * width + t + localCol];
        tileB[localRow * TILE_SIZE + localCol] = B[(t + localRow) * width + col];
        workgroupBarrier();

        // Compute partial results
        for (var k = 0u; k < TILE_SIZE; k = k + 1u) {
            sum += tileA[localRow * TILE_SIZE + k] * tileB[k * TILE_SIZE + localCol];
        }
        workgroupBarrier();
    }

    // Write results to the output matrix
    C[row * width + col] = sum;
}
