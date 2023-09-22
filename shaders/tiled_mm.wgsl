// Define the size of the tile
struct Params {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<storage> A: array<f32>;
@group(0) @binding(1) var<storage> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params : Params;

override tile_size: u32 = 16;

// initialize shared memory
var<workgroup> tileA : array<f32, tile_size * tile_size>;
var<workgroup> tileB : array<f32, tile_size * tile_size>;

@compute @workgroup_size(tile_size, tile_size)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>, 
    @builtin(local_invocation_id) local_id: vec3<u32>
    ) {
    var sum: f32 = 0.0;

    //let group_y: u32 = global_id.y; let group_x: u32 = global_id.x;
    let group_y: u32 = group_id.y; let group_x: u32 = group_id.x;
    let local_y: u32 = local_id.y; let local_x: u32 = local_id.x;

    let row: u32 = group_y * tile_size + local_y;
    let col: u32 = group_x * tile_size + local_x;

    //Loop over tiles
    for (var t = 0u; t < params.width / tile_size; t = t + 1u) {
        // Load data into shared memory
        tileA[local_y * params.width + local_x] = A[row * params.width + (t * tile_size + local_x)];
        tileB[local_y * params.width + local_x] = B[(t * tile_size + local_y) * params.width + col];
        workgroupBarrier();

        // Compute partial results
        for (var k = 0u; k < tile_size; k = k + 1u) {
            sum += tileA[local_y * params.width + k] * tileB[k * params.width + local_x];
            workgroupBarrier();
        }

    }
    C[row * params.width + col] = sum;   
}