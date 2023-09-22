const T=function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const o of document.querySelectorAll('link[rel="modulepreload"]'))u(o);new MutationObserver(o=>{for(const a of o)if(a.type==="childList")for(const n of a.addedNodes)n.tagName==="LINK"&&n.rel==="modulepreload"&&u(n)}).observe(document,{childList:!0,subtree:!0});function t(o){const a={};return o.integrity&&(a.integrity=o.integrity),o.referrerpolicy&&(a.referrerPolicy=o.referrerpolicy),o.crossorigin==="use-credentials"?a.credentials="include":o.crossorigin==="anonymous"?a.credentials="omit":a.credentials="same-origin",a}function u(o){if(o.ep)return;o.ep=!0;const a=t(o);fetch(o.href,a)}};T();var w=typeof globalThis!="undefined"?globalThis:typeof window!="undefined"?window:typeof global!="undefined"?global:typeof self!="undefined"?self:{},P={},b={},h={};Object.defineProperty(h,"__esModule",{value:!0});h.Shader=void 0;h.Shader=`
@group(0) @binding(0)
var<storage,read> array_a: array<vec4<f32>>;

@group(0) @binding(1)
var<storage,read> array_b: array<vec4<f32>>;

@group(0) @binding(2)
var<storage,read_write> array_c: array<vec4<f32>>;

struct CMeta {
  M: f32,
  N: f32,
  K: f32,
  MD4: f32,
  ND4: f32,
  KD4: f32,
  alpha: f32,
}

@group(0) @binding(3)
var<storage,read> cmeta: CMeta;

@compute @workgroup_size(8,8,1)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  var M: u32 = u32(cmeta.M);
  var N: u32 = u32(cmeta.N);
  var K: u32 = u32(cmeta.K);
  var MD4: u32 = u32(cmeta.KD4);
  var ND4: u32 = u32(cmeta.ND4);
  var KD4: u32 = u32(cmeta.KD4);
  var x: u32 = global_id.x;
  var y: u32 = global_id.y;
  if (x >= N || y >= M) {
    return;
  }
  var alpha: f32 = cmeta.alpha;
  var sum00: vec4<f32> = vec4<f32>();
  var sum01: vec4<f32> = vec4<f32>();
  var sum02: vec4<f32> = vec4<f32>();
  var sum03: vec4<f32> = vec4<f32>();
  var sum10: vec4<f32> = vec4<f32>();
  var sum11: vec4<f32> = vec4<f32>();
  var sum12: vec4<f32> = vec4<f32>();
  var sum13: vec4<f32> = vec4<f32>();
  for(var k: u32 = 0u; k < KD4; k = k + 1u) {
    var arow0: vec4<f32> = array_a[(y * 4u + 0u) * KD4 + k];
    var arow1: vec4<f32> = array_a[(y * 4u + 1u) * KD4 + k];
    var arow2: vec4<f32> = array_a[(y * 4u + 2u) * KD4 + k];
    var arow3: vec4<f32> = array_a[(y * 4u + 3u) * KD4 + k];
    var brow: vec4<f32>;
    brow = array_b[(k * 4u + 0u) * ND4 + x * 2u + 0u];
    sum00 = vec4<f32>(arow0.x) * brow + sum00;
    sum01 = vec4<f32>(arow1.x) * brow + sum01;
    sum02 = vec4<f32>(arow2.x) * brow + sum02;
    sum03 = vec4<f32>(arow3.x) * brow + sum03;
    brow = array_b[(k * 4u + 0u) * ND4 + x * 2u + 1u];
    sum10 = vec4<f32>(arow0.x) * brow + sum10;
    sum11 = vec4<f32>(arow1.x) * brow + sum11;
    sum12 = vec4<f32>(arow2.x) * brow + sum12;
    sum13 = vec4<f32>(arow3.x) * brow + sum13;
    
    brow = array_b[(k * 4u + 1u) * ND4 + x * 2u + 0u];
    sum00 = vec4<f32>(arow0.y) * brow + sum00;
    sum01 = vec4<f32>(arow1.y) * brow + sum01;
    sum02 = vec4<f32>(arow2.y) * brow + sum02;
    sum03 = vec4<f32>(arow3.y) * brow + sum03;
    brow = array_b[(k * 4u + 1u) * ND4 + x * 2u + 1u];
    sum10 = vec4<f32>(arow0.y) * brow + sum10;
    sum11 = vec4<f32>(arow1.y) * brow + sum11;
    sum12 = vec4<f32>(arow2.y) * brow + sum12;
    sum13 = vec4<f32>(arow3.y) * brow + sum13;
    
    brow = array_b[(k * 4u + 2u) * ND4 + x * 2u + 0u];
    sum00 = vec4<f32>(arow0.z) * brow + sum00;
    sum01 = vec4<f32>(arow1.z) * brow + sum01;
    sum02 = vec4<f32>(arow2.z) * brow + sum02;
    sum03 = vec4<f32>(arow3.z) * brow + sum03;
    brow = array_b[(k * 4u + 2u) * ND4 + x * 2u + 1u];
    sum10 = vec4<f32>(arow0.z) * brow + sum10;
    sum11 = vec4<f32>(arow1.z) * brow + sum11;
    sum12 = vec4<f32>(arow2.z) * brow + sum12;
    sum13 = vec4<f32>(arow3.z) * brow + sum13;
    
    brow = array_b[(k * 4u + 3u) * ND4 + x * 2u + 0u];
    sum00 = vec4<f32>(arow0.w) * brow + sum00;
    sum01 = vec4<f32>(arow1.w) * brow + sum01;
    sum02 = vec4<f32>(arow2.w) * brow + sum02;
    sum03 = vec4<f32>(arow3.w) * brow + sum03;
    brow = array_b[(k * 4u + 3u) * ND4 + x * 2u + 1u];
    sum10 = vec4<f32>(arow0.w) * brow + sum10;
    sum11 = vec4<f32>(arow1.w) * brow + sum11;
    sum12 = vec4<f32>(arow2.w) * brow + sum12;
    sum13 = vec4<f32>(arow3.w) * brow + sum13;
  }
  array_c[x * 2u + 0u + (y * 4u + 0u) * ND4] = sum00 * alpha;
  array_c[x * 2u + 0u + (y * 4u + 1u) * ND4] = sum01 * alpha;
  array_c[x * 2u + 0u + (y * 4u + 2u) * ND4] = sum02 * alpha;
  array_c[x * 2u + 0u + (y * 4u + 3u) * ND4] = sum03 * alpha;
  array_c[x * 2u + 1u + (y * 4u + 0u) * ND4] = sum10 * alpha;
  array_c[x * 2u + 1u + (y * 4u + 1u) * ND4] = sum11 * alpha;
  array_c[x * 2u + 1u + (y * 4u + 2u) * ND4] = sum12 * alpha;
  array_c[x * 2u + 1u + (y * 4u + 3u) * ND4] = sum13 * alpha;
}
`;var E={};Object.defineProperty(E,"__esModule",{value:!0});E.Shader=void 0;E.Shader=`
@group(0) @binding(0)
var<storage,read> array_a: array<f32>;

@group(0) @binding(1)
var<storage,read> array_b: array<f32>;

@group(0) @binding(2)
var<storage,read_write> array_c: array<f32>;

struct CMeta {
  M: f32,
  N: f32,
  K: f32,
  alpha: f32,
}

@group(0) @binding(3)
var<storage,read> cmeta: CMeta;

@compute @workgroup_size(8,8,1)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  var M: u32 = u32(cmeta.M);
  var N: u32 = u32(cmeta.N);
  var K: u32 = u32(cmeta.K);
  var x: u32 = global_id.x;
  var y: u32 = global_id.y;
  if (x >= N || y >= M) {
    return;
  }
  var sum: f32 = 0.0;
  for(var k: u32 = 0u; k < K; k = k + 1u) {
    sum = array_a[y * K + k] * array_b[k * N + x] + sum;
  }
  array_c[x + y * N] = sum * cmeta.alpha;
}
`;var _=w&&w.__awaiter||function(i,e,t,u){function o(a){return a instanceof t?a:new t(function(n){n(a)})}return new(t||(t=Promise))(function(a,n){function f(r){try{c(u.next(r))}catch(l){n(l)}}function s(r){try{c(u.throw(r))}catch(l){n(l)}}function c(r){r.done?a(r.value):o(r.value).then(f,s)}c((u=u.apply(i,e||[])).next())})};Object.defineProperty(b,"__esModule",{value:!0});b.sgemm=void 0;const U=h,I=E;class z{constructor(){this._initialized=!1,this.pipelineCache=new Map,this.isSupportedDevice=!1}init(){return _(this,void 0,void 0,function*(){if(!this._initialized){try{const e=yield navigator.gpu.requestAdapter();if(!e)throw new Error("requestAdapter failed");this._device=yield e.requestDevice(),this.isSupportedDevice=!0}catch(e){console.error("Unsupported device: ",e.message)}this._initialized=!0}})}createPipeline(e,t){const u=this._device,o=[];for(let c=0;c<t.length;c++)o.push({binding:c,visibility:GPUShaderStage.COMPUTE,buffer:{type:t[c]}});const a=u.createBindGroupLayout({entries:o}),n=u.createPipelineLayout({bindGroupLayouts:[a]}),f=u.createShaderModule({code:e}),s=u.createComputePipeline({layout:n,compute:{module:f,entryPoint:"main"}});return{bindGroupLayout:a,pipeline:s}}run(e){return _(this,void 0,void 0,function*(){const t=this._device;let u=[];const o=e.buffers.map((r,l)=>{if(l!==r.index)throw new Error("request.buffers is not sorted in order of index");let d=GPUBufferUsage.STORAGE;r.output&&(d|=GPUBufferUsage.COPY_SRC);const m=t.createBuffer({mappedAtCreation:!!r.input,size:r.length*Float32Array.BYTES_PER_ELEMENT,usage:d});if(r.output){const y=t.createBuffer({size:r.length*Float32Array.BYTES_PER_ELEMENT,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});u.push({src:m,dst:y,size:r.length*Float32Array.BYTES_PER_ELEMENT,name:r.name})}return m}),a=t.createBindGroup({layout:e.pipeline.bindGroupLayout,entries:e.buffers.map((r,l)=>({binding:l,resource:{buffer:o[l],size:r.length*Float32Array.BYTES_PER_ELEMENT}}))});for(let r=0;r<e.buffers.length;r++){const l=e.buffers[r];if(l.input){const d=o[r],m=d.getMappedRange(),y=new Float32Array(m),g=e.inputData[l.name];if(!g){console.error(`input array '${l.name}' is not supplied.`);continue}if(g.length!==y.length){console.error(`length of input array '${l.name}' does not match GPU buffer (${g.length} !== ${y.length}).`);continue}y.set(g),d.unmap()}}const n=t.createCommandEncoder(),f=n.beginComputePass();f.setBindGroup(0,a),f.setPipeline(e.pipeline.pipeline),f.dispatchWorkgroups(e.threadGroups.x,e.threadGroups.y,e.threadGroups.z),f.end?f.end():f.endPass();for(const r of u)n.copyBufferToBuffer(r.src,0,r.dst,0,r.size);t.queue.submit([n.finish()]);const s={};for(const r of u){yield r.dst.mapAsync(GPUMapMode.READ);const l=r.dst.getMappedRange(),d=new Float32Array(l),m=new Float32Array(d);r.dst.unmap(),r.dst.destroy(),s[r.name]=m}for(const r of o)r.destroy();return{outputData:s}})}}const p=new z;function O(i,e,t,u,o,a){return _(this,void 0,void 0,function*(){const n=U.Shader,f="sgemm_block";let s=p.pipelineCache.get(f);s||(s=p.createPipeline(n,["read-only-storage","read-only-storage","storage","read-only-storage"]),p.pipelineCache.set(f,s));const c={pipeline:s,buffers:[{index:0,name:"array_a",length:i*t,input:!0,output:!1},{index:1,name:"array_b",length:t*e,input:!0,output:!1},{index:2,name:"array_c",length:i*e,input:!1,output:!0},{index:3,name:"meta",length:7,input:!0,output:!1}],inputData:{array_a:o,array_b:a,meta:new Float32Array([i,e,t,i/4,e/4,t/4,u])},threadGroups:{x:e/64,y:i/32,z:1}};return(yield p.run(c)).outputData.array_c})}function F(i,e,t,u,o,a){return _(this,void 0,void 0,function*(){const n=I.Shader,f="sgemm_generic";let s=p.pipelineCache.get(f);s||(s=p.createPipeline(n,["read-only-storage","read-only-storage","storage","read-only-storage"]),p.pipelineCache.set(f,s));const c={pipeline:s,buffers:[{index:0,name:"array_a",length:i*t,input:!0,output:!1},{index:1,name:"array_b",length:t*e,input:!0,output:!1},{index:2,name:"array_c",length:i*e,input:!1,output:!0},{index:3,name:"meta",length:4,input:!0,output:!1}],inputData:{array_a:o,array_b:a,meta:new Float32Array([i,e,t,u])},threadGroups:{x:Math.ceil(e/8),y:Math.ceil(i/8),z:1}};return(yield p.run(c)).outputData.array_c})}function R(i,e,t,u,o,a,n=0,f){return _(this,void 0,void 0,function*(){if(n!==0)throw new Error("beta !== 0.0 is not yet supported");if(yield p.init(),!p.isSupportedDevice)throw new Error("unsupported device");return i%32===0&&e%64===0&&t%4===0&&u===1?O(i,e,t,u,o,a):F(i,e,t,u,o,a)})}b.sgemm=R;var x={};Object.defineProperty(x,"__esModule",{value:!0});x.sgemm=void 0;function K(i,e,t,u,o,a,n=0,f){if(n!==0)throw new Error("beta !== 0.0 is not yet supported");const s=new Float32Array(i*e);for(let c=0;c<i;c++)for(let r=0;r<e;r++){let l=0;for(let d=0;d<t;d++)l+=o[c*t+d]*a[d*e+r];s[c*e+r]=l*u}return s}x.sgemm=K;var Z=w&&w.__awaiter||function(i,e,t,u){function o(a){return a instanceof t?a:new t(function(n){n(a)})}return new(t||(t=Promise))(function(a,n){function f(r){try{c(u.next(r))}catch(l){n(l)}}function s(r){try{c(u.throw(r))}catch(l){n(l)}}function c(r){r.done?a(r.value):o(r.value).then(f,s)}c((u=u.apply(i,e||[])).next())})};Object.defineProperty(P,"__esModule",{value:!0});P.sgemm=void 0;const Y=b,B=x;let L=!1;function j(i,e,t,u,o,a,n=0,f){return Z(this,void 0,void 0,function*(){if(L)return B.sgemm(i,e,t,u,o,a,n,f);let s=null;try{s=yield Y.sgemm(i,e,t,u,o,a,n,f)}catch(c){console.warn("Error using WebGPU; fallback to pure JavaScript",c)}return s===null?(L=!0,B.sgemm(i,e,t,u,o,a,n,f)):s})}P.sgemm=j;var $=`// Define the size of the tile
const TILE_SIZE: u32 = 16;
const width: u32 = 4096;
const height: u32 = 4096;

@group(0) @binding(0) var<storage> A: array<f32>;
@group(0) @binding(1) var<storage> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

// initialize shared memory
var<workgroup> tileA : array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> tileB : array<array<f32, TILE_SIZE>, TILE_SIZE>;

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
    for (var t = 0u; t < width / TILE_SIZE; t = t + 1u) {
        // Load data into shared memory
        tileA[local_y][local_x] = A[row * width + (t * TILE_SIZE + local_x)];
        tileB[local_y][local_x] = B[(t * TILE_SIZE + local_y) * width + col];
        workgroupBarrier();

        // Compute partial results
        for (var k = 0u; k < TILE_SIZE; k = k + 1u) {
            sum += tileA[local_y][k] * tileB[k][local_x];
            workgroupBarrier();
        }
    }
    C[row * width + col] = sum;   
}
`;function G(i,e){return Float32Array.from(Array(i*e).fill(0),()=>Math.random())}async function W(i,e,t,u){const a=await(await navigator.gpu.requestAdapter()).requestDevice();console.log("Limits: ",a.limits);const n=$,f=a.createShaderModule({code:n}),s=new Float32Array(t*u).fill(0),[c,r,l]=[i,e,s].map(C=>a.createBuffer({size:C.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,mappedAtCreation:!0})),d=a.createBuffer({size:s.byteLength,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});new Float32Array(c.getMappedRange()).set(i),new Float32Array(r.getMappedRange()).set(e),new Float32Array(l.getMappedRange()).set(s),c.unmap(),r.unmap(),l.unmap();const m=a.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),y=a.createBindGroup({layout:m,entries:[{binding:0,resource:{buffer:c}},{binding:1,resource:{buffer:r}},{binding:2,resource:{buffer:l}}]}),g=a.createComputePipeline({layout:a.createPipelineLayout({bindGroupLayouts:[m]}),compute:{module:f,entryPoint:"main"}});let k=performance.now();const M=a.createCommandEncoder(),v=M.beginComputePass();v.setPipeline(g),v.setBindGroup(0,y),v.dispatchWorkgroups(Math.ceil(t/16),Math.ceil(u/16)),v.end(),M.copyBufferToBuffer(l,0,d,0,s.byteLength),a.queue.submit([M.finish()]);let N=performance.now();console.log("Time: ",N-k,"ms"),await d.mapAsync(GPUMapMode.READ,0,s.byteLength);const A=d.getMappedRange(0,s.byteLength).slice();return d.unmap(),console.log(new Float32Array(A)),new Float32Array(A)}let D=4096,S=4096,q=G(D,S),J=G(D,S);W(q,J,D,S);
