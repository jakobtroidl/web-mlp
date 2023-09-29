export class MLP {
  constructor(device, layers) {
    this.device = device;
    this.layers = layers;
    this.output;

    this.outputSize = layers[layers.length - 1].outputSize;
    this.batchSize = layers[layers.length - 1].batchSize;
    this.inputSize = layers[0].inputSize;

    this.stagingBuffer = device.createBuffer({
      size: this.outputSize * this.batchSize * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  async inference(data) {
    let outputBytes = this.outputSize * this.batchSize * 4;
    let commandEncoder = this.device.createCommandEncoder();
    let now = performance.now();
    // map data to first data buffer

    for (let i = 0; i < this.layers.length; i++) {
      this.layers[i].inference();
    }

    commandEncoder.copyBufferToBuffer(
      this.layers[this.layers.length - 1].outputBuffer,
      0, // Source offset
      this.stagingBuffer,
      0, // Destination offset
      outputBytes
    );

    let end = performance.now();
    console.log("Inference time: ", end - now, "ms");

    // map staging buffer to read results back to JS
    await this.stagingBuffer.mapAsync(
      GPUMapMode.READ,
      0, // Offset
      outputBytes // Length
    );

    const copyArrayBuffer = this.stagingBuffer.getMappedRange(0, outputBytes);
    const output = copyArrayBuffer.slice();
    this.stagingBuffer.unmap();

    return new Float32Array(output);
  }
}

export class Linear {
  constructor(
    device,
    bindGroup,
    outputBuffer,
    computePipeline,
    inputSize,
    outputSize,
    batchSize,
    tile_size
  ) {
    this.device = device;
    this.bindGroup = bindGroup;
    this.outputBuffer = outputBuffer;
    this.computePipeline = computePipeline;
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.batchSize = batchSize;
    this.tile_size = tile_size;
  }

  inference() {
    let commandEncoder = this.device.createCommandEncoder();
    let passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.computePipeline);
    passEncoder.setBindGroup(0, this.bindGroup);
    passEncoder.dispatchWorkgroups(
      Math.ceil(this.outputSize / this.tile_size),
      Math.ceil(this.batchSize / this.tile_size)
    );
    passEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);
  }
}
