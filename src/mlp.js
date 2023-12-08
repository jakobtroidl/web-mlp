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
    /**
     * @param {Float32Array | GPUBuffer} data for forward pass. Can be either a WebGPU buffer or a Float32Array
     * @returns {GPUBuffer} output buffer, which can be mapped to a Float32Array using the transferToCPU method
     */
    if (data instanceof Float32Array) {
      // map data to first data buffer
      let input = this.layers[0].inputBuffer;
      this.device.queue.writeBuffer(input, 0, data, 0);
      await this.device.queue.onSubmittedWorkDone();
    } else {
      // data is an instance of GPUBuffer
      let input = this.layers[0].inputBuffer;
      this.device.queue.copyBufferToBuffer(data, 0, input, 0, data.byteLength);
      await this.device.queue.onSubmittedWorkDone();
    }

    let now = performance.now();

    for (let i = 0; i < this.layers.length; i++) {
      await this.layers[i].inference();
    }

    let end = performance.now();
    console.log("Inference time: ", end - now, "ms");

    let output = this.layers[this.layers.length - 1].outputBuffer;

    return output;
  }

  async transferToCPU(buffer, commandEncoder = undefined) {
    /**
     * Transfers a GPUBuffer to the CPU and returns data as Float32Array
     * @param {GPUBuffer} buffer to transfer to CPU
     * @param {GPUCommandEncoder} commandEncoder (optional) to use for the transfer
     */

    if (!commandEncoder) {
      commandEncoder = this.device.createCommandEncoder();
    }

    commandEncoder.copyBufferToBuffer(
      buffer,
      0, // Source offset
      this.stagingBuffer,
      0, // Destination offset
      outputBytes
    );

    this.device.queue.submit([commandEncoder.finish()]);

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
    index,
    device,
    bindGroup,
    inputBuffer,
    outputBuffer,
    computePipeline,
    inputSize,
    outputSize,
    batchSize,
    tile_size
  ) {
    this.index = index;
    this.device = device;
    this.bindGroup = bindGroup;
    this.inputBuffer = inputBuffer;
    this.outputBuffer = outputBuffer;
    this.computePipeline = computePipeline;
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.batchSize = batchSize;
    this.tile_size = tile_size;

    this.commandEncoder = this.device.createCommandEncoder();
    let passEncoder = this.commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.computePipeline);
    passEncoder.setBindGroup(0, this.bindGroup);
    passEncoder.dispatchWorkgroups(
      Math.ceil(this.outputSize / this.tile_size),
      Math.ceil(this.batchSize / this.tile_size)
    );
    passEncoder.end();
  }

  async inference() {
    this.device.queue.submit([this.commandEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
  }
}
