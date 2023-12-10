export class MLP {
  constructor(device, layers) {
    this.device = device;
    this.layers = layers;
    this.output;

    this.outputSize = layers[layers.length - 1].outputSize;
    this.batchSize = layers[layers.length - 1].batchSize;
    this.inputSize = layers[0].inputSize;
  }

  inference(data, commandEncoder = undefined) {
    /**
     * @param {Float32Array | GPUBuffer} data for forward pass. Can be either a WebGPU buffer or a Float32Array
     * @returns {GPUBuffer} output buffer, which can be mapped to a Float32Array using the transferToCPU method
     */

    if (!commandEncoder) {
      commandEncoder = this.device.createCommandEncoder();
    }

    if (data instanceof Float32Array) {
      // map data to first data buffer
      let input = this.layers[0].inputBuffer;
      this.device.queue.writeBuffer(input, 0, data, 0);
    } else {
      // data is an instance of GPUBuffer
      let input = this.layers[0].inputBuffer;
      commandEncoder.copyBufferToBuffer(data, 0, input, 0, data.size);
    }

    for (let i = 0; i < this.layers.length; i++) {
      this.layers[i].inference(commandEncoder);
    }
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

    let outputBytes = this.outputSize * this.batchSize * 4;

    let stagingBuffer = this.device.createBuffer({
      size: outputBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    commandEncoder.copyBufferToBuffer(
      buffer,
      0, // Source offset
      stagingBuffer,
      0, // Destination offset
      outputBytes // Size
    );

    this.device.queue.submit([commandEncoder.finish()]);

    // map staging buffer to read results back to JS
    await stagingBuffer.mapAsync(
      GPUMapMode.READ,
      0, // Offset
      outputBytes // Length
    );

    const copyArrayBuffer = stagingBuffer.getMappedRange(0, outputBytes);
    const output = copyArrayBuffer.slice();
    stagingBuffer.unmap();

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
  }

  inference(commandEncoder) {
    // this.device.queue.submit([this.commandEncoder.finish()]);
    // await this.device.queue.onSubmittedWorkDone();

    //this.commandEncoder = this.device.createCommandEncoder();
    let passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.computePipeline);
    passEncoder.setBindGroup(0, this.bindGroup);
    passEncoder.dispatchWorkgroups(
      Math.ceil(this.outputSize / this.tile_size),
      Math.ceil(this.batchSize / this.tile_size)
    );
    passEncoder.end();
  }
}
