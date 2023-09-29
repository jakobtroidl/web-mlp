export class MLP {
  constructor(layers) {
    this.layers = layers;
  }

  inference(data) {
    let now = performance.now();
    // map data to first data buffer

    for (let i = 0; i < this.layers.length; i++) {
      this.layers[i].inference();
    }

    let end = performance.now();
    console.log("Inference time: ", end - now, "ms");

    return "Something has happened";
  }

  getInputSize() {
    return this.layers[0].inputSize;
  }

  getOutputSize() {
    return this.layers[this.layers.length - 1].outputSize;
  }
}

export class Linear {
  constructor(
    device,
    bindGroup,
    computePipeline,
    inputSize,
    outputSize,
    batchSize,
    tile_size
  ) {
    this.device = device;
    this.bindGroup = bindGroup;
    this.computePipeline = computePipeline;
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.batchSize = batchSize;
    this.tile_size = tile_size;
  }

  inference() {
    console.log("Linear inference");

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
