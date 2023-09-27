export class MLP {
  constructor(layers) {
    this.layers = layers;
  }

  inference(data) {
    // map data to first data buffer

    for (let i = 0; i < this.layers.length; i++) {
      this.layers[i].inference();
    }

    return "Something has happened";
  }
}

export class Linear {
  constructor(
    device,
    computePipeline,
    layout,
    batchSize,
    out_features,
    tile_size,
    inputBuffer,
    weightsBuffer,
    biasBuffer,
    paramsBuffer,
    outputBuffer
  ) {
    this.device = device;
    this.computePipeline = computePipeline;
    this.inputBuffer = inputBuffer;
    this.weightsBuffer = weightsBuffer;
    this.biasBuffer = biasBuffer;
    this.paramsBuffer = paramsBuffer;
    this.outputBuffer = outputBuffer;

    this.bindGroup = device.createBindGroup({
      layout: layout,
      entries: [
        { binding: 0, resource: { buffer: this.inputBuffer } },
        { binding: 1, resource: { buffer: this.weightsBuffer } },
        // { binding: 2, resource: { buffer: this.biasBuffer } },
        { binding: 2, resource: { buffer: this.outputBuffer } },
        { binding: 3, resource: { buffer: this.paramsBuffer } },
      ],
    });

    this.commandEncoder = device.createCommandEncoder();
    this.passEncoder = this.commandEncoder.beginComputePass();
    this.passEncoder.setPipeline(this.computePipeline);
    this.passEncoder.setBindGroup(0, this.bindGroup);
    this.passEncoder.dispatchWorkgroups(
      Math.ceil(out_features / tile_size),
      Math.ceil(batchSize / tile_size)
    );
    this.passEncoder.end();
  }

  inference() {
    console.log("Linear inference");
    this.device.queue.submit([this.commandEncoder.finish()]);
  }
}
