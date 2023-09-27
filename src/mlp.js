class MLP {
  constructor(bla, bla2) {
    this.layers = [];
  }

  inference(data) {
    console.log("MLP inference");
  }
}

class Linear {
  constructor(weightsBuffer, biasBuffer, paramsBuffer) {
    console.log("Linear constructor");
    this.weightsBuffer = weightsBuffer;
    this.biasBuffer = biasBuffer;
    this.paramsBuffer = paramsBuffer;
  }

  inference(data) {
    console.log("Linear inference");
  }
}
