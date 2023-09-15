import init, { main, Session, Input } from "@webonnx/wonnx-wasm";

async function fetchBytes(url) {
  const reply = await fetch(url);
  const blob = await reply.arrayBuffer();
  const arr = new Uint8Array(blob);
  return arr;
}

async function run() {
  try {
    // Load model, labels file and WONNX
    const labels = fetch("./data/models/squeeze-labels.txt").then(r => r.text());
    const [modelBytes, initResult, labelsResult] = await Promise.all([fetchBytes("./data/models/opt-squeeze.onnx"), init(), labels])
    console.log("Initialized", { modelBytes, initResult, Session, labelsResult});
    const squeezeWidth = 224;
    const squeezeHeight = 224;

    // Start inference session
    const session = await Session.fromBytes(modelBytes);

    // Parse labels
    const labelsList = labelsResult.split(/\n/g);
    console.log({labelsList});

    // Start video
    const player = document.getElementById('player');
    const constraints = {
      video: true,
    };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    player.srcObject = stream;

    // Create a canvas to capture video frames
    const canvas = document.createElement('canvas');
    canvas.width = squeezeWidth;
    canvas.height = squeezeHeight;
    const context = canvas.getContext('2d', {willReadFrequently: true});

    let inferenceCount = 0;
    let inferenceTime = 0;

    // Captures a frame and produces inference
    async function inferImage() {
      try {
        // Draw the video frame to the canvas.
        context.drawImage(player, 0, 0, canvas.width, canvas.height);

        const data = context.getImageData(0, 0, canvas.width, canvas.height);
        const image = new Float32Array(224 * 224 * 3);

        // Transform the image data in the format expected by SqueezeNet
        const planes = 3; // SqueezeNet expects RGB
        const valuesPerPixel = 4; // source data is RGBA
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];

        for (let plane = 0; plane < planes; plane++) {
          for (let y = 0; y < squeezeHeight; y++) {
            for (let x = 0; x < squeezeWidth; x++) {
              const v = data.data[y * squeezeWidth * valuesPerPixel + x * valuesPerPixel + plane] / 255.0;
              image[plane * (squeezeWidth * squeezeHeight) + y * squeezeWidth + x] = (v - mean[plane]) / std[plane];
            }
          }
        }


        console.log({image});
        
        // Start inference
        const input = new Input();
        input.insert("data", image);
        const start = performance.now();
        const result = await session.run(input);
        const duration = performance.now() - start;
        inferenceCount++;
        inferenceTime += duration;
        input.free();

        // Find the label with the highest probability
        const probs = result.get("squeezenet0_flatten0_reshape0");
        let maxProb = -1;
        let maxIndex = -1;
        for (let index = 0; index < probs.length; index++) {
          const p = probs[index];
          if (p > maxProb) {
            maxProb = p;
            maxIndex = index;
          }
        }

        // Write result
        document.getElementById("log").innerText = `${labelsList[maxIndex]} (${maxProb})`;
        const avgFrameTime = inferenceTime / inferenceCount;
        document.getElementById("perf").innerText = `Inference time: ${avgFrameTime.toFixed(2)}ms, at most ${Math.floor(1000/avgFrameTime)} fps`;
      }
      catch (e) {
        console.error(e, e.toString());
      }
    }

    // Capture and infer as fast as the browser allows
    function tick() {
      window.requestAnimationFrame(async () => {
        console.time("frame");
        await inferImage();
        console.timeEnd("frame");
        tick();
      });
    }
    tick();
  }
  catch(e) {
    console.error(e, e.toString());
  }
}

run();