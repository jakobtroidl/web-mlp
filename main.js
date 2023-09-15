import init, { Session, Input } from "@webonnx/wonnx-wasm";

async function fetchBytes(url) {
  const reply = await fetch(url);
  const blob = await reply.arrayBuffer();
  const arr = new Uint8Array(blob);
  return arr;
}

async function run() {
  try {
    let x = 32768;
    let y = 102;
    let model_path="https://jakobtroidl.github.io/data/mlp_divisible_by_2_simplified.onnx""

    const data = Float32Array.from({ length: x * y }, () => Math.random());
    const [modelBytes] = await Promise.all([
      fetchBytes(model_path),
    ]);

    await init();
    const session = await Session.fromBytes(modelBytes);

    //Start inference
    const input = new Input();
    input.insert("x", data);

    for (let i = 0; i < 100; i++) {
      console.time("session.run");
      let result = await session.run(input);
      console.timeEnd("session.run");
    }

    session.free();
    input.free();

  } catch (e) {
    console.error(e, e.toString());
  }
}

run();
