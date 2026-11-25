// The agent: builds the observation, runs the ONNX Q-network, and applies the action mask in JS.
// The model is fully convolutional, so the same file runs on any board size.

/** Same semantics as mask_q: illegal (mask == 0) cells become -Infinity. Returns a new Float32Array. */
export function maskQ(q, mask) {
  const out = new Float32Array(q.length);
  for (let i = 0; i < q.length; i++) out[i] = mask[i] ? q[i] : -Infinity;
  return out;
}

/** Index of the largest value (first one on ties, like torch.argmax). Returns -1 if all are -Infinity. */
export function argmax(values) {
  let best = -1, bestV = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > bestV) { bestV = values[i]; best = i; }
  }
  return best;
}

export class Agent {
  /** @param ort the onnxruntime-web namespace  @param session an InferenceSession */
  constructor(ort, session) {
    this.ort = ort;
    this.session = session;
  }

  static async load(ort, modelUrl) {
    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    return new Agent(ort, session);
  }

  /** Raw (unmasked) Q-values, Float32Array of length H*W (row-major). */
  async rawQ(board) {
    const input = new this.ort.Tensor("float32", board.observation(), [1, 10, board.height, board.width]);
    const out = await this.session.run({ obs: input });
    return out.q.data;
  }

  /** Masked Q-values (revealed cells = -Infinity) and the greedy action (flat index). */
  async evaluate(board) {
    const q = maskQ(await this.rawQ(board), board.actionMask());
    return { q, action: argmax(q) };
  }
}
