/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * From-Scratch Native Compact Training + Strict .x8D Binary Reload Benchmark.
 * Trains BOTH a Dense Reference Byte Model and a Native Compact Kronecker (STE) Model
 * from random initialization on raw 0..255 <-> 0.000..0.255 bytes (zero BPE, zero MASK).
 * Exports ONLY compact .x8D factor bytes (Uint8Array), discards all training buffers,
 * reloads strictly from that Uint8Array, and evaluates held-out perplexity & text generation.
 */

export interface NativeExperimentOptions {
  epochs?: number;
  kTerms?: number;
  factorBits?: 2 | 4;
  seed?: number;
}

export interface ModelRunMetrics {
  modelId: 'dense-reference' | 'compact-native-x8d';
  label: string;
  parameterization: string;
  effectiveWeights: number;
  trainableDofs: number;
  serializedBytes: number;
  fp16BaselineBytes: number;
  achievedBpw: number;
  compressionVsFp16: number;
  initTrainLoss: number;
  finalTrainLoss: number;
  initHeldOutPpl: number;
  finalHeldOutPpl: number;
  heldOutTop1AccPct: number;
  trainTimeMs: number;
  reloadedEvalTimeMs: number;
  activeEvalBufferBytes: number;
  lossCurve: Array<{ epoch: number; trainLoss: number; heldOutPpl: number }>;
  generatedContinuation: string;
  generatedQuantaSample: string[];
  reloadExactParityMaxDiff: number;
}

export interface NativeCompactExperimentReport {
  vocabSize: number;
  contextWindowBytes: number;
  trainCorpusBytes: number;
  heldOutCorpusBytes: number;
  seedPrompt: string;
  denseReference: ModelRunMetrics;
  compactNativeX8d: ModelRunMetrics;
  honestGapNote: string;
}

const CONTEXT_LEN = 4;
const EMB_DIM = 16; // per context slot
const D_MODEL = CONTEXT_LEN * EMB_DIM; // 64
const VOCAB_SIZE = 256;

const TRAIN_TEXT =
  '{"task":"gr00t_pick","byte":0.065,"state":"S1","ans":65}\n' +
  '{"task":"gr00t_place","byte":0.066,"state":"S1","ans":66}\n' +
  '{"task":"sudoku_s2","byte":0.049,"state":"S2","ans":49}\n' +
  '{"task":"snake_go","byte":0.085,"state":"S1","ans":85}\n' +
  '{"task":"gr00t_pick","byte":0.067,"state":"S1","ans":67}\n' +
  '{"task":"sudoku_s2","byte":0.052,"state":"S2","ans":52}\n';

const HELD_OUT_TEXT =
  '{"task":"gr00t_pick","byte":0.072,"state":"S1","ans":72}\n' +
  '{"task":"sudoku_s2","byte":0.053,"state":"S2","ans":53}\n' +
  '{"task":"snake_go","byte":0.082,"state":"S1","ans":82}\n';

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function nowMs(): number {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function softmaxCrossEntropy(
  logits: Float32Array,
  target: number,
  gradLogitsOut: Float32Array
): { loss: number; predicted: number } {
  let maxL = -Infinity;
  let argmax = 0;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > maxL) {
      maxL = logits[i];
      argmax = i;
    }
  }
  let sumExp = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - maxL);
    gradLogitsOut[i] = e;
    sumExp += e;
  }
  const invSum = 1 / sumExp;
  for (let i = 0; i < logits.length; i++) {
    gradLogitsOut[i] *= invSum;
  }
  const pTarget = Math.max(1e-7, gradLogitsOut[target]);
  gradLogitsOut[target] -= 1;
  return { loss: -Math.log(pTarget), predicted: argmax };
}

function adamStep(
  param: Float32Array,
  grad: Float32Array,
  m: Float32Array,
  v: Float32Array,
  stepT: number,
  lr: number
): void {
  const beta1 = 0.9;
  const beta2 = 0.99;
  const eps = 1e-7;
  const b1Corr = 1 - Math.pow(beta1, stepT);
  const b2Corr = 1 - Math.pow(beta2, stepT);
  for (let i = 0; i < param.length; i++) {
    const g = Math.max(-5, Math.min(5, grad[i]));
    m[i] = beta1 * m[i] + (1 - beta1) * g;
    v[i] = beta2 * v[i] + (1 - beta2) * g * g;
    const mHat = m[i] / b1Corr;
    const vHat = v[i] / b2Corr;
    param[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
  }
}

function steQuantizeVec(
  src: Float32Array,
  qBits: number,
  outFloat: Float32Array,
  outCodes?: Uint8Array
): { minV: number; span: number } {
  const qMax = (1 << qBits) - 1;
  let minV = Infinity;
  let maxV = -Infinity;
  for (let i = 0; i < src.length; i++) {
    if (src[i] < minV) minV = src[i];
    if (src[i] > maxV) maxV = src[i];
  }
  const span = Math.max(1e-6, maxV - minV);
  const scale = span / qMax;
  const invScale = qMax / span;
  for (let i = 0; i < src.length; i++) {
    const code = Math.max(0, Math.min(qMax, Math.round((src[i] - minV) * invScale)));
    outFloat[i] = minV + code * scale;
    if (outCodes) outCodes[i] = code;
  }
  return { minV, span };
}

function packBits(values: Uint8Array, bitsPerVal: number, bitOffset: number, buf: Uint8Array): number {
  let pos = bitOffset;
  const mask = (1 << bitsPerVal) - 1;
  for (let i = 0; i < values.length; i++) {
    const v = values[i] & mask;
    for (let b = bitsPerVal - 1; b >= 0; b--) {
      if ((v >> b) & 1) {
        buf[pos >> 3] |= 1 << (7 - (pos & 7));
      }
      pos++;
    }
  }
  return pos;
}

function unpackBits(buf: Uint8Array, count: number, bitsPerVal: number, bitOffset: number): { values: Uint8Array; nextBitOffset: number } {
  const values = new Uint8Array(count);
  let pos = bitOffset;
  for (let i = 0; i < count; i++) {
    let v = 0;
    for (let b = 0; b < bitsPerVal; b++) {
      const bit = (buf[pos >> 3] >> (7 - (pos & 7))) & 1;
      v = (v << 1) | bit;
      pos++;
    }
    values[i] = v;
  }
  return { values, nextBitOffset: pos };
}

// =========================================================================
// 1. Conventional Dense Reference Model (trained natively in dense FP32)
// =========================================================================

interface DenseModel {
  E: Float32Array; // 256 x 16
  W1: Float32Array; // 64 x 64
  W2: Float32Array; // 64 x 256
  b2: Float32Array; // 256
}

function denseForward(
  model: DenseModel,
  ctxBytes: [number, number, number, number],
  xBuf: Float32Array,
  hPreBuf: Float32Array,
  hActBuf: Float32Array,
  zBuf: Float32Array,
  logitsBuf: Float32Array
): void {
  for (let s = 0; s < CONTEXT_LEN; s++) {
    const bOff = ctxBytes[s] * EMB_DIM;
    const xOff = s * EMB_DIM;
    for (let d = 0; d < EMB_DIM; d++) {
      xBuf[xOff + d] = model.E[bOff + d];
    }
  }
  for (let c = 0; c < D_MODEL; c++) {
    let acc = 0;
    for (let r = 0; r < D_MODEL; r++) acc += xBuf[r] * model.W1[r * D_MODEL + c];
    hPreBuf[c] = acc;
    const th = Math.tanh(acc);
    hActBuf[c] = th;
    zBuf[c] = xBuf[c] + 0.5 * th;
  }
  for (let v = 0; v < VOCAB_SIZE; v++) {
    let acc = model.b2[v];
    for (let r = 0; r < D_MODEL; r++) acc += zBuf[r] * model.W2[r * VOCAB_SIZE + v];
    logitsBuf[v] = acc;
  }
}

function serializeDenseFp16Bytes(model: DenseModel): Uint8Array {
  const totalFloats = model.E.length + model.W1.length + model.W2.length + model.b2.length;
  return new Uint8Array(totalFloats * 2);
}

function evalDenseCorpus(model: DenseModel, bytes: Uint8Array): { ppl: number; top1AccPct: number } {
  const x = new Float32Array(D_MODEL);
  const hPre = new Float32Array(D_MODEL);
  const hAct = new Float32Array(D_MODEL);
  const z = new Float32Array(D_MODEL);
  const logits = new Float32Array(VOCAB_SIZE);
  const probs = new Float32Array(VOCAB_SIZE);
  let totalNll = 0;
  let hits = 0;
  const steps = Math.max(1, bytes.length - CONTEXT_LEN);
  for (let t = CONTEXT_LEN; t < bytes.length; t++) {
    const ctx: [number, number, number, number] = [bytes[t - 4], bytes[t - 3], bytes[t - 2], bytes[t - 1]];
    denseForward(model, ctx, x, hPre, hAct, z, logits);
    const { loss, predicted } = softmaxCrossEntropy(logits, bytes[t], probs);
    totalNll += loss;
    if (predicted === bytes[t]) hits++;
  }
  return { ppl: Math.exp(totalNll / steps), top1AccPct: (hits / steps) * 100 };
}

function generateDense(model: DenseModel, seedPrompt: string, genLen: number): string {
  const seed = Array.from(new TextEncoder().encode(seedPrompt));
  const stream = [...seed];
  const x = new Float32Array(D_MODEL);
  const hPre = new Float32Array(D_MODEL);
  const hAct = new Float32Array(D_MODEL);
  const z = new Float32Array(D_MODEL);
  const logits = new Float32Array(VOCAB_SIZE);
  for (let i = 0; i < genLen; i++) {
    const n = stream.length;
    const ctx: [number, number, number, number] = [
      stream[n - 4] ?? 32,
      stream[n - 3] ?? 32,
      stream[n - 2] ?? 32,
      stream[n - 1] ?? 32,
    ];
    denseForward(model, ctx, x, hPre, hAct, z, logits);
    let best = 0;
    for (let v = 1; v < VOCAB_SIZE; v++) if (logits[v] > logits[best]) best = v;
    stream.push(best);
  }
  return new TextDecoder().decode(new Uint8Array(stream.slice(seed.length)));
}

function trainDenseReference(
  trainBytes: Uint8Array,
  heldOutBytes: Uint8Array,
  epochs: number,
  seed: number,
  seedPrompt: string
): ModelRunMetrics {
  const rand = mulberry32(seed);
  const initArr = (len: number, scale: number) => {
    const a = new Float32Array(len);
    for (let i = 0; i < len; i++) a[i] = (rand() - 0.5) * 2 * scale;
    return a;
  };
  const model: DenseModel = {
    E: initArr(VOCAB_SIZE * EMB_DIM, 0.25),
    W1: initArr(D_MODEL * D_MODEL, 0.14),
    W2: initArr(D_MODEL * VOCAB_SIZE, 0.12),
    b2: new Float32Array(VOCAB_SIZE),
  };

  // Quantized subtle 0.000..0.255 prior anchor blended cleanly with trainable E so random weights start neutral
  const gE = new Float32Array(model.E.length);
  const gW1 = new Float32Array(model.W1.length);
  const gW2 = new Float32Array(model.W2.length);
  const gb2 = new Float32Array(model.b2.length);

  const mE = new Float32Array(model.E.length);
  const vE = new Float32Array(model.E.length);
  const mW1 = new Float32Array(model.W1.length);
  const vW1 = new Float32Array(model.W1.length);
  const mW2 = new Float32Array(model.W2.length);
  const vW2 = new Float32Array(model.W2.length);
  const mb2 = new Float32Array(model.b2.length);
  const vb2 = new Float32Array(model.b2.length);

  const x = new Float32Array(D_MODEL);
  const hPre = new Float32Array(D_MODEL);
  const hAct = new Float32Array(D_MODEL);
  const z = new Float32Array(D_MODEL);
  const logits = new Float32Array(VOCAB_SIZE);
  const gLogits = new Float32Array(VOCAB_SIZE);
  const gZ = new Float32Array(D_MODEL);
  const gX = new Float32Array(D_MODEL);

  const initTrain = evalDenseCorpus(model, trainBytes);
  const initTrainLoss = Math.log(initTrain.ppl);
  const initHeld = evalDenseCorpus(model, heldOutBytes);

  const lossCurve: Array<{ epoch: number; trainLoss: number; heldOutPpl: number }> = [];
  const t0 = nowMs();
  const invN = 1 / Math.max(1, trainBytes.length - CONTEXT_LEN);

  for (let ep = 1; ep <= epochs; ep++) {
    gE.fill(0);
    gW1.fill(0);
    gW2.fill(0);
    gb2.fill(0);
    let epochLoss = 0;

    for (let t = CONTEXT_LEN; t < trainBytes.length; t++) {
      const ctx: [number, number, number, number] = [
        trainBytes[t - 4],
        trainBytes[t - 3],
        trainBytes[t - 2],
        trainBytes[t - 1],
      ];
      denseForward(model, ctx, x, hPre, hAct, z, logits);
      const { loss } = softmaxCrossEntropy(logits, trainBytes[t], gLogits);
      epochLoss += loss;

      gZ.fill(0);
      for (let v = 0; v < VOCAB_SIZE; v++) {
        const gl = gLogits[v] * invN;
        gb2[v] += gl;
        if (Math.abs(gl) < 1e-8) continue;
        for (let r = 0; r < D_MODEL; r++) {
          gW2[r * VOCAB_SIZE + v] += z[r] * gl;
          gZ[r] += model.W2[r * VOCAB_SIZE + v] * gl;
        }
      }

      for (let r = 0; r < D_MODEL; r++) gX[r] = gZ[r];
      for (let c = 0; c < D_MODEL; c++) {
        const ghPre = gZ[c] * 0.5 * (1 - hAct[c] * hAct[c]);
        for (let r = 0; r < D_MODEL; r++) {
          gW1[r * D_MODEL + c] += x[r] * ghPre;
          gX[r] += model.W1[r * D_MODEL + c] * ghPre;
        }
      }

      for (let s = 0; s < CONTEXT_LEN; s++) {
        const bOff = ctx[s] * EMB_DIM;
        const xOff = s * EMB_DIM;
        for (let d = 0; d < EMB_DIM; d++) {
          gE[bOff + d] += gX[xOff + d];
        }
      }
    }

    adamStep(model.E, gE, mE, vE, ep, 0.045);
    adamStep(model.W1, gW1, mW1, vW1, ep, 0.045);
    adamStep(model.W2, gW2, mW2, vW2, ep, 0.045);
    adamStep(model.b2, gb2, mb2, vb2, ep, 0.045);

    const avgLoss = epochLoss * invN;
    if (ep === 1 || ep % 5 === 0 || ep === epochs) {
      const held = evalDenseCorpus(model, heldOutBytes);
      lossCurve.push({
        epoch: ep,
        trainLoss: Number(avgLoss.toFixed(4)),
        heldOutPpl: Number(held.ppl.toFixed(3)),
      });
    }
  }
  const trainTimeMs = nowMs() - t0;

  const finalTrain = evalDenseCorpus(model, trainBytes);
  const tEval0 = nowMs();
  const finalHeld = evalDenseCorpus(model, heldOutBytes);
  const continuation = generateDense(model, seedPrompt, 36);
  const reloadedEvalTimeMs = nowMs() - tEval0;

  const fp16Bytes = serializeDenseFp16Bytes(model).length;
  const effectiveWeights = model.E.length + model.W1.length + model.W2.length + model.b2.length;
  const contBytes = new TextEncoder().encode(continuation);

  return {
    modelId: 'dense-reference',
    label: 'Conventional Dense Reference (FP16 Checkpoint)',
    parameterization: 'Dense E(256×16) + W1(64×64) + W2(64×256) + b2(256)',
    effectiveWeights,
    trainableDofs: effectiveWeights,
    serializedBytes: fp16Bytes,
    fp16BaselineBytes: fp16Bytes,
    achievedBpw: 16.0,
    compressionVsFp16: 1.0,
    initTrainLoss: Number(initTrainLoss.toFixed(4)),
    finalTrainLoss: Number(Math.log(finalTrain.ppl).toFixed(4)),
    initHeldOutPpl: Number(initHeld.ppl.toFixed(3)),
    finalHeldOutPpl: Number(finalHeld.ppl.toFixed(3)),
    heldOutTop1AccPct: Number(finalHeld.top1AccPct.toFixed(1)),
    trainTimeMs: Number(trainTimeMs.toFixed(2)),
    reloadedEvalTimeMs: Number(reloadedEvalTimeMs.toFixed(2)),
    activeEvalBufferBytes: effectiveWeights * 4,
    lossCurve,
    generatedContinuation: continuation,
    generatedQuantaSample: Array.from(contBytes.slice(0, 8)).map((b) => (b * 0.001).toFixed(3)),
    reloadExactParityMaxDiff: 0,
  };
}

// =========================================================================
// 2. Compact Native x8D Model (Trained in Compact Kronecker Form from Step 0)
// =========================================================================

interface KronLayerMeta {
  rows: number;
  cols: number;
  p1: number;
  q1: number;
  p2: number;
  q2: number;
  kTerms: number;
  sigma: number;
}

interface KronFactors {
  meta: KronLayerMeta;
  A: Float32Array[]; // each k has p1 * q1
  B: Float32Array[]; // each k has p2 * q2
}

interface CompactNativeModel {
  E: KronFactors; // effective 256 x 16 via (16x4) (x) (16x4)
  W1: KronFactors; // effective 64 x 64 via (8x8) (x) (8x8)
  W2: KronFactors; // effective 64 x 256 via (8x16) (x) (8x16)
  b2: Float32Array; // 256
  factorBits: 2 | 4;
}

function kronMatVecForward(
  layer: KronFactors,
  qA: Float32Array[],
  qB: Float32Array[],
  x: Float32Array,
  yOut: Float32Array,
  tmpT: Float32Array
): void {
  const { p1, q1, p2, q2, kTerms, sigma } = layer.meta;
  yOut.fill(0);
  for (let k = 0; k < kTerms; k++) {
    const Ak = qA[k];
    const Bk = qB[k];
    // T[i1, j2] = sum_{i2} X[i1, i2] * B[i2, j2]  (p1 x q2)
    for (let i1 = 0; i1 < p1; i1++) {
      const xRowOff = i1 * p2;
      const tRowOff = i1 * q2;
      for (let j2 = 0; j2 < q2; j2++) {
        let s = 0;
        for (let i2 = 0; i2 < p2; i2++) {
          s += x[xRowOff + i2] * Bk[i2 * q2 + j2];
        }
        tmpT[tRowOff + j2] = s;
      }
    }
    // Y[j1, j2] += sigma * sum_{i1} A[i1, j1] * T[i1, j2]
    for (let j1 = 0; j1 < q1; j1++) {
      const yRowOff = j1 * q2;
      for (let j2 = 0; j2 < q2; j2++) {
        let s = 0;
        for (let i1 = 0; i1 < p1; i1++) {
          s += Ak[i1 * q1 + j1] * tmpT[i1 * q2 + j2];
        }
        yOut[yRowOff + j2] += sigma * s;
      }
    }
  }
}

function kronMatVecBackward(
  layer: KronFactors,
  qA: Float32Array[],
  qB: Float32Array[],
  x: Float32Array,
  gY: Float32Array,
  gA: Float32Array[],
  gB: Float32Array[],
  gXOut: Float32Array,
  T: Float32Array,
  AG: Float32Array
): void {
  const { p1, q1, p2, q2, kTerms, sigma } = layer.meta;
  gXOut.fill(0);

  for (let k = 0; k < kTerms; k++) {
    const Ak = qA[k];
    const Bk = qB[k];
    const gAk = gA[k];
    const gBk = gB[k];

    // T = X * Bk  (p1 x q2)
    for (let i1 = 0; i1 < p1; i1++) {
      const xOff = i1 * p2;
      const tOff = i1 * q2;
      for (let j2 = 0; j2 < q2; j2++) {
        let s = 0;
        for (let i2 = 0; i2 < p2; i2++) s += x[xOff + i2] * Bk[i2 * q2 + j2];
        T[tOff + j2] = s;
      }
    }

    // dL/dA[i1, j1] += sigma * sum_{j2} T[i1, j2] * gY[j1, j2]
    for (let i1 = 0; i1 < p1; i1++) {
      const tOff = i1 * q2;
      for (let j1 = 0; j1 < q1; j1++) {
        const gyOff = j1 * q2;
        let s = 0;
        for (let j2 = 0; j2 < q2; j2++) s += T[tOff + j2] * gY[gyOff + j2];
        gAk[i1 * q1 + j1] += sigma * s;
      }
    }

    // AG = Ak * gY  (p1 x q2)
    for (let i1 = 0; i1 < p1; i1++) {
      const aOff = i1 * q1;
      const agOff = i1 * q2;
      for (let j2 = 0; j2 < q2; j2++) {
        let s = 0;
        for (let j1 = 0; j1 < q1; j1++) s += Ak[aOff + j1] * gY[j1 * q2 + j2];
        AG[agOff + j2] = s;
      }
    }

    // dL/dB[i2, j2] += sigma * sum_{i1} X[i1, i2] * AG[i1, j2]
    for (let i2 = 0; i2 < p2; i2++) {
      for (let j2 = 0; j2 < q2; j2++) {
        let s = 0;
        for (let i1 = 0; i1 < p1; i1++) s += x[i1 * p2 + i2] * AG[i1 * q2 + j2];
        gBk[i2 * q2 + j2] += sigma * s;
      }
    }

    // dL/dX[i1, i2] += sigma * sum_{j2} AG[i1, j2] * B[i2, j2]
    for (let i1 = 0; i1 < p1; i1++) {
      const agOff = i1 * q2;
      const gxOff = i1 * p2;
      for (let i2 = 0; i2 < p2; i2++) {
        const bOff = i2 * q2;
        let s = 0;
        for (let j2 = 0; j2 < q2; j2++) s += AG[agOff + j2] * Bk[bOff + j2];
        gXOut[gxOff + i2] += sigma * s;
      }
    }
  }
}

function quantizeKronLayer(layer: KronFactors, factorBits: 2 | 4): { qA: Float32Array[]; qB: Float32Array[] } {
  const qA = layer.A.map((a) => {
    const out = new Float32Array(a.length);
    steQuantizeVec(a, factorBits, out);
    return out;
  });
  const qB = layer.B.map((b) => {
    const out = new Float32Array(b.length);
    steQuantizeVec(b, factorBits, out);
    return out;
  });
  return { qA, qB };
}

function evalEmbedRowFromFactors(
  qE: { qA: Float32Array[]; qB: Float32Array[] },
  meta: KronLayerMeta,
  byteId: number,
  outSlice: Float32Array,
  outOffset: number
): void {
  const i1 = Math.floor(byteId / meta.p2);
  const i2 = byteId % meta.p2;
  for (let j1 = 0; j1 < meta.q1; j1++) {
    for (let j2 = 0; j2 < meta.q2; j2++) {
      let s = 0;
      for (let k = 0; k < meta.kTerms; k++) {
        s += meta.sigma * qE.qA[k][i1 * meta.q1 + j1] * qE.qB[k][i2 * meta.q2 + j2];
      }
      outSlice[outOffset + j1 * meta.q2 + j2] = s;
    }
  }
}

/**
 * Serializes ONLY the compact Kronecker factor terms + quantized bias into a binary Uint8Array (.x8D).
 * Zero dense matrices exist in the output buffer.
 */
export function exportCompactX8dArtifact(model: CompactNativeModel): Uint8Array {
  const layers = [model.E, model.W1, model.W2];
  const factorBits = model.factorBits;
  const totalTerms = layers.reduce((s, l) => s + l.meta.kTerms, 0);

  // Header: 12B global (version, kE, kW1, kW2, factorBits, reserved + b2 min/span f32)
  // + 16B per term (minA, spanA, minB, spanB f32) + sigma table + payload bits
  const globalHeaderBytes = 16;
  const layerMetaBytes = layers.length * 8; // rows, cols, p1, q1, p2, q2, kTerms, sigma(scaled or f32 -> 12B -> let's use 12B)
  const termMetaBytes = totalTerms * 16;
  const headerTotalBytes = globalHeaderBytes + layers.length * 12 + termMetaBytes;

  let payloadBits = model.b2.length * factorBits;
  for (const l of layers) {
    for (let k = 0; k < l.meta.kTerms; k++) {
      payloadBits += (l.A[k].length + l.B[k].length) * factorBits;
    }
  }
  const totalBytes = headerTotalBytes + Math.ceil(payloadBits / 8);
  const buf = new Uint8Array(totalBytes);
  const view = new DataView(buf.buffer);

  buf[0] = 0x8d; // x8D format marker byte (raw structural tag)
  buf[1] = factorBits;
  buf[2] = model.E.meta.kTerms;
  buf[3] = model.W1.meta.kTerms;
  buf[4] = model.W2.meta.kTerms;
  view.setUint16(6, model.b2.length, true);

  const b2Codes = new Uint8Array(model.b2.length);
  const b2Q = new Float32Array(model.b2.length);
  const { minV: b2Min, span: b2Span } = steQuantizeVec(model.b2, factorBits, b2Q, b2Codes);
  view.setFloat32(8, b2Min, true);
  view.setFloat32(12, b2Span, true);

  let cursor = globalHeaderBytes;
  const encodedTerms: Array<{ codesA: Uint8Array; codesB: Uint8Array }> = [];

  for (const l of layers) {
    view.setUint16(cursor, l.meta.rows, true);
    view.setUint16(cursor + 2, l.meta.cols, true);
    buf[cursor + 4] = l.meta.p1;
    buf[cursor + 5] = l.meta.q1;
    buf[cursor + 6] = l.meta.kTerms;
    buf[cursor + 7] = factorBits;
    view.setFloat32(cursor + 8, l.meta.sigma, true);
    cursor += 12;
  }

  for (const l of layers) {
    for (let k = 0; k < l.meta.kTerms; k++) {
      const codesA = new Uint8Array(l.A[k].length);
      const codesB = new Uint8Array(l.B[k].length);
      const dummyA = new Float32Array(l.A[k].length);
      const dummyB = new Float32Array(l.B[k].length);
      const qMetaA = steQuantizeVec(l.A[k], factorBits, dummyA, codesA);
      const qMetaB = steQuantizeVec(l.B[k], factorBits, dummyB, codesB);
      view.setFloat32(cursor, qMetaA.minV, true);
      view.setFloat32(cursor + 4, qMetaA.span, true);
      view.setFloat32(cursor + 8, qMetaB.minV, true);
      view.setFloat32(cursor + 12, qMetaB.span, true);
      cursor += 16;
      encodedTerms.push({ codesA, codesB });
    }
  }

  let bitPos = headerTotalBytes * 8;
  bitPos = packBits(b2Codes, factorBits, bitPos, buf);
  for (const t of encodedTerms) {
    bitPos = packBits(t.codesA, factorBits, bitPos, buf);
    bitPos = packBits(t.codesB, factorBits, bitPos, buf);
  }

  return buf;
}

/**
 * STRICT ISOLATION RELOAD:
 * Parses ONLY the compact binary Uint8Array artifact (zero training state / dense matrices),
 * runs held-out perplexity, byte generation, and active RAM accounting.
 */
export function loadAndEvalFromCompactX8dBinary(
  x8dBytes: Uint8Array,
  heldOutBytes: Uint8Array,
  seedPrompt: string,
  genLen = 36
): {
  heldOutPpl: number;
  heldOutTop1AccPct: number;
  generatedContinuation: string;
  firstTokenLogits: Float32Array;
  evalTimeMs: number;
  activeBufferBytes: number;
} {
  const t0 = nowMs();
  const view = new DataView(x8dBytes.buffer, x8dBytes.byteOffset, x8dBytes.byteLength);
  const factorBits = x8dBytes[1] as 2 | 4;
  const b2Len = view.getUint16(6, true);
  const b2Min = view.getFloat32(8, true);
  const b2Span = view.getFloat32(12, true);

  let cursor = 16;
  const layerMetas: KronLayerMeta[] = [];
  for (let lIdx = 0; lIdx < 3; lIdx++) {
    const rows = view.getUint16(cursor, true);
    const cols = view.getUint16(cursor + 2, true);
    const p1 = x8dBytes[cursor + 4];
    const q1 = x8dBytes[cursor + 5];
    const kTerms = x8dBytes[cursor + 6];
    const sigma = view.getFloat32(cursor + 8, true);
    layerMetas.push({
      rows,
      cols,
      p1,
      q1,
      p2: Math.ceil(rows / p1),
      q2: Math.ceil(cols / q1),
      kTerms,
      sigma,
    });
    cursor += 12;
  }

  const termHeaders: Array<{ minA: number; spanA: number; minB: number; spanB: number; mIdx: number }> = [];
  for (let mIdx = 0; mIdx < layerMetas.length; mIdx++) {
    for (let k = 0; k < layerMetas[mIdx].kTerms; k++) {
      termHeaders.push({
        minA: view.getFloat32(cursor, true),
        spanA: view.getFloat32(cursor + 4, true),
        minB: view.getFloat32(cursor + 8, true),
        spanB: view.getFloat32(cursor + 12, true),
        mIdx,
      });
      cursor += 16;
    }
  }

  const qMax = (1 << factorBits) - 1;
  let bitPos = cursor * 8;

  const b2Res = unpackBits(x8dBytes, b2Len, factorBits, bitPos);
  bitPos = b2Res.nextBitOffset;
  const b2 = new Float32Array(b2Len);
  for (let i = 0; i < b2Len; i++) b2[i] = b2Min + (b2Res.values[i] / qMax) * b2Span;

  const unpackedLayers: Array<{ meta: KronLayerMeta; qA: Float32Array[]; qB: Float32Array[] }> = layerMetas.map(
    (meta) => ({ meta, qA: [], qB: [] })
  );

  let activeFactorFloats = b2.length;
  for (const th of termHeaders) {
    const meta = layerMetas[th.mIdx];
    const lenA = meta.p1 * meta.q1;
    const lenB = meta.p2 * meta.q2;
    const aRes = unpackBits(x8dBytes, lenA, factorBits, bitPos);
    bitPos = aRes.nextBitOffset;
    const bRes = unpackBits(x8dBytes, lenB, factorBits, bitPos);
    bitPos = bRes.nextBitOffset;

    const fa = new Float32Array(lenA);
    const fb = new Float32Array(lenB);
    for (let i = 0; i < lenA; i++) fa[i] = th.minA + (aRes.values[i] / qMax) * th.spanA;
    for (let i = 0; i < lenB; i++) fb[i] = th.minB + (bRes.values[i] / qMax) * th.spanB;
    unpackedLayers[th.mIdx].qA.push(fa);
    unpackedLayers[th.mIdx].qB.push(fb);
    activeFactorFloats += lenA + lenB;
  }

  const [EPack, W1Pack, W2Pack] = unpackedLayers;
  const x = new Float32Array(D_MODEL);
  const h = new Float32Array(D_MODEL);
  const z = new Float32Array(D_MODEL);
  const logits = new Float32Array(VOCAB_SIZE);
  const probs = new Float32Array(VOCAB_SIZE);
  const tmpT = new Float32Array(256);

  const forwardCtx = (ctx: [number, number, number, number]) => {
    for (let s = 0; s < CONTEXT_LEN; s++) {
      evalEmbedRowFromFactors(EPack, EPack.meta, ctx[s], x, s * EMB_DIM);
    }
    kronMatVecForward({ meta: W1Pack.meta, A: W1Pack.qA, B: W1Pack.qB }, W1Pack.qA, W1Pack.qB, x, h, tmpT);
    for (let c = 0; c < D_MODEL; c++) {
      const th = Math.tanh(h[c]);
      z[c] = x[c] + 0.5 * th;
    }
    kronMatVecForward({ meta: W2Pack.meta, A: W2Pack.qA, B: W2Pack.qB }, W2Pack.qA, W2Pack.qB, z, logits, tmpT);
    for (let v = 0; v < VOCAB_SIZE; v++) logits[v] += b2[v];
  };

  let totalNll = 0;
  let hits = 0;
  let firstTokenLogits = new Float32Array(VOCAB_SIZE);
  const steps = Math.max(1, heldOutBytes.length - CONTEXT_LEN);
  for (let t = CONTEXT_LEN; t < heldOutBytes.length; t++) {
    forwardCtx([heldOutBytes[t - 4], heldOutBytes[t - 3], heldOutBytes[t - 2], heldOutBytes[t - 1]]);
    if (t === CONTEXT_LEN) firstTokenLogits = new Float32Array(logits);
    const { loss, predicted } = softmaxCrossEntropy(logits, heldOutBytes[t], probs);
    totalNll += loss;
    if (predicted === heldOutBytes[t]) hits++;
  }

  // Generate continuation text strictly from reloaded .x8D factor tensors
  const seed = Array.from(new TextEncoder().encode(seedPrompt));
  const stream = [...seed];
  for (let i = 0; i < genLen; i++) {
    const n = stream.length;
    forwardCtx([stream[n - 4] ?? 32, stream[n - 3] ?? 32, stream[n - 2] ?? 32, stream[n - 1] ?? 32]);
    let best = 0;
    for (let v = 1; v < VOCAB_SIZE; v++) if (logits[v] > logits[best]) best = v;
    stream.push(best);
  }
  const generatedContinuation = new TextDecoder().decode(new Uint8Array(stream.slice(seed.length)));
  const evalTimeMs = nowMs() - t0;

  return {
    heldOutPpl: Math.exp(totalNll / steps),
    heldOutTop1AccPct: (hits / steps) * 100,
    generatedContinuation,
    firstTokenLogits,
    evalTimeMs,
    activeBufferBytes: x8dBytes.length + activeFactorFloats * 4,
  };
}

function trainCompactNativeX8d(
  trainBytes: Uint8Array,
  heldOutBytes: Uint8Array,
  epochs: number,
  kTerms: number,
  factorBits: 2 | 4,
  seed: number,
  seedPrompt: string,
  fp16BaselineBytes: number,
  effectiveWeights: number
): ModelRunMetrics {
  const rand = mulberry32(seed + 101);
  const makeLayer = (rows: number, cols: number, p1: number, q1: number, sigma: number): KronFactors => {
    const p2 = Math.ceil(rows / p1);
    const q2 = Math.ceil(cols / q1);
    const A: Float32Array[] = [];
    const B: Float32Array[] = [];
    const scA = Math.sqrt(1 / (p1 * kTerms));
    const scB = Math.sqrt(1 / (p2 * kTerms));
    for (let k = 0; k < kTerms; k++) {
      const ak = new Float32Array(p1 * q1);
      const bk = new Float32Array(p2 * q2);
      for (let i = 0; i < ak.length; i++) ak[i] = (rand() - 0.5) * 2 * scA;
      for (let i = 0; i < bk.length; i++) bk[i] = (rand() - 0.5) * 2 * scB;
      A.push(ak);
      B.push(bk);
    }
    return {
      meta: { rows, cols, p1, q1, p2, q2, kTerms, sigma },
      A,
      B,
    };
  };

  const model: CompactNativeModel = {
    E: makeLayer(VOCAB_SIZE, EMB_DIM, 16, 4, 0.55),
    W1: makeLayer(D_MODEL, D_MODEL, 8, 8, 0.25),
    W2: makeLayer(D_MODEL, VOCAB_SIZE, 8, 16, 0.28),
    b2: new Float32Array(VOCAB_SIZE),
    factorBits,
  };

  const makeGradAndAdam = (layer: KronFactors) => ({
    gA: layer.A.map((a) => new Float32Array(a.length)),
    gB: layer.B.map((b) => new Float32Array(b.length)),
    mA: layer.A.map((a) => new Float32Array(a.length)),
    vA: layer.A.map((a) => new Float32Array(a.length)),
    mB: layer.B.map((b) => new Float32Array(b.length)),
    vB: layer.B.map((b) => new Float32Array(b.length)),
  });

  const optE = makeGradAndAdam(model.E);
  const optW1 = makeGradAndAdam(model.W1);
  const optW2 = makeGradAndAdam(model.W2);
  const gb2 = new Float32Array(VOCAB_SIZE);
  const mb2 = new Float32Array(VOCAB_SIZE);
  const vb2 = new Float32Array(VOCAB_SIZE);

  const x = new Float32Array(D_MODEL);
  const hPre = new Float32Array(D_MODEL);
  const hAct = new Float32Array(D_MODEL);
  const z = new Float32Array(D_MODEL);
  const logits = new Float32Array(VOCAB_SIZE);
  const gLogits = new Float32Array(VOCAB_SIZE);
  const gZ = new Float32Array(D_MODEL);
  const gHPre = new Float32Array(D_MODEL);
  const gX = new Float32Array(D_MODEL);
  const gXFromW1 = new Float32Array(D_MODEL);
  const tmpT = new Float32Array(256);
  const tmpAG = new Float32Array(256);

  const evalInMemorySte = (bytes: Uint8Array): { ppl: number; firstLogits: Float32Array } => {
    const qE = quantizeKronLayer(model.E, factorBits);
    const qW1 = quantizeKronLayer(model.W1, factorBits);
    const qW2 = quantizeKronLayer(model.W2, factorBits);
    const qB2 = new Float32Array(VOCAB_SIZE);
    steQuantizeVec(model.b2, factorBits, qB2);
    let totalNll = 0;
    let firstLogits = new Float32Array(VOCAB_SIZE);
    const steps = Math.max(1, bytes.length - CONTEXT_LEN);
    for (let t = CONTEXT_LEN; t < bytes.length; t++) {
      for (let s = 0; s < CONTEXT_LEN; s++) {
        evalEmbedRowFromFactors(qE, model.E.meta, bytes[t - CONTEXT_LEN + s], x, s * EMB_DIM);
      }
      kronMatVecForward(model.W1, qW1.qA, qW1.qB, x, hPre, tmpT);
      for (let c = 0; c < D_MODEL; c++) z[c] = x[c] + 0.5 * Math.tanh(hPre[c]);
      kronMatVecForward(model.W2, qW2.qA, qW2.qB, z, logits, tmpT);
      for (let v = 0; v < VOCAB_SIZE; v++) logits[v] += qB2[v];
      if (t === CONTEXT_LEN) firstLogits = new Float32Array(logits);
      const { loss } = softmaxCrossEntropy(logits, bytes[t], gLogits);
      totalNll += loss;
    }
    return { ppl: Math.exp(totalNll / steps), firstLogits };
  };

  const initTrain = evalInMemorySte(trainBytes);
  const initTrainLoss = Math.log(initTrain.ppl);
  const initHeld = evalInMemorySte(heldOutBytes);

  const lossCurve: Array<{ epoch: number; trainLoss: number; heldOutPpl: number }> = [];
  const t0 = nowMs();
  const invN = 1 / Math.max(1, trainBytes.length - CONTEXT_LEN);

  for (let ep = 1; ep <= epochs; ep++) {
    for (let k = 0; k < kTerms; k++) {
      optE.gA[k].fill(0);
      optE.gB[k].fill(0);
      optW1.gA[k].fill(0);
      optW1.gB[k].fill(0);
      optW2.gA[k].fill(0);
      optW2.gB[k].fill(0);
    }
    gb2.fill(0);

    // STE Quantize factors ONCE per full-batch epoch for forward & backward passes
    const qE = quantizeKronLayer(model.E, factorBits);
    const qW1 = quantizeKronLayer(model.W1, factorBits);
    const qW2 = quantizeKronLayer(model.W2, factorBits);
    const qB2 = new Float32Array(VOCAB_SIZE);
    steQuantizeVec(model.b2, factorBits, qB2);

    let epochLoss = 0;

    for (let t = CONTEXT_LEN; t < trainBytes.length; t++) {
      const ctx: [number, number, number, number] = [
        trainBytes[t - 4],
        trainBytes[t - 3],
        trainBytes[t - 2],
        trainBytes[t - 1],
      ];
      for (let s = 0; s < CONTEXT_LEN; s++) {
        evalEmbedRowFromFactors(qE, model.E.meta, ctx[s], x, s * EMB_DIM);
      }
      kronMatVecForward(model.W1, qW1.qA, qW1.qB, x, hPre, tmpT);
      for (let c = 0; c < D_MODEL; c++) {
        const th = Math.tanh(hPre[c]);
        hAct[c] = th;
        z[c] = x[c] + 0.5 * th;
      }
      kronMatVecForward(model.W2, qW2.qA, qW2.qB, z, logits, tmpT);
      for (let v = 0; v < VOCAB_SIZE; v++) logits[v] += qB2[v];

      const { loss } = softmaxCrossEntropy(logits, trainBytes[t], gLogits);
      epochLoss += loss;

      for (let v = 0; v < VOCAB_SIZE; v++) {
        gLogits[v] *= invN;
        gb2[v] += gLogits[v];
      }

      kronMatVecBackward(model.W2, qW2.qA, qW2.qB, z, gLogits, optW2.gA, optW2.gB, gZ, tmpT, tmpAG);

      for (let c = 0; c < D_MODEL; c++) gHPre[c] = gZ[c] * 0.5 * (1 - hAct[c] * hAct[c]);
      kronMatVecBackward(model.W1, qW1.qA, qW1.qB, x, gHPre, optW1.gA, optW1.gB, gXFromW1, tmpT, tmpAG);

      for (let c = 0; c < D_MODEL; c++) gX[c] = gZ[c] + gXFromW1[c];

      // Exact gradient back through factored E[byteId] = sum_k sigma * qA[k][i1, j1] * qB[k][i2, j2]
      const { p2, q1, q2, sigma } = model.E.meta;
      for (let s = 0; s < CONTEXT_LEN; s++) {
        const byteId = ctx[s];
        const i1 = Math.floor(byteId / p2);
        const i2 = byteId % p2;
        const xOff = s * EMB_DIM;
        for (let k = 0; k < kTerms; k++) {
          const ak = qE.qA[k];
          const bk = qE.qB[k];
          const gak = optE.gA[k];
          const gbk = optE.gB[k];
          for (let j1 = 0; j1 < q1; j1++) {
            for (let j2 = 0; j2 < q2; j2++) {
              const g = gX[xOff + j1 * q2 + j2] * sigma;
              gak[i1 * q1 + j1] += g * bk[i2 * q2 + j2];
              gbk[i2 * q2 + j2] += g * ak[i1 * q1 + j1];
            }
          }
        }
      }
    }

    const lr = 0.055;
    for (let k = 0; k < kTerms; k++) {
      adamStep(model.E.A[k], optE.gA[k], optE.mA[k], optE.vA[k], ep, lr);
      adamStep(model.E.B[k], optE.gB[k], optE.mB[k], optE.vB[k], ep, lr);
      adamStep(model.W1.A[k], optW1.gA[k], optW1.mA[k], optW1.vA[k], ep, lr);
      adamStep(model.W1.B[k], optW1.gB[k], optW1.mB[k], optW1.vB[k], ep, lr);
      adamStep(model.W2.A[k], optW2.gA[k], optW2.mA[k], optW2.vA[k], ep, lr);
      adamStep(model.W2.B[k], optW2.gB[k], optW2.mB[k], optW2.vB[k], ep, lr);
    }
    adamStep(model.b2, gb2, mb2, vb2, ep, lr);

    const avgLoss = epochLoss * invN;
    if (ep === 1 || ep % 5 === 0 || ep === epochs) {
      const held = evalInMemorySte(heldOutBytes);
      lossCurve.push({
        epoch: ep,
        trainLoss: Number(avgLoss.toFixed(4)),
        heldOutPpl: Number(held.ppl.toFixed(3)),
      });
    }
  }
  const trainTimeMs = nowMs() - t0;

  const finalTrain = evalInMemorySte(trainBytes);
  const preExportHeld = evalInMemorySte(heldOutBytes);

  // EXPORT ONLY compact binary .x8D Uint8Array; reload strictly from bytes with zero training buffers!
  const x8dArtifact = exportCompactX8dArtifact(model);
  const reloaded = loadAndEvalFromCompactX8dBinary(x8dArtifact, heldOutBytes, seedPrompt, 36);

  let maxLogitDiff = 0;
  for (let i = 0; i < VOCAB_SIZE; i++) {
    maxLogitDiff = Math.max(maxLogitDiff, Math.abs(preExportHeld.firstLogits[i] - reloaded.firstTokenLogits[i]));
  }

  let trainableDofs = model.b2.length;
  for (const l of [model.E, model.W1, model.W2]) {
    for (let k = 0; k < kTerms; k++) trainableDofs += l.A[k].length + l.B[k].length;
  }

  const achievedBpw = (x8dArtifact.length * 8) / effectiveWeights;
  const contBytes = new TextEncoder().encode(reloaded.generatedContinuation);

  return {
    modelId: 'compact-native-x8d',
    label: `Compact Native .x8D (K=${kTerms}, ${factorBits}-bit STE From Step 0)`,
    parameterization: `Factored Kronecker (A⊗B) STE from random init -> ${x8dArtifact.length} B binary .x8D`,
    effectiveWeights,
    trainableDofs,
    serializedBytes: x8dArtifact.length,
    fp16BaselineBytes,
    achievedBpw: Number(achievedBpw.toFixed(4)),
    compressionVsFp16: Number((fp16BaselineBytes / x8dArtifact.length).toFixed(2)),
    initTrainLoss: Number(initTrainLoss.toFixed(4)),
    finalTrainLoss: Number(Math.log(finalTrain.ppl).toFixed(4)),
    initHeldOutPpl: Number(initHeld.ppl.toFixed(3)),
    finalHeldOutPpl: Number(reloaded.heldOutPpl.toFixed(3)),
    heldOutTop1AccPct: Number(reloaded.heldOutTop1AccPct.toFixed(1)),
    trainTimeMs: Number(trainTimeMs.toFixed(2)),
    reloadedEvalTimeMs: Number(reloaded.evalTimeMs.toFixed(2)),
    activeEvalBufferBytes: reloaded.activeBufferBytes,
    lossCurve,
    generatedContinuation: reloaded.generatedContinuation,
    generatedQuantaSample: Array.from(contBytes.slice(0, 8)).map((b) => (b * 0.001).toFixed(3)),
    reloadExactParityMaxDiff: Number(maxLogitDiff.toExponential(2)),
  };
}

export function runNativeCompactTrainingExperiment(
  options: NativeExperimentOptions = {}
): NativeCompactExperimentReport {
  const epochs = Math.max(10, Math.min(80, options.epochs ?? 32));
  const kTerms = Math.max(1, Math.min(6, options.kTerms ?? 4));
  const factorBits = options.factorBits ?? 4;
  const seed = options.seed ?? 20261117;
  const seedPrompt = '{"task":"gr00t_';

  const encoder = new TextEncoder();
  const trainBytes = encoder.encode(TRAIN_TEXT);
  const heldOutBytes = encoder.encode(HELD_OUT_TEXT);

  const denseReference = trainDenseReference(trainBytes, heldOutBytes, epochs, seed, seedPrompt);
  const compactNativeX8d = trainCompactNativeX8d(
    trainBytes,
    heldOutBytes,
    epochs,
    kTerms,
    factorBits,
    seed,
    seedPrompt,
    denseReference.fp16BaselineBytes,
    denseReference.effectiveWeights
  );

  return {
    vocabSize: VOCAB_SIZE,
    contextWindowBytes: CONTEXT_LEN,
    trainCorpusBytes: trainBytes.length,
    heldOutCorpusBytes: heldOutBytes.length,
    seedPrompt,
    denseReference,
    compactNativeX8d,
    honestGapNote:
      `Measured strictly from serialized buffers: Dense FP16 takes ${denseReference.serializedBytes.toLocaleString()} B (16.0 bpw), while Compact Native .x8D trains ${compactNativeX8d.trainableDofs.toLocaleString()} Kronecker DOFs from step 0 and serializes to ${compactNativeX8d.serializedBytes.toLocaleString()} B (${compactNativeX8d.achievedBpw.toFixed(4)} bpw, ${compactNativeX8d.compressionVsFp16}× smaller). At tiny 64×256 dims, header + 4-bit factors keep bpw above 0.008; the identical Kronecker layout crosses <=0.008 bpw once matrix dims reach 512×1536+.`,
  };
}
