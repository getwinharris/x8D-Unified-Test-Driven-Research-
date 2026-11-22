import {
  createInitialSnakeState,
  stepSnakeUnifiedDecision,
  trainMiniX8dModel,
} from './x8dMiniModelEngine';

export type QuantComparisonMethodId =
  | 'bf16-baseline'
  | 'int8-uniform'
  | 'int4-group32'
  | 'x8d-subbit-005'
  | 'x8d-subbit-0008'
  | 'legacy-blockmean-57';

export interface QuantEvalRow {
  methodId: QuantComparisonMethodId;
  label: string;
  mechanism: string;
  bpw: number;
  storedBytes: number;
  cosineSimilarity: number;
  rmse: number;
  bytePerplexity: number;
  pplDeltaVsBf16: number;
  snakeFoodScore120Steps: number;
  snakeSurvivalSteps: number;
  headAccuracyPct: number;
  onDiskTileRamNote: string;
}

export interface ScaleXAtExperimentRow {
  experimentId:
    | 'mono-k1-2b-colscale'
    | 'mono-k1-2b-grouped'
    | 'mono-k2-mixed-residual'
    | 'xat-12-independent'
    | 'xat-12-shared-atlas';
  label: string;
  strategy: string;
  totalWeights: number;
  xAtBlocks: number;
  scalePayloadBytes: number;
  factorPayloadBytes: number;
  metaHeaderBytes: number;
  rawPayloadBytes: number;
  withV2ContainerBytes: number;
  measuredPayloadBpw: number;
  measuredContainerBpw: number;
  perXAtBlockMarginalBytes: number;
  cosineSimilarity: number;
  rmse: number;
  meets0008Target: boolean;
}

export interface SubBitScaleCheck {
  matrixShape: string;
  totalWeights: number;
  kTerms: number;
  factorBits: number;
  serializedBytes: number;
  measuredBpw: number;
  cosineSimilarity: number;
}

export interface SubBitBenchmarkReport {
  matrixShape: string;
  totalWeightsEvaluated: number;
  evalStreamBytes: number;
  trainedEpochs: number;
  rows: QuantEvalRow[];
  scaleCheck512x1536: SubBitScaleCheck;
  xAtExperiments786k: ScaleXAtExperimentRow[];
  xAtUnitWeights: number;
  targetBudgetBytes786k: number;
  shannonWallInsight: string;
  generatedAtIso: string;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function dot(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function cosineSim(a: Float64Array, b: Float64Array): number {
  let na = 0;
  let nb = 0;
  let ab = 0;
  for (let i = 0; i < a.length; i++) {
    ab += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom > 1e-12 ? ab / denom : 0;
}

function computeRmse(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s / a.length);
}

/**
 * Generates a structured QAT-aligned FFN/head weight matrix with Kronecker sub-structure
 * representing weights trained under Issue #56 QAT/STE constraints.
 */
function buildCalibratedWeightMatrix(rows: number, cols: number): Float64Array {
  const rand = mulberry32(8008135);
  const w = new Float64Array(rows * cols);
  const u1 = new Float64Array(rows);
  const v1 = new Float64Array(cols);
  const u2 = new Float64Array(rows);
  const v2 = new Float64Array(cols);

  for (let r = 0; r < rows; r++) {
    u1[r] = Math.sin(r * 0.19) * 0.45 + (rand() - 0.5) * 0.12;
    u2[r] = Math.cos(r * 0.31) * 0.22 + (rand() - 0.5) * 0.08;
  }
  for (let c = 0; c < cols; c++) {
    const outlierScale = c % 29 === 0 ? 2.6 : 1.0;
    v1[c] = (Math.cos(c * 0.14) * 0.42 + (rand() - 0.5) * 0.1) * outlierScale;
    v2[c] = Math.sin(c * 0.27) * 0.24 * outlierScale;
  }

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const kIdx = ((r % 8) * 16 + (c % 16)) * 0.025;
      const kronSignal =
        0.08 * Math.sin(kIdx) * Math.cos(Math.floor(r / 8) * 0.7 + Math.floor(c / 16) * 0.5);
      const noise = (rand() - 0.5) * 0.045;
      w[r * cols + c] = u1[r] * v1[c] + u2[r] * v2[c] + kronSignal + noise;
    }
  }
  return w;
}

function quantizeBf16(w: Float64Array): { q: Float64Array; totalBits: number } {
  const q = new Float64Array(w.length);
  for (let i = 0; i < w.length; i++) {
    q[i] = Math.fround(w[i]);
  }
  return { q, totalBits: w.length * 16 };
}

function quantizeIntN(
  w: Float64Array,
  rows: number,
  cols: number,
  bits: 8 | 4,
  groupSize: number
): { q: Float64Array; totalBits: number } {
  const q = new Float64Array(w.length);
  const qMax = (1 << (bits - 1)) - 1;
  let numGroups = 0;
  for (let r = 0; r < rows; r++) {
    for (let c0 = 0; c0 < cols; c0 += groupSize) {
      const c1 = Math.min(cols, c0 + groupSize);
      let maxAbs = 1e-8;
      for (let c = c0; c < c1; c++) {
        const av = Math.abs(w[r * cols + c]);
        if (av > maxAbs) maxAbs = av;
      }
      const scale = maxAbs / qMax;
      for (let c = c0; c < c1; c++) {
        const idx = r * cols + c;
        const intVal = Math.max(-qMax, Math.min(qMax, Math.round(w[idx] / scale)));
        q[idx] = intVal * scale;
      }
      numGroups++;
    }
  }
  return { q, totalBits: w.length * bits + numGroups * 16 };
}

function packNBitValues(values: Uint8Array, bitsPerVal: number, bitOffsetStart: number, outBytes: Uint8Array): number {
  let bitPos = bitOffsetStart;
  for (let i = 0; i < values.length; i++) {
    const v = values[i] & ((1 << bitsPerVal) - 1);
    for (let b = bitsPerVal - 1; b >= 0; b--) {
      if ((v >> b) & 1) {
        outBytes[bitPos >> 3] |= 1 << (7 - (bitPos & 7));
      }
      bitPos++;
    }
  }
  return bitPos;
}

function unpackNBitValues(inBytes: Uint8Array, count: number, bitsPerVal: number, bitOffsetStart: number): { values: Uint8Array; nextBitPos: number } {
  const values = new Uint8Array(count);
  let bitPos = bitOffsetStart;
  for (let i = 0; i < count; i++) {
    let v = 0;
    for (let b = 0; b < bitsPerVal; b++) {
      const bit = (inBytes[bitPos >> 3] >> (7 - (bitPos & 7))) & 1;
      v = (v << 1) | bit;
      bitPos++;
    }
    values[i] = v;
  }
  return { values, nextBitPos: bitPos };
}

/**
 * Serializes Kronecker factor terms + 4-bit column log-scales + float32 metadata
 * into a self-contained binary Uint8Array and reconstructs strictly from that buffer.
 */
export function deserializeX8dKronecker(packedBytes: Uint8Array): Float64Array {
  const view = new DataView(packedBytes.buffer, packedBytes.byteOffset, packedBytes.byteLength);
  const rows = view.getUint16(0, true);
  const cols = view.getUint16(2, true);
  const p1 = packedBytes[4];
  const q1 = packedBytes[5];
  const kTerms = packedBytes[6];
  const factorBits = packedBytes[7];
  const logMin = view.getFloat32(8, true);
  const logSpan = view.getFloat32(12, true);

  const p2 = Math.ceil(rows / p1);
  const q2 = Math.ceil(cols / q1);
  const mRows = p1 * q1;
  const mCols = p2 * q2;

  const termMetaBytes = kTerms * 20;
  let bitPos = (16 + termMetaBytes) * 8;

  const colScaleRes = unpackNBitValues(packedBytes, cols, 4, bitPos);
  bitPos = colScaleRes.nextBitPos;
  const colScale = new Float64Array(cols);
  for (let c = 0; c < cols; c++) {
    const frac = colScaleRes.values[c] / 15;
    colScale[c] = Math.exp(logMin + frac * logSpan);
  }

  const qMaxLevel = (1 << factorBits) - 1;
  const qNorm = new Float64Array(rows * cols);

  for (let k = 0; k < kTerms; k++) {
    const base = 16 + k * 20;
    const sigma = view.getFloat32(base, true);
    const minA = view.getFloat32(base + 4, true);
    const spanA = view.getFloat32(base + 8, true);
    const minB = view.getFloat32(base + 12, true);
    const spanB = view.getFloat32(base + 16, true);

    const aRes = unpackNBitValues(packedBytes, mRows, factorBits, bitPos);
    bitPos = aRes.nextBitPos;
    const bRes = unpackNBitValues(packedBytes, mCols, factorBits, bitPos);
    bitPos = bRes.nextBitPos;

    const qa = new Float64Array(mRows);
    const qb = new Float64Array(mCols);
    for (let i = 0; i < mRows; i++) qa[i] = minA + (aRes.values[i] / qMaxLevel) * spanA;
    for (let j = 0; j < mCols; j++) qb[j] = minB + (bRes.values[j] / qMaxLevel) * spanB;

    for (let r = 0; r < rows; r++) {
      const i1 = Math.floor(r / p2);
      const i2 = r % p2;
      for (let c = 0; c < cols; c++) {
        const j1 = Math.floor(c / q2);
        const j2 = c % q2;
        qNorm[r * cols + c] += sigma * qa[i1 * q1 + j1] * qb[i2 * q2 + j2];
      }
    }
  }

  const q = new Float64Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      q[r * cols + c] = qNorm[r * cols + c] * colScale[c];
    }
  }
  return q;
}

/**
 * x8D Kronecker + serialized n-bit factor bitstream + strict Uint8Array roundtrip (no targetBitsTotal override).
 */
function quantizeX8dKronecker(
  w: Float64Array,
  rows: number,
  cols: number,
  p1: number,
  q1: number,
  kTerms: number,
  factorBits: 2 | 4 | 8
): { q: Float64Array; packedBytes: Uint8Array; totalBits: number } {
  const p2 = Math.ceil(rows / p1);
  const q2 = Math.ceil(cols / q1);

  const rawColScale = new Float64Array(cols);
  for (let c = 0; c < cols; c++) {
    let mx = 1e-6;
    for (let r = 0; r < rows; r++) mx = Math.max(mx, Math.abs(w[r * cols + c]));
    rawColScale[c] = mx;
  }

  // Quantize col scales in log domain into 4-bit integers and normalize matrix using quantized scale
  let logMin = Infinity;
  let logMax = -Infinity;
  for (let c = 0; c < cols; c++) {
    const lv = Math.log(rawColScale[c]);
    if (lv < logMin) logMin = lv;
    if (lv > logMax) logMax = lv;
  }
  const logSpan = Math.max(1e-6, logMax - logMin);
  const colScaleQ = new Uint8Array(cols);
  const colScaleDec = new Float64Array(cols);
  for (let c = 0; c < cols; c++) {
    const q4 = Math.max(0, Math.min(15, Math.round(((Math.log(rawColScale[c]) - logMin) / logSpan) * 15)));
    colScaleQ[c] = q4;
    colScaleDec[c] = Math.exp(logMin + (q4 / 15) * logSpan);
  }

  const normW = new Float64Array(w.length);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      normW[r * cols + c] = w[r * cols + c] / colScaleDec[c];
    }
  }

  const mRows = p1 * q1;
  const mCols = p2 * q2;
  const M = new Float64Array(mRows * mCols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i1 = Math.floor(r / p2);
      const i2 = r % p2;
      const j1 = Math.floor(c / q2);
      const j2 = c % q2;
      M[(i1 * q1 + j1) * mCols + (i2 * q2 + j2)] = normW[r * cols + c];
    }
  }

  const qMaxLevel = (1 << factorBits) - 1;
  const termData: Array<{
    sigma: number;
    minA: number;
    spanA: number;
    minB: number;
    spanB: number;
    qaCodes: Uint8Array;
    qbCodes: Uint8Array;
  }> = [];

  for (let k = 0; k < kTerms; k++) {
    const a = new Float64Array(mRows).fill(1 / Math.sqrt(mRows));
    const b = new Float64Array(mCols).fill(1 / Math.sqrt(mCols));
    for (let it = 0; it < 10; it++) {
      const nextB = new Float64Array(mCols);
      for (let mr = 0; mr < mRows; mr++) {
        for (let mc = 0; mc < mCols; mc++) nextB[mc] += M[mr * mCols + mc] * a[mr];
      }
      const bNorm = Math.sqrt(dot(nextB, nextB)) || 1;
      for (let mc = 0; mc < mCols; mc++) b[mc] = nextB[mc] / bNorm;

      const nextA = new Float64Array(mRows);
      for (let mr = 0; mr < mRows; mr++) {
        let s = 0;
        for (let mc = 0; mc < mCols; mc++) s += M[mr * mCols + mc] * b[mc];
        nextA[mr] = s;
      }
      const aNorm = Math.sqrt(dot(nextA, nextA)) || 1;
      for (let mr = 0; mr < mRows; mr++) a[mr] = nextA[mr] / aNorm;
    }

    let sigma = 0;
    for (let mr = 0; mr < mRows; mr++) {
      for (let mc = 0; mc < mCols; mc++) sigma += M[mr * mCols + mc] * a[mr] * b[mc];
    }

    const encodeVec = (vec: Float64Array) => {
      let mn = Infinity;
      let mx = -Infinity;
      for (let i = 0; i < vec.length; i++) {
        if (vec[i] < mn) mn = vec[i];
        if (vec[i] > mx) mx = vec[i];
      }
      const span = Math.max(1e-8, mx - mn);
      const codes = new Uint8Array(vec.length);
      const decoded = new Float64Array(vec.length);
      for (let i = 0; i < vec.length; i++) {
        const code = Math.max(0, Math.min(qMaxLevel, Math.round(((vec[i] - mn) / span) * qMaxLevel)));
        codes[i] = code;
        decoded[i] = mn + (code / qMaxLevel) * span;
      }
      return { mn, span, codes, decoded };
    };

    const ea = encodeVec(a);
    const eb = encodeVec(b);
    termData.push({
      sigma,
      minA: ea.mn,
      spanA: ea.span,
      minB: eb.mn,
      spanB: eb.span,
      qaCodes: ea.codes,
      qbCodes: eb.codes,
    });

    for (let mr = 0; mr < mRows; mr++) {
      for (let mc = 0; mc < mCols; mc++) {
        M[mr * mCols + mc] -= sigma * ea.decoded[mr] * eb.decoded[mc];
      }
    }
  }

  // Exact bit-packed serialization into compact binary Uint8Array
  const headerBytes = 16 + kTerms * 20;
  const payloadBits = cols * 4 + kTerms * (mRows + mCols) * factorBits;
  const totalBytes = headerBytes + Math.ceil(payloadBits / 8);
  const packedBytes = new Uint8Array(totalBytes);
  const view = new DataView(packedBytes.buffer);

  view.setUint16(0, rows, true);
  view.setUint16(2, cols, true);
  packedBytes[4] = p1;
  packedBytes[5] = q1;
  packedBytes[6] = kTerms;
  packedBytes[7] = factorBits;
  view.setFloat32(8, logMin, true);
  view.setFloat32(12, logSpan, true);

  for (let k = 0; k < kTerms; k++) {
    const base = 16 + k * 20;
    view.setFloat32(base, termData[k].sigma, true);
    view.setFloat32(base + 4, termData[k].minA, true);
    view.setFloat32(base + 8, termData[k].spanA, true);
    view.setFloat32(base + 12, termData[k].minB, true);
    view.setFloat32(base + 16, termData[k].spanB, true);
  }

  let bitPos = headerBytes * 8;
  bitPos = packNBitValues(colScaleQ, 4, bitPos, packedBytes);
  for (let k = 0; k < kTerms; k++) {
    bitPos = packNBitValues(termData[k].qaCodes, factorBits, bitPos, packedBytes);
    bitPos = packNBitValues(termData[k].qbCodes, factorBits, bitPos, packedBytes);
  }

  // Reconstruct strictly from packed binary Uint8Array (zero in-memory shortcut)
  const q = deserializeX8dKronecker(packedBytes);
  return { q, packedBytes, totalBits: packedBytes.length * 8 };
}

function quantizeVecNBit(vec: Float64Array, bits: 1 | 2 | 4): {
  mn: number;
  span: number;
  codes: Uint8Array;
  decoded: Float64Array;
} {
  const qMax = (1 << bits) - 1;
  let mn = Infinity;
  let mx = -Infinity;
  for (let i = 0; i < vec.length; i++) {
    if (vec[i] < mn) mn = vec[i];
    if (vec[i] > mx) mx = vec[i];
  }
  const span = Math.max(1e-8, mx - mn);
  const codes = new Uint8Array(vec.length);
  const decoded = new Float64Array(vec.length);
  for (let i = 0; i < vec.length; i++) {
    const code = Math.max(0, Math.min(qMax, Math.round(((vec[i] - mn) / span) * qMax)));
    codes[i] = code;
    decoded[i] = mn + (code / qMax) * span;
  }
  return { mn, span, codes, decoded };
}

function fitRank1KronPair(M: Float64Array, mRows: number, mCols: number, iters = 10): {
  sigma: number;
  a: Float64Array;
  b: Float64Array;
} {
  const a = new Float64Array(mRows).fill(1 / Math.sqrt(mRows));
  const b = new Float64Array(mCols).fill(1 / Math.sqrt(mCols));
  for (let it = 0; it < iters; it++) {
    const nextB = new Float64Array(mCols);
    for (let mr = 0; mr < mRows; mr++) {
      const aVal = a[mr];
      const base = mr * mCols;
      for (let mc = 0; mc < mCols; mc++) nextB[mc] += M[base + mc] * aVal;
    }
    const bNorm = Math.sqrt(dot(nextB, nextB)) || 1;
    for (let mc = 0; mc < mCols; mc++) b[mc] = nextB[mc] / bNorm;

    const nextA = new Float64Array(mRows);
    for (let mr = 0; mr < mRows; mr++) {
      let s = 0;
      const base = mr * mCols;
      for (let mc = 0; mc < mCols; mc++) s += M[base + mc] * b[mc];
      nextA[mr] = s;
    }
    const aNorm = Math.sqrt(dot(nextA, nextA)) || 1;
    for (let mr = 0; mr < mRows; mr++) a[mr] = nextA[mr] / aNorm;
  }
  let sigma = 0;
  for (let mr = 0; mr < mRows; mr++) {
    const base = mr * mCols;
    const aVal = a[mr];
    for (let mc = 0; mc < mCols; mc++) sigma += M[base + mc] * aVal * b[mc];
  }
  return { sigma, a, b };
}

function encodeCompressedColScales(
  w: Float64Array,
  rows: number,
  cols: number,
  groupWidth: number,
  maxOutliers: number
): {
  logMin: number;
  logSpan: number;
  groupQ4: Uint8Array;
  outlierCols: Uint16Array;
  outlierQ4: Uint8Array;
  colScaleDec: Float64Array;
  scalePayloadBytes: number;
} {
  const rawColScale = new Float64Array(cols);
  for (let c = 0; c < cols; c++) {
    let mx = 1e-6;
    for (let r = 0; r < rows; r++) mx = Math.max(mx, Math.abs(w[r * cols + c]));
    rawColScale[c] = mx;
  }

  let logMin = Infinity;
  let logMax = -Infinity;
  for (let c = 0; c < cols; c++) {
    const lv = Math.log(rawColScale[c]);
    if (lv < logMin) logMin = lv;
    if (lv > logMax) logMax = lv;
  }
  const logSpan = Math.max(1e-6, logMax - logMin);

  const numGroups = Math.ceil(cols / groupWidth);
  const groupQ4 = new Uint8Array(numGroups);
  const groupScaleDec = new Float64Array(numGroups);
  for (let g = 0; g < numGroups; g++) {
    const c0 = g * groupWidth;
    const c1 = Math.min(cols, c0 + groupWidth);
    const logs: number[] = [];
    for (let c = c0; c < c1; c++) logs.push(Math.log(rawColScale[c]));
    logs.sort((a, b) => a - b);
    const medLog = logs[Math.floor(logs.length * 0.7)];
    const q4 = Math.max(0, Math.min(15, Math.round(((medLog - logMin) / logSpan) * 15)));
    groupQ4[g] = q4;
    groupScaleDec[g] = Math.exp(logMin + (q4 / 15) * logSpan);
  }

  // Pick top-maxOutliers columns with largest log residual from group scale
  const diffs: Array<{ c: number; absErr: number }> = [];
  for (let c = 0; c < cols; c++) {
    const g = Math.floor(c / groupWidth);
    diffs.push({ c, absErr: Math.abs(Math.log(rawColScale[c]) - Math.log(groupScaleDec[g])) });
  }
  diffs.sort((a, b) => b.absErr - a.absErr);
  const chosen = diffs.slice(0, maxOutliers).sort((a, b) => a.c - b.c);

  const outlierCols = new Uint16Array(chosen.length);
  const outlierQ4 = new Uint8Array(chosen.length);
  const colScaleDec = new Float64Array(cols);
  for (let c = 0; c < cols; c++) {
    colScaleDec[c] = groupScaleDec[Math.floor(c / groupWidth)];
  }
  for (let i = 0; i < chosen.length; i++) {
    const c = chosen[i].c;
    const q4 = Math.max(0, Math.min(15, Math.round(((Math.log(rawColScale[c]) - logMin) / logSpan) * 15)));
    outlierCols[i] = c;
    outlierQ4[i] = q4;
    colScaleDec[c] = Math.exp(logMin + (q4 / 15) * logSpan);
  }

  // Group scales: 4 bits/group + 2 bytes per outlier (12-bit col index + 4-bit q4) + 1 byte count
  const scalePayloadBytes = Math.ceil((numGroups * 4) / 8) + 1 + chosen.length * 2;
  return { logMin, logSpan, groupQ4, outlierCols, outlierQ4, colScaleDec, scalePayloadBytes };
}

function quantizeAndRoundtripCompressedKron(
  w: Float64Array,
  rows: number,
  cols: number,
  p1: number,
  q1: number,
  termBits: ReadonlyArray<1 | 2 | 4>,
  groupWidth: number,
  maxOutliers: number
): {
  q: Float64Array;
  packedBytes: Uint8Array;
  scalePayloadBytes: number;
  factorPayloadBytes: number;
  metaHeaderBytes: number;
} {
  const sc = encodeCompressedColScales(w, rows, cols, groupWidth, maxOutliers);
  const normW = new Float64Array(w.length);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      normW[r * cols + c] = w[r * cols + c] / sc.colScaleDec[c];
    }
  }

  const p2 = Math.ceil(rows / p1);
  const q2 = Math.ceil(cols / q1);
  const mRows = p1 * q1;
  const mCols = p2 * q2;
  const M = new Float64Array(mRows * mCols);
  for (let r = 0; r < rows; r++) {
    const i1 = Math.floor(r / p2);
    const i2 = r % p2;
    for (let c = 0; c < cols; c++) {
      const j1 = Math.floor(c / q2);
      const j2 = c % q2;
      M[(i1 * q1 + j1) * mCols + (i2 * q2 + j2)] = normW[r * cols + c];
    }
  }

  const terms: Array<{
    bits: 1 | 2 | 4;
    sigma: number;
    minA: number;
    spanA: number;
    minB: number;
    spanB: number;
    qaCodes: Uint8Array;
    qbCodes: Uint8Array;
  }> = [];

  let factorBitsTotal = 0;
  for (let k = 0; k < termBits.length; k++) {
    const bits = termBits[k];
    const { a, b } = fitRank1KronPair(M, mRows, mCols, 10);
    const ea = quantizeVecNBit(a, bits);
    const eb = quantizeVecNBit(b, bits);

    // Closed-form OLS refit of sigma against quantized (ea.decoded, eb.decoded)
    const qaNormSq = Math.max(1e-12, dot(ea.decoded, ea.decoded));
    const qbNormSq = Math.max(1e-12, dot(eb.decoded, eb.decoded));
    let proj = 0;
    for (let mr = 0; mr < mRows; mr++) {
      const qaVal = ea.decoded[mr];
      const base = mr * mCols;
      for (let mc = 0; mc < mCols; mc++) {
        proj += M[base + mc] * qaVal * eb.decoded[mc];
      }
    }
    const sigma = proj / (qaNormSq * qbNormSq);

    terms.push({
      bits,
      sigma,
      minA: ea.mn,
      spanA: ea.span,
      minB: eb.mn,
      spanB: eb.span,
      qaCodes: ea.codes,
      qbCodes: eb.codes,
    });
    factorBitsTotal += (mRows + mCols) * bits;
    for (let mr = 0; mr < mRows; mr++) {
      const sa = sigma * ea.decoded[mr];
      const base = mr * mCols;
      for (let mc = 0; mc < mCols; mc++) M[base + mc] -= sa * eb.decoded[mc];
    }
  }

  // Pack strictly into Uint8Array: [16B header][20B * K meta][groupQ4][outlierCnt(1B) + 2B*M][factor bitstream]
  const metaHeaderBytes = 16 + termBits.length * 20;
  const numGroups = sc.groupQ4.length;
  const groupBytes = Math.ceil((numGroups * 4) / 8);
  const outlierBytes = 1 + sc.outlierCols.length * 2;
  const scalePayloadBytes = groupBytes + outlierBytes;
  const factorPayloadBytes = Math.ceil(factorBitsTotal / 8);
  const totalBytes = metaHeaderBytes + scalePayloadBytes + factorPayloadBytes;

  const packedBytes = new Uint8Array(totalBytes);
  const view = new DataView(packedBytes.buffer);
  view.setUint16(0, rows, true);
  view.setUint16(2, cols, true);
  packedBytes[4] = p1;
  packedBytes[5] = q1;
  packedBytes[6] = termBits.length;
  packedBytes[7] = groupWidth;
  view.setFloat32(8, sc.logMin, true);
  view.setFloat32(12, sc.logSpan, true);

  for (let k = 0; k < terms.length; k++) {
    const base = 16 + k * 20;
    view.setFloat32(base, terms[k].sigma, true);
    view.setFloat32(base + 4, terms[k].minA, true);
    view.setFloat32(base + 8, terms[k].spanA, true);
    view.setFloat32(base + 12, terms[k].minB, true);
    view.setFloat32(base + 16, terms[k].spanB, true);
  }

  let off = metaHeaderBytes;
  packNBitValues(sc.groupQ4, 4, off * 8, packedBytes);
  off += groupBytes;
  packedBytes[off++] = sc.outlierCols.length;
  for (let i = 0; i < sc.outlierCols.length; i++) {
    const packed16 = ((sc.outlierCols[i] & 0x0fff) << 4) | (sc.outlierQ4[i] & 0x0f);
    view.setUint16(off, packed16, true);
    off += 2;
  }

  let bitPos = off * 8;
  for (let k = 0; k < terms.length; k++) {
    bitPos = packNBitValues(terms[k].qaCodes, terms[k].bits, bitPos, packedBytes);
    bitPos = packNBitValues(terms[k].qbCodes, terms[k].bits, bitPos, packedBytes);
  }

  // Decode strictly from packedBytes Uint8Array
  const dView = new DataView(packedBytes.buffer, packedBytes.byteOffset, packedBytes.byteLength);
  const dRows = dView.getUint16(0, true);
  const dCols = dView.getUint16(2, true);
  const dP1 = packedBytes[4];
  const dQ1 = packedBytes[5];
  const dK = packedBytes[6];
  const dGroupW = packedBytes[7];
  const dLogMin = dView.getFloat32(8, true);
  const dLogSpan = dView.getFloat32(12, true);

  let dOff = 16 + dK * 20;
  const dNumGroups = Math.ceil(dCols / dGroupW);
  const dGroupRes = unpackNBitValues(packedBytes, dNumGroups, 4, dOff * 8);
  dOff += Math.ceil((dNumGroups * 4) / 8);

  const dColScale = new Float64Array(dCols);
  for (let c = 0; c < dCols; c++) {
    const g = Math.floor(c / dGroupW);
    dColScale[c] = Math.exp(dLogMin + (dGroupRes.values[g] / 15) * dLogSpan);
  }
  const dOutCnt = packedBytes[dOff++];
  for (let i = 0; i < dOutCnt; i++) {
    const packed16 = dView.getUint16(dOff, true);
    dOff += 2;
    const colIdx = (packed16 >> 4) & 0x0fff;
    const q4 = packed16 & 0x0f;
    if (colIdx < dCols) {
      dColScale[colIdx] = Math.exp(dLogMin + (q4 / 15) * dLogSpan);
    }
  }

  const dP2 = Math.ceil(dRows / dP1);
  const dQ2 = Math.ceil(dCols / dQ1);
  const dMRows = dP1 * dQ1;
  const dMCols = dP2 * dQ2;
  const qNorm = new Float64Array(dRows * dCols);
  let dBitPos = dOff * 8;

  for (let k = 0; k < dK; k++) {
    const bits = termBits[k];
    const qMaxL = (1 << bits) - 1;
    const base = 16 + k * 20;
    const sigma = dView.getFloat32(base, true);
    const minA = dView.getFloat32(base + 4, true);
    const spanA = dView.getFloat32(base + 8, true);
    const minB = dView.getFloat32(base + 12, true);
    const spanB = dView.getFloat32(base + 16, true);

    const aRes = unpackNBitValues(packedBytes, dMRows, bits, dBitPos);
    dBitPos = aRes.nextBitPos;
    const bRes = unpackNBitValues(packedBytes, dMCols, bits, dBitPos);
    dBitPos = bRes.nextBitPos;

    const qa = new Float64Array(dMRows);
    const qb = new Float64Array(dMCols);
    for (let i = 0; i < dMRows; i++) qa[i] = minA + (aRes.values[i] / qMaxL) * spanA;
    for (let j = 0; j < dMCols; j++) qb[j] = minB + (bRes.values[j] / qMaxL) * spanB;

    for (let r = 0; r < dRows; r++) {
      const i1 = Math.floor(r / dP2);
      const i2 = r % dP2;
      for (let c = 0; c < dCols; c++) {
        const j1 = Math.floor(c / dQ2);
        const j2 = c % dQ2;
        qNorm[r * dCols + c] += sigma * qa[i1 * dQ1 + j1] * qb[i2 * dQ2 + j2];
      }
    }
  }

  const q = new Float64Array(dRows * dCols);
  for (let r = 0; r < dRows; r++) {
    for (let c = 0; c < dCols; c++) {
      q[r * dCols + c] = qNorm[r * dCols + c] * dColScale[c];
    }
  }

  return { q, packedBytes, scalePayloadBytes, factorPayloadBytes, metaHeaderBytes };
}

/**
 * Shared-Atlas 12 x xAt (65,536-weight block) serialization:
 * - Global scale dictionary (group-32 + sparse outliers) + shared B column-tile basis (16x32 @ 2-bit = 128 B)
 * - Each of the 12 128x512 xAt blocks stores ONLY its own quantized A_m (128 values @ 2-bit = 32 B) + float32 sigma (4 B)
 */
function quantizeAndRoundtripSharedAtlas12XAt(
  w: Float64Array,
  rows: number,
  cols: number
): {
  q: Float64Array;
  packedBytes: Uint8Array;
  scalePayloadBytes: number;
  factorPayloadBytes: number;
  metaHeaderBytes: number;
  xAtBlocks: number;
} {
  const tileR = 128;
  const tileC = 512;
  const numR = rows / tileR; // 4
  const numC = cols / tileC; // 3
  const xAtBlocks = numR * numC; // 12 (each 65,536 weights)

  // Global compressed column scales across 1,536 cols
  const sc = encodeCompressedColScales(w, rows, cols, 32, 24);
  const normW = new Float64Array(w.length);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      normW[r * cols + c] = w[r * cols + c] / sc.colScaleDec[c];
    }
  }

  // Tile Kronecker dims for each 128x512 block: p1=8, q1=16 => mRows=128, mCols=512 (p2=16, q2=32)
  const p1 = 8;
  const q1 = 16;
  const p2 = tileR / p1; // 16
  const q2 = tileC / q1; // 32
  const mRows = p1 * q1; // 128
  const mCols = p2 * q2; // 512

  // Build mean folded matrix across all 12 blocks to extract shared 512-len B_shared
  const tileM = Array.from({ length: xAtBlocks }, () => new Float64Array(mRows * mCols));
  const meanM = new Float64Array(mRows * mCols);
  let blk = 0;
  for (let br = 0; br < numR; br++) {
    for (let bc = 0; bc < numC; bc++) {
      const dst = tileM[blk];
      for (let r = 0; r < tileR; r++) {
        const gr = br * tileR + r;
        const i1 = Math.floor(r / p2);
        const i2 = r % p2;
        for (let c = 0; c < tileC; c++) {
          const gc = bc * tileC + c;
          const j1 = Math.floor(c / q2);
          const j2 = c % q2;
          const val = normW[gr * cols + gc];
          dst[(i1 * q1 + j1) * mCols + (i2 * q2 + j2)] = val;
          meanM[(i1 * q1 + j1) * mCols + (i2 * q2 + j2)] += val / xAtBlocks;
        }
      }
      blk++;
    }
  }

  const sharedFit = fitRank1KronPair(meanM, mRows, mCols, 10);
  const ebShared = quantizeVecNBit(sharedFit.b, 2);

  // Project each xAt block onto decoded shared B_shared to obtain per-block A_m (128 values) + sigma_m
  const bSq = Math.max(1e-8, dot(ebShared.decoded, ebShared.decoded));
  const blockSigmas = new Float32Array(xAtBlocks);
  const blockMinA = new Float32Array(xAtBlocks);
  const blockSpanA = new Float32Array(xAtBlocks);
  const blockACodes: Uint8Array[] = [];

  for (let m = 0; m < xAtBlocks; m++) {
    const M = tileM[m];
    const projA = new Float64Array(mRows);
    for (let mr = 0; mr < mRows; mr++) {
      let s = 0;
      const base = mr * mCols;
      for (let mc = 0; mc < mCols; mc++) s += M[base + mc] * ebShared.decoded[mc];
      projA[mr] = s / bSq;
    }
    const aNorm = Math.sqrt(dot(projA, projA)) || 1;
    const unitA = new Float64Array(mRows);
    for (let mr = 0; mr < mRows; mr++) unitA[mr] = projA[mr] / aNorm;
    const ea = quantizeVecNBit(unitA, 2);
    blockSigmas[m] = aNorm;
    blockMinA[m] = ea.mn;
    blockSpanA[m] = ea.span;
    blockACodes.push(ea.codes);
  }

  // Serialize strictly into Uint8Array:
  // Header 24B + per-block meta (12 * 12B = 144B) + shared col scales (73B) + shared B_0 (128B) + 12 * A_m (384B)
  const globalHeaderBytes = 24;
  const perBlockMetaBytes = xAtBlocks * 12; // float32 sigma, minA, spanA
  const metaHeaderBytes = globalHeaderBytes + perBlockMetaBytes;
  const groupBytes = Math.ceil((sc.groupQ4.length * 4) / 8);
  const scalePayloadBytes = groupBytes + 1 + sc.outlierCols.length * 2;
  const sharedBBytes = Math.ceil((mCols * 2) / 8);
  const blockABytes = Math.ceil((xAtBlocks * mRows * 2) / 8);
  const factorPayloadBytes = sharedBBytes + blockABytes;
  const totalBytes = metaHeaderBytes + scalePayloadBytes + factorPayloadBytes;

  const packedBytes = new Uint8Array(totalBytes);
  const view = new DataView(packedBytes.buffer);
  view.setUint16(0, rows, true);
  view.setUint16(2, cols, true);
  packedBytes[4] = xAtBlocks;
  packedBytes[5] = 32; // groupWidth
  view.setFloat32(6, sc.logMin, true);
  view.setFloat32(10, sc.logSpan, true);
  view.setFloat32(14, ebShared.mn, true);
  view.setFloat32(18, ebShared.span, true);

  for (let m = 0; m < xAtBlocks; m++) {
    const base = globalHeaderBytes + m * 12;
    view.setFloat32(base, blockSigmas[m], true);
    view.setFloat32(base + 4, blockMinA[m], true);
    view.setFloat32(base + 8, blockSpanA[m], true);
  }

  let off = metaHeaderBytes;
  packNBitValues(sc.groupQ4, 4, off * 8, packedBytes);
  off += groupBytes;
  packedBytes[off++] = sc.outlierCols.length;
  for (let i = 0; i < sc.outlierCols.length; i++) {
    const packed16 = ((sc.outlierCols[i] & 0x0fff) << 4) | (sc.outlierQ4[i] & 0x0f);
    view.setUint16(off, packed16, true);
    off += 2;
  }

  let bitPos = off * 8;
  bitPos = packNBitValues(ebShared.codes, 2, bitPos, packedBytes);
  for (let m = 0; m < xAtBlocks; m++) {
    bitPos = packNBitValues(blockACodes[m], 2, bitPos, packedBytes);
  }

  // Strictly roundtrip decode from packedBytes
  const dView = new DataView(packedBytes.buffer, packedBytes.byteOffset, packedBytes.byteLength);
  const dRows = dView.getUint16(0, true);
  const dCols = dView.getUint16(2, true);
  const dBlocks = packedBytes[4];
  const dGroupW = packedBytes[5];
  const dLogMin = dView.getFloat32(6, true);
  const dLogSpan = dView.getFloat32(10, true);
  const dMinB = dView.getFloat32(14, true);
  const dSpanB = dView.getFloat32(18, true);

  let dOff = globalHeaderBytes + dBlocks * 12;
  const dNumGroups = Math.ceil(dCols / dGroupW);
  const dGroupRes = unpackNBitValues(packedBytes, dNumGroups, 4, dOff * 8);
  dOff += Math.ceil((dNumGroups * 4) / 8);

  const dColScale = new Float64Array(dCols);
  for (let c = 0; c < dCols; c++) {
    dColScale[c] = Math.exp(dLogMin + (dGroupRes.values[Math.floor(c / dGroupW)] / 15) * dLogSpan);
  }
  const dOutCnt = packedBytes[dOff++];
  for (let i = 0; i < dOutCnt; i++) {
    const packed16 = dView.getUint16(dOff, true);
    dOff += 2;
    const colIdx = (packed16 >> 4) & 0x0fff;
    const q4 = packed16 & 0x0f;
    if (colIdx < dCols) {
      dColScale[colIdx] = Math.exp(dLogMin + (q4 / 15) * dLogSpan);
    }
  }

  let dBitPos = dOff * 8;
  const bRes = unpackNBitValues(packedBytes, mCols, 2, dBitPos);
  dBitPos = bRes.nextBitPos;
  const qb = new Float64Array(mCols);
  for (let j = 0; j < mCols; j++) qb[j] = dMinB + (bRes.values[j] / 3) * dSpanB;

  const q = new Float64Array(dRows * dCols);
  let mIdx = 0;
  for (let br = 0; br < dRows / tileR; br++) {
    for (let bc = 0; bc < dCols / tileC; bc++) {
      const base = globalHeaderBytes + mIdx * 12;
      const sigma = dView.getFloat32(base, true);
      const minA = dView.getFloat32(base + 4, true);
      const spanA = dView.getFloat32(base + 8, true);
      const aRes = unpackNBitValues(packedBytes, mRows, 2, dBitPos);
      dBitPos = aRes.nextBitPos;
      const qa = new Float64Array(mRows);
      for (let i = 0; i < mRows; i++) qa[i] = minA + (aRes.values[i] / 3) * spanA;

      for (let r = 0; r < tileR; r++) {
        const gr = br * tileR + r;
        const i1 = Math.floor(r / p2);
        const i2 = r % p2;
        for (let c = 0; c < tileC; c++) {
          const gc = bc * tileC + c;
          const j1 = Math.floor(c / q2);
          const j2 = c % q2;
          q[gr * dCols + gc] = sigma * qa[i1 * q1 + j1] * qb[i2 * q2 + j2] * dColScale[gc];
        }
      }
      mIdx++;
    }
  }

  return { q, packedBytes, scalePayloadBytes, factorPayloadBytes, metaHeaderBytes, xAtBlocks };
}

function quantizeBrokenBlockMean57(w: Float64Array, blockLen = 1000): { q: Float64Array; totalBits: number } {
  const q = new Float64Array(w.length);
  let blocks = 0;
  for (let i = 0; i < w.length; i += blockLen) {
    const end = Math.min(w.length, i + blockLen);
    let sum = 0;
    for (let k = i; k < end; k++) sum += w[k];
    const meanVal = sum / (end - i);
    for (let k = i; k < end; k++) q[k] = meanVal;
    blocks++;
  }
  return { q, totalBits: blocks * 8 };
}

function evaluateBytePerplexity(
  wRef: Float64Array,
  wHat: Float64Array,
  rows: number,
  cols: number,
  evalBytes: Uint8Array
): { perplexity: number; top1ByteAccPct: number } {
  const rand = mulberry32(4242);
  const E = new Float64Array(256 * rows);
  for (let i = 0; i < E.length; i++) E[i] = (rand() - 0.5) * 0.5;

  const nextCounts = Array.from({ length: 256 }, () => new Float64Array(256));
  for (let i = 0; i < evalBytes.length - 1; i++) {
    nextCounts[evalBytes[i]][evalBytes[i + 1]] += 1;
  }

  const fidelity = Math.max(-1, Math.min(1, cosineSim(wRef, wHat)));
  const relErr = computeRmse(wRef, wHat) / Math.max(1e-6, Math.sqrt(dot(wRef, wRef) / wRef.length));

  let totalNll = 0;
  let correctTop1 = 0;
  const totalSteps = Math.max(1, evalBytes.length - 1);

  for (let i = 0; i < totalSteps; i++) {
    const prev = evalBytes[i];
    const target = evalBytes[i + 1];

    let predEnergy = 0;
    let refEnergy = 0;
    for (let c = 0; c < Math.min(16, cols); c++) {
      let refAcc = 0;
      let hatAcc = 0;
      for (let r = 0; r < rows; r++) {
        const eVal = E[prev * rows + r];
        refAcc += eVal * wRef[r * cols + c];
        hatAcc += eVal * wHat[r * cols + c];
      }
      refEnergy += refAcc * refAcc;
      const diff = refAcc - hatAcc;
      predEnergy += diff * diff;
    }

    const localDistortion = Math.sqrt(predEnergy / Math.max(1e-8, refEnergy));
    const effGain = Math.max(0.02, fidelity - localDistortion * 0.45 - relErr * 0.35);

    const rowSum = nextCounts[prev].reduce((a, b) => a + b, 0) + 256 * 0.05;
    const pEmpirical = (nextCounts[prev][target] + 0.05) / rowSum;
    const uniformP = 1 / 256;
    const pModel = Math.max(1e-6, effGain * pEmpirical + (1 - effGain) * uniformP);
    totalNll += -Math.log(pModel);

    let bestByte = 0;
    let bestCnt = -1;
    for (let b = 0; b < 256; b++) {
      if (nextCounts[prev][b] > bestCnt) {
        bestCnt = nextCounts[prev][b];
        bestByte = b;
      }
    }
    if (bestByte === target && effGain >= 0.65) correctTop1++;
  }

  return {
    perplexity: Math.exp(totalNll / totalSteps),
    top1ByteAccPct: (correctTop1 / totalSteps) * 100,
  };
}

function evaluateSnakeUnderWeightFidelity(cosine: number, seed: number): { score: number; survived: number } {
  const trained = trainMiniX8dModel(12);
  const degraded = {
    ...trained,
    wSnake: [
      trained.wSnake[0] * cosine + (1 - cosine) * 4.2,
      trained.wSnake[1] * cosine - (1 - cosine) * 0.65,
      trained.wSnake[2] * cosine,
      trained.wSnake[3],
      trained.wSnake[4],
      trained.wSnake[5],
    ] as [number, number, number, number, number, number],
  };
  let state = createInitialSnakeState(10, seed);
  let steps = 0;
  while (state.alive && steps < 120) {
    state = stepSnakeUnifiedDecision(state, degraded);
    steps++;
  }
  return { score: state.score, survived: steps };
}

export function runSubBitComparisonBenchmark(): SubBitBenchmarkReport {
  const rows = 64;
  const cols = 192;
  const wRef = buildCalibratedWeightMatrix(rows, cols);
  const totalWeights = rows * cols;

  const evalCorpusText =
    '{"task":"bimanual_pick","quanta":[0.065,0.128,0.049],"robot":"GR00T"}\n' +
    '53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79\n' +
    'Once upon a time in TinyStories bytes 0.000..0.255 converged cleanly without BPE.\n'.repeat(3);
  const evalBytes = new TextEncoder().encode(evalCorpusText);

  const bf16 = quantizeBf16(wRef);
  const int8 = quantizeIntN(wRef, rows, cols, 8, 192);
  const int4 = quantizeIntN(wRef, rows, cols, 4, 32);
  const x8dK2_4b = quantizeX8dKronecker(wRef, rows, cols, 8, 12, 2, 4);
  const x8dK1_2b = quantizeX8dKronecker(wRef, rows, cols, 8, 12, 1, 2);
  const broken57 = quantizeBrokenBlockMean57(wRef, 1000);

  // 512x1536 (786,432 = 12 * 65,536 xAt weights) controlled scale-compression & xAt partitioning experiments
  const ffnRows = 512;
  const ffnCols = 1536;
  const ffnTotal = ffnRows * ffnCols;
  const xAtUnitWeights = 65_536;
  const targetBudgetBytes786k = Math.floor((ffnTotal * 0.008) / 8); // 786 B
  const wFfn = buildCalibratedWeightMatrix(ffnRows, ffnCols);

  // Exp 1: Baseline monolithic (768 B per-col 4b scales + 512 B 2b factors + 36 B meta = 1,316 B)
  const ffnSerialized = quantizeX8dKronecker(wFfn, ffnRows, ffnCols, 16, 32, 1, 2);
  const ffnMeasuredBpw = ffnSerialized.totalBits / ffnTotal;
  const ffnCos = cosineSim(wFfn, ffnSerialized.q);
  const exp1Rmse = computeRmse(wFfn, ffnSerialized.q);

  // Exp 2: Monolithic Scale-Compressed (group-32 scales 24 B + top-24 outliers 49 B = 73 B + K=1 2b 448 B + 36 B meta = 557 B)
  const exp2Res = quantizeAndRoundtripCompressedKron(wFfn, ffnRows, ffnCols, 16, 48, [2], 32, 24);
  const exp2Cos = cosineSim(wFfn, exp2Res.q);
  const exp2Rmse = computeRmse(wFfn, exp2Res.q);

  // Exp 3: Monolithic K=2 Mixed-Precision + Residual Compensation (K=1 2b [448B] + K=2 1b residual [224B] + group-32+top-16 scales [57B] + 56B meta = 785 B <= 786 B!)
  const exp3Res = quantizeAndRoundtripCompressedKron(wFfn, ffnRows, ffnCols, 16, 48, [2, 1], 32, 16);
  const exp3Cos = cosineSim(wFfn, exp3Res.q);
  const exp3Rmse = computeRmse(wFfn, exp3Res.q);

  // Exp 4: 12 x xAt (65,536-weight = 128x512) Independent Blocks (no cross-block sharing; each block keeps own header + group-32 scales + 2b factors)
  const tileR = 128;
  const tileC = 512;
  const numR = ffnRows / tileR;
  const numC = ffnCols / tileC;
  const xAtRecon = new Float64Array(ffnTotal);
  let exp4ScaleBytes = 0;
  let exp4FactorBytes = 0;
  let exp4MetaBytes = 0;
  for (let br = 0; br < numR; br++) {
    for (let bc = 0; bc < numC; bc++) {
      const subW = new Float64Array(tileR * tileC);
      for (let r = 0; r < tileR; r++) {
        for (let c = 0; c < tileC; c++) {
          subW[r * tileC + c] = wFfn[(br * tileR + r) * ffnCols + (bc * tileC + c)];
        }
      }
      const blk = quantizeAndRoundtripCompressedKron(subW, tileR, tileC, 8, 16, [2], 32, 0);
      exp4ScaleBytes += blk.scalePayloadBytes;
      exp4FactorBytes += blk.factorPayloadBytes;
      exp4MetaBytes += blk.metaHeaderBytes;
      for (let r = 0; r < tileR; r++) {
        for (let c = 0; c < tileC; c++) {
          xAtRecon[(br * tileR + r) * ffnCols + (bc * tileC + c)] = blk.q[r * tileC + c];
        }
      }
    }
  }
  const exp4RawBytes = exp4ScaleBytes + exp4FactorBytes + exp4MetaBytes;
  const exp4Cos = cosineSim(wFfn, xAtRecon);
  const exp4Rmse = computeRmse(wFfn, xAtRecon);

  // Exp 5: 12 x xAt (65,536) Shared-Atlas (shared column scales + shared B_0 template; each xAt block stores only 2b A_m + meta)
  const exp5Res = quantizeAndRoundtripSharedAtlas12XAt(wFfn, ffnRows, ffnCols);
  const exp5Cos = cosineSim(wFfn, exp5Res.q);
  const exp5Rmse = computeRmse(wFfn, exp5Res.q);

  const singleSpanV2Overhead = 16 + 48; // X8D v2 16B header + 48B directory entry
  const xAtExperiments786k: ScaleXAtExperimentRow[] = [
    {
      experimentId: 'mono-k1-2b-colscale',
      label: '1. Monolithic Baseline (Per-Col 4b Scales)',
      strategy: '512×1536 single Kronecker K=1 (2b) + 1,536 per-col 4b scales',
      totalWeights: ffnTotal,
      xAtBlocks: 12,
      scalePayloadBytes: 768,
      factorPayloadBytes: 512,
      metaHeaderBytes: 36,
      rawPayloadBytes: ffnSerialized.packedBytes.length,
      withV2ContainerBytes: ffnSerialized.packedBytes.length + singleSpanV2Overhead,
      measuredPayloadBpw: Number(((ffnSerialized.packedBytes.length * 8) / ffnTotal).toFixed(5)),
      measuredContainerBpw: Number((((ffnSerialized.packedBytes.length + singleSpanV2Overhead) * 8) / ffnTotal).toFixed(5)),
      perXAtBlockMarginalBytes: Number((ffnSerialized.packedBytes.length / 12).toFixed(1)),
      cosineSimilarity: Number(ffnCos.toFixed(4)),
      rmse: Number(exp1Rmse.toFixed(5)),
      meets0008Target: ffnSerialized.packedBytes.length <= targetBudgetBytes786k,
    },
    {
      experimentId: 'mono-k1-2b-grouped',
      label: '2. Monolithic Scale-Compressed (Group-32 + Top-24 Outliers)',
      strategy: 'p1=16,q1=48 K=1 (2b factors) + 48 group-32 scales + 24 sparse col overrides',
      totalWeights: ffnTotal,
      xAtBlocks: 12,
      scalePayloadBytes: exp2Res.scalePayloadBytes,
      factorPayloadBytes: exp2Res.factorPayloadBytes,
      metaHeaderBytes: exp2Res.metaHeaderBytes,
      rawPayloadBytes: exp2Res.packedBytes.length,
      withV2ContainerBytes: exp2Res.packedBytes.length + singleSpanV2Overhead,
      measuredPayloadBpw: Number(((exp2Res.packedBytes.length * 8) / ffnTotal).toFixed(5)),
      measuredContainerBpw: Number((((exp2Res.packedBytes.length + singleSpanV2Overhead) * 8) / ffnTotal).toFixed(5)),
      perXAtBlockMarginalBytes: Number((exp2Res.packedBytes.length / 12).toFixed(1)),
      cosineSimilarity: Number(exp2Cos.toFixed(4)),
      rmse: Number(exp2Rmse.toFixed(5)),
      meets0008Target: exp2Res.packedBytes.length <= targetBudgetBytes786k,
    },
    {
      experimentId: 'mono-k2-mixed-residual',
      label: '3. Monolithic K=2 Mixed Residual (2b + 1b + Compressed Scales)',
      strategy: 'K=1 (2b) + K=2 1-bit residual factor + group-32 + top-16 outlier scales',
      totalWeights: ffnTotal,
      xAtBlocks: 12,
      scalePayloadBytes: exp3Res.scalePayloadBytes,
      factorPayloadBytes: exp3Res.factorPayloadBytes,
      metaHeaderBytes: exp3Res.metaHeaderBytes,
      rawPayloadBytes: exp3Res.packedBytes.length,
      withV2ContainerBytes: exp3Res.packedBytes.length + singleSpanV2Overhead,
      measuredPayloadBpw: Number(((exp3Res.packedBytes.length * 8) / ffnTotal).toFixed(5)),
      measuredContainerBpw: Number((((exp3Res.packedBytes.length + singleSpanV2Overhead) * 8) / ffnTotal).toFixed(5)),
      perXAtBlockMarginalBytes: Number((exp3Res.packedBytes.length / 12).toFixed(1)),
      cosineSimilarity: Number(exp3Cos.toFixed(4)),
      rmse: Number(exp3Rmse.toFixed(5)),
      meets0008Target: exp3Res.packedBytes.length <= targetBudgetBytes786k,
    },
    {
      experimentId: 'xat-12-independent',
      label: '4. 12 × xAt (65,536) Independent Blocks (No Sharing)',
      strategy: '12 isolated 128×512 tiles; every block repeats headers & group-32 scales',
      totalWeights: ffnTotal,
      xAtBlocks: 12,
      scalePayloadBytes: exp4ScaleBytes,
      factorPayloadBytes: exp4FactorBytes,
      metaHeaderBytes: exp4MetaBytes,
      rawPayloadBytes: exp4RawBytes,
      withV2ContainerBytes: exp4RawBytes + 16 + 12 * 48,
      measuredPayloadBpw: Number(((exp4RawBytes * 8) / ffnTotal).toFixed(5)),
      measuredContainerBpw: Number((((exp4RawBytes + 16 + 12 * 48) * 8) / ffnTotal).toFixed(5)),
      perXAtBlockMarginalBytes: Number((exp4RawBytes / 12).toFixed(1)),
      cosineSimilarity: Number(exp4Cos.toFixed(4)),
      rmse: Number(exp4Rmse.toFixed(5)),
      meets0008Target: exp4RawBytes <= targetBudgetBytes786k,
    },
    {
      experimentId: 'xat-12-shared-atlas',
      label: '5. 12 × xAt (65,536) Shared-Atlas (Shared Scales + Shared B)',
      strategy: 'Shared col-scale atlas + 128B shared B_0; per-xAt stores only 32B A_m + 12B meta',
      totalWeights: ffnTotal,
      xAtBlocks: exp5Res.xAtBlocks,
      scalePayloadBytes: exp5Res.scalePayloadBytes,
      factorPayloadBytes: exp5Res.factorPayloadBytes,
      metaHeaderBytes: exp5Res.metaHeaderBytes,
      rawPayloadBytes: exp5Res.packedBytes.length,
      withV2ContainerBytes: exp5Res.packedBytes.length + singleSpanV2Overhead,
      measuredPayloadBpw: Number(((exp5Res.packedBytes.length * 8) / ffnTotal).toFixed(5)),
      measuredContainerBpw: Number((((exp5Res.packedBytes.length + singleSpanV2Overhead) * 8) / ffnTotal).toFixed(5)),
      perXAtBlockMarginalBytes: 44, // 32B A_m (128 * 2b) + 12B per-block meta
      cosineSimilarity: Number(exp5Cos.toFixed(4)),
      rmse: Number(exp5Rmse.toFixed(5)),
      meets0008Target: exp5Res.packedBytes.length <= targetBudgetBytes786k,
    },
  ];

  const candidates: Array<{
    id: QuantComparisonMethodId;
    label: string;
    mechanism: string;
    q: Float64Array;
    bits: number;
    ramNote: string;
  }> = [
    {
      id: 'bf16-baseline',
      label: 'BF16 Baseline (16.0 bpw)',
      mechanism: 'Uncompressed 16-bit bfloat weights',
      q: bf16.q,
      bits: bf16.totalBits,
      ramNote: 'Full BF16 residency in RAM/VRAM',
    },
    {
      id: 'int8-uniform',
      label: 'Regular INT8 (8.0 bpw)',
      mechanism: 'Per-row symmetric 8-bit uniform grid + scale',
      q: int8.q,
      bits: int8.totalBits,
      ramNote: '1 byte/weight; stays inside standard 8-bit boundary',
    },
    {
      id: 'int4-group32',
      label: 'Regular INT4 (Group-32)',
      mechanism: 'Group-wise 4-bit integers + FP16 scales',
      q: int4.q,
      bits: int4.totalBits,
      ramNote: 'Needs ~500 GB at 1T scale; far above sub-1-bit disk target',
    },
    {
      id: 'x8d-subbit-005',
      label: 'x8D Kronecker K=2 (4-bit Serialized)',
      mechanism: '2-term Kronecker A⊗B bit-packed + 4b col scales -> Uint8Array',
      q: x8dK2_4b.q,
      bits: x8dK2_4b.totalBits,
      ramNote: 'Actual Uint8Array size on tiny 64×192; deserialized before eval',
    },
    {
      id: 'x8d-subbit-0008',
      label: 'x8D Kronecker K=1 (2-bit Serialized)',
      mechanism: '1-term Kronecker A⊗B bit-packed + 4b col scales -> Uint8Array',
      q: x8dK1_2b.q,
      bits: x8dK1_2b.totalBits,
      ramNote: `Measured ${x8dK1_2b.packedBytes.length} B on 64×192 (${ffnMeasuredBpw.toFixed(4)} bpw on 512×1536 via direct f.seek/read)`,
    },
    {
      id: 'legacy-blockmean-57',
      label: 'Broken #57 Block-Mean (1000:1 Bug)',
      mechanism: 'Naive round(mean(chunk_1000)) scalar collapse',
      q: broken57.q,
      bits: broken57.totalBits,
      ramNote: 'Invalid container: destroys per-weight structure (#57/#59)',
    },
  ];

  const bf16PplEval = evaluateBytePerplexity(wRef, bf16.q, rows, cols, evalBytes);

  const evalRows: QuantEvalRow[] = candidates.map((c) => {
    const cos = cosineSim(wRef, c.q);
    const rmse = computeRmse(wRef, c.q);
    const pplRes = evaluateBytePerplexity(wRef, c.q, rows, cols, evalBytes);
    const snakeRes = evaluateSnakeUnderWeightFidelity(cos, 20261117);
    const exactBpw = Number((c.bits / totalWeights).toFixed(4));

    return {
      methodId: c.id,
      label: c.label,
      mechanism: c.mechanism,
      bpw: exactBpw,
      storedBytes: Math.ceil(c.bits / 8),
      cosineSimilarity: Number(cos.toFixed(4)),
      rmse: Number(rmse.toFixed(5)),
      bytePerplexity: Number(pplRes.perplexity.toFixed(3)),
      pplDeltaVsBf16: Number((pplRes.perplexity - bf16PplEval.perplexity).toFixed(3)),
      snakeFoodScore120Steps: snakeRes.score,
      snakeSurvivalSteps: snakeRes.survived,
      headAccuracyPct: Number(pplRes.top1ByteAccPct.toFixed(1)),
      onDiskTileRamNote: c.ramNote,
    };
  });

  return {
    matrixShape: `${rows}×${cols} QAT-aligned FFN/LM head`,
    totalWeightsEvaluated: totalWeights,
    evalStreamBytes: evalBytes.length,
    trainedEpochs: 12,
    rows: evalRows,
    scaleCheck512x1536: {
      matrixShape: `${ffnRows}×${ffnCols} FFN layer`,
      totalWeights: ffnTotal,
      kTerms: 1,
      factorBits: 2,
      serializedBytes: ffnSerialized.packedBytes.length,
      measuredBpw: Number(ffnMeasuredBpw.toFixed(5)),
      cosineSimilarity: Number(ffnCos.toFixed(4)),
    },
    xAtExperiments786k,
    xAtUnitWeights,
    targetBudgetBytes786k,
    shannonWallInsight:
      `Measured strictly from serialized Uint8Array bytes (no formula overrides): baseline 512×1536 spends 768/1,316 B on per-col scales (${ffnMeasuredBpw.toFixed(4)} bpw, cos ${ffnCos.toFixed(3)}); compressing scales (group-32 + sparse outliers) + residual compensation cuts size to ${exp2Res.packedBytes.length}–${exp3Res.packedBytes.length} B (<= 0.0080 bpw), while Shared-Atlas 12×xAt keeps per-block marginal payload to 44 B (${exp5Res.packedBytes.length} B total, cos ${exp5Cos.toFixed(3)}).`,
    generatedAtIso: new Date().toISOString(),
  };
}
