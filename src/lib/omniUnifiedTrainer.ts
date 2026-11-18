/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * Unified 0.73M Effective-Param Compact-Native x8D Trainer & Direct f.seek/f.read Runtime.
 * - Domain: strictly 256 byte states (0..255 <-> 0.000..0.255), zero BPE, zero MASK, zero -1 ternary.
 * - Architecture: Kronecker E + Trunk + Trainable System-1/System-2 4-Expert Router + Kronecker Head.
 * - Trained natively in compact Kronecker STE space from step 0 over real HF Omni datasets on disk.
 * - Saved to magic-free .x8D file with 24-byte tail seek directory; served strictly via fs.openSync + fs.readSync.
 */

import fs from 'fs';
import path from 'path';
import {
  OmniDomainId,
  getDefaultOmniDataDir,
  loadDomainSplitBytes,
  readOmniManifest,
} from './omniDatasetIngestor';

export const OMNI_CONTEXT_LEN = 8;
export const OMNI_EMB_DIM = 32;
export const OMNI_D_MODEL = OMNI_CONTEXT_LEN * OMNI_EMB_DIM; // 256
export const OMNI_HIDDEN_EXP = 512;
export const OMNI_VOCAB_SIZE = 256;
export const OMNI_NUM_EXPERTS = 4;

export interface OmniKronMeta {
  rows: number;
  cols: number;
  p1: number;
  q1: number;
  p2: number;
  q2: number;
  kTerms: number;
  sigma: number;
}

export interface OmniKronLayer {
  meta: OmniKronMeta;
  A: Float32Array[];
  B: Float32Array[];
}

export interface OmniTrainingMetricsSnapshot {
  status: 'idle' | 'running' | 'paused' | 'completed' | 'error';
  step: number;
  targetSteps: number;
  epochsApprox: number;
  bytesProcessed: number;
  throughputBytesPerSec: number;
  elapsedMs: number;
  currentTrainLoss: number;
  initialTrainLoss: number;
  lossHistory: Array<{ step: number; trainLoss: number; meanHeldOutPpl: number }>;
  perDomainHeldOutPpl: Partial<Record<OmniDomainId, { initPpl: number; currentPpl: number; top1AccPct: number }>>;
  expertUsagePct: [number, number, number, number];
  effectiveWeights: number;
  trainableDofs: number;
  fp16EquivalentBytes: number;
  checkpointPath: string;
  checkpointBytesOnDisk: number;
  achievedBpw: number;
  lastError: string | null;
}

export interface OmniDiskInferResult {
  prompt: string;
  continuation: string;
  outputQuantaSample: string[];
  activeExpertsUsed: number[];
  diskFileSizeBytes: number;
  diskBytesSeekRead: number;
  diskSeeksPerformed: number;
  latencyMs: number;
}

const DOMAIN_ORDER: OmniDomainId[] = [
  'groot_sim',
  'lerobot_control',
  'tinystories_prose',
  'gsm8k_math',
  'humaneval_code',
  'sudoku_extreme',
];

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
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

function pack4Bit(codes: Uint8Array, outBytes: Uint8Array, byteOffset: number): number {
  const needed = Math.ceil(codes.length / 2);
  for (let i = 0; i < needed; i++) {
    const hi = codes[i * 2] & 0x0f;
    const lo = i * 2 + 1 < codes.length ? codes[i * 2 + 1] & 0x0f : 0;
    outBytes[byteOffset + i] = (hi << 4) | lo;
  }
  return byteOffset + needed;
}

function unpack4Bit(inBytes: Uint8Array, byteOffset: number, count: number): Uint8Array {
  const out = new Uint8Array(count);
  for (let i = 0; i < count; i++) {
    const b = inBytes[byteOffset + (i >> 1)];
    out[i] = (i & 1) === 0 ? (b >> 4) & 0x0f : b & 0x0f;
  }
  return out;
}

function kronForward(
  meta: OmniKronMeta,
  qA: Float32Array[],
  qB: Float32Array[],
  x: Float32Array,
  yOut: Float32Array,
  tmpT: Float32Array
): void {
  const { p1, q1, p2, q2, kTerms, sigma } = meta;
  yOut.fill(0);
  for (let k = 0; k < kTerms; k++) {
    const Ak = qA[k];
    const Bk = qB[k];
    for (let i1 = 0; i1 < p1; i1++) {
      const xOff = i1 * p2;
      const tOff = i1 * q2;
      for (let j2 = 0; j2 < q2; j2++) {
        let s = 0;
        for (let i2 = 0; i2 < p2; i2++) s += x[xOff + i2] * Bk[i2 * q2 + j2];
        tmpT[tOff + j2] = s;
      }
    }
    for (let j1 = 0; j1 < q1; j1++) {
      const yOff = j1 * q2;
      for (let j2 = 0; j2 < q2; j2++) {
        let s = 0;
        for (let i1 = 0; i1 < p1; i1++) s += Ak[i1 * q1 + j1] * tmpT[i1 * q2 + j2];
        yOut[yOff + j2] += sigma * s;
      }
    }
  }
}

function kronBackward(
  meta: OmniKronMeta,
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
  const { p1, q1, p2, q2, kTerms, sigma } = meta;
  gXOut.fill(0);
  for (let k = 0; k < kTerms; k++) {
    const Ak = qA[k];
    const Bk = qB[k];
    const gAk = gA[k];
    const gBk = gB[k];

    for (let i1 = 0; i1 < p1; i1++) {
      const xOff = i1 * p2;
      const tOff = i1 * q2;
      for (let j2 = 0; j2 < q2; j2++) {
        let s = 0;
        for (let i2 = 0; i2 < p2; i2++) s += x[xOff + i2] * Bk[i2 * q2 + j2];
        T[tOff + j2] = s;
      }
    }
    for (let i1 = 0; i1 < p1; i1++) {
      const tOff = i1 * q2;
      for (let j1 = 0; j1 < q1; j1++) {
        const gyOff = j1 * q2;
        let s = 0;
        for (let j2 = 0; j2 < q2; j2++) s += T[tOff + j2] * gY[gyOff + j2];
        gAk[i1 * q1 + j1] += sigma * s;
      }
    }
    for (let i1 = 0; i1 < p1; i1++) {
      const aOff = i1 * q1;
      const agOff = i1 * q2;
      for (let j2 = 0; j2 < q2; j2++) {
        let s = 0;
        for (let j1 = 0; j1 < q1; j1++) s += Ak[aOff + j1] * gY[j1 * q2 + j2];
        AG[agOff + j2] = s;
      }
    }
    for (let i2 = 0; i2 < p2; i2++) {
      for (let j2 = 0; j2 < q2; j2++) {
        let s = 0;
        for (let i1 = 0; i1 < p1; i1++) s += x[i1 * p2 + i2] * AG[i1 * q2 + j2];
        gBk[i2 * q2 + j2] += sigma * s;
      }
    }
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

function evalEmbedSlot(
  qA: Float32Array[],
  qB: Float32Array[],
  meta: OmniKronMeta,
  byteId: number,
  outSlice: Float32Array,
  outOffset: number
): void {
  const i1 = Math.floor(byteId / meta.p2);
  const i2 = byteId % meta.p2;
  // Blend direct 0.000..0.255 quanta anchor on dim 0 so pure numeric scale aligns cleanly
  for (let j1 = 0; j1 < meta.q1; j1++) {
    for (let j2 = 0; j2 < meta.q2; j2++) {
      let s = 0;
      for (let k = 0; k < meta.kTerms; k++) {
        s += meta.sigma * qA[k][i1 * meta.q1 + j1] * qB[k][i2 * meta.q2 + j2];
      }
      outSlice[outOffset + j1 * meta.q2 + j2] = s;
    }
  }
}

function serializeKronSpan(layer: OmniKronLayer, extraBias?: Float32Array): Uint8Array {
  const { meta, A, B } = layer;
  const headerBytes = 12 + meta.kTerms * 16 + (extraBias ? 8 : 0);
  let payloadBytes = extraBias ? Math.ceil(extraBias.length / 2) : 0;
  for (let k = 0; k < meta.kTerms; k++) {
    payloadBytes += Math.ceil(A[k].length / 2) + Math.ceil(B[k].length / 2);
  }
  const buf = new Uint8Array(headerBytes + payloadBytes);
  const view = new DataView(buf.buffer);
  view.setUint16(0, meta.rows, true);
  view.setUint16(2, meta.cols, true);
  buf[4] = meta.p1;
  buf[5] = meta.q1;
  buf[6] = meta.kTerms;
  buf[7] = extraBias ? 1 : 0;
  view.setFloat32(8, meta.sigma, true);

  let hdrCursor = 12;
  let payCursor = headerBytes;

  if (extraBias) {
    const bCodes = new Uint8Array(extraBias.length);
    const dummy = new Float32Array(extraBias.length);
    const bm = steQuantizeVec(extraBias, 4, dummy, bCodes);
    view.setFloat32(hdrCursor, bm.minV, true);
    view.setFloat32(hdrCursor + 4, bm.span, true);
    hdrCursor += 8;
    payCursor = pack4Bit(bCodes, buf, payCursor);
  }

  for (let k = 0; k < meta.kTerms; k++) {
    const cA = new Uint8Array(A[k].length);
    const cB = new Uint8Array(B[k].length);
    const dA = new Float32Array(A[k].length);
    const dB = new Float32Array(B[k].length);
    const mA = steQuantizeVec(A[k], 4, dA, cA);
    const mB = steQuantizeVec(B[k], 4, dB, cB);
    view.setFloat32(hdrCursor, mA.minV, true);
    view.setFloat32(hdrCursor + 4, mA.span, true);
    view.setFloat32(hdrCursor + 8, mB.minV, true);
    view.setFloat32(hdrCursor + 12, mB.span, true);
    hdrCursor += 16;
    payCursor = pack4Bit(cA, buf, payCursor);
    payCursor = pack4Bit(cB, buf, payCursor);
  }

  return buf;
}

function deserializeKronSpan(buf: Uint8Array): {
  meta: OmniKronMeta;
  qA: Float32Array[];
  qB: Float32Array[];
  bias: Float32Array | null;
} {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const rows = view.getUint16(0, true);
  const cols = view.getUint16(2, true);
  const p1 = buf[4];
  const q1 = buf[5];
  const kTerms = buf[6];
  const hasBias = buf[7] === 1;
  const sigma = view.getFloat32(8, true);
  const meta: OmniKronMeta = {
    rows,
    cols,
    p1,
    q1,
    p2: Math.ceil(rows / p1),
    q2: Math.ceil(cols / q1),
    kTerms,
    sigma,
  };

  let hdrCursor = 12;
  let payCursor = 12 + kTerms * 16 + (hasBias ? 8 : 0);

  let bias: Float32Array | null = null;
  if (hasBias) {
    const bMin = view.getFloat32(hdrCursor, true);
    const bSpan = view.getFloat32(hdrCursor + 4, true);
    hdrCursor += 8;
    const bCodes = unpack4Bit(buf, payCursor, cols);
    payCursor += Math.ceil(cols / 2);
    bias = new Float32Array(cols);
    for (let i = 0; i < cols; i++) bias[i] = bMin + (bCodes[i] / 15) * bSpan;
  }

  const lenA = meta.p1 * meta.q1;
  const lenB = meta.p2 * meta.q2;
  const qA: Float32Array[] = [];
  const qB: Float32Array[] = [];
  for (let k = 0; k < kTerms; k++) {
    const minA = view.getFloat32(hdrCursor, true);
    const spanA = view.getFloat32(hdrCursor + 4, true);
    const minB = view.getFloat32(hdrCursor + 8, true);
    const spanB = view.getFloat32(hdrCursor + 12, true);
    hdrCursor += 16;

    const cA = unpack4Bit(buf, payCursor, lenA);
    payCursor += Math.ceil(lenA / 2);
    const cB = unpack4Bit(buf, payCursor, lenB);
    payCursor += Math.ceil(lenB / 2);

    const fA = new Float32Array(lenA);
    const fB = new Float32Array(lenB);
    for (let i = 0; i < lenA; i++) fA[i] = minA + (cA[i] / 15) * spanA;
    for (let i = 0; i < lenB; i++) fB[i] = minB + (cB[i] / 15) * spanB;
    qA.push(fA);
    qB.push(fB);
  }

  return { meta, qA, qB, bias };
}

function serializeRouterSpan(routerW: Float32Array): Uint8Array {
  const codes = new Uint8Array(routerW.length);
  const dummy = new Float32Array(routerW.length);
  const { minV, span } = steQuantizeVec(routerW, 4, dummy, codes);
  const buf = new Uint8Array(12 + Math.ceil(codes.length / 2));
  const view = new DataView(buf.buffer);
  view.setUint16(0, OMNI_D_MODEL, true);
  view.setUint16(2, OMNI_NUM_EXPERTS, true);
  view.setFloat32(4, minV, true);
  view.setFloat32(8, span, true);
  pack4Bit(codes, buf, 12);
  return buf;
}

function deserializeRouterSpan(buf: Uint8Array): Float32Array {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const rows = view.getUint16(0, true);
  const cols = view.getUint16(2, true);
  const minV = view.getFloat32(4, true);
  const span = view.getFloat32(8, true);
  const count = rows * cols;
  const codes = unpack4Bit(buf, 12, count);
  const w = new Float32Array(count);
  for (let i = 0; i < count; i++) w[i] = minV + (codes[i] / 15) * span;
  return w;
}

export function getDefaultCheckpointPath(): string {
  return path.resolve(process.cwd(), 'checkpoints', 'omni_unified.x8d');
}

export class OmniUnifiedTrainer {
  private E: OmniKronLayer;
  private W1: OmniKronLayer;
  private routerW: Float32Array;
  private experts: OmniKronLayer[];
  private Wout: OmniKronLayer;
  private bout: Float32Array;

  private adamState = new Map<Float32Array, { g: Float32Array; m: Float32Array; v: Float32Array }>();
  private trainChunks: Partial<Record<OmniDomainId, Uint8Array>> = {};
  private heldOutChunks: Partial<Record<OmniDomainId, Uint8Array>> = {};
  private activeDomains: OmniDomainId[] = [];
  private rand: () => number;

  public metrics: OmniTrainingMetricsSnapshot;
  private startedAtMs = 0;
  private timer: NodeJS.Timeout | null = null;
  private checkpointPath: string;
  private dataDir: string;

  // Pre-allocated work buffers
  private xBuf = new Float32Array(OMNI_D_MODEL);
  private h1Pre = new Float32Array(OMNI_D_MODEL);
  private h1Act = new Float32Array(OMNI_D_MODEL);
  private z1 = new Float32Array(OMNI_D_MODEL);
  private rLogits = new Float32Array(OMNI_NUM_EXPERTS);
  private expOut = new Float32Array(OMNI_HIDDEN_EXP);
  private expAct = new Float32Array(OMNI_HIDDEN_EXP);
  private logits = new Float32Array(OMNI_VOCAB_SIZE);
  private gLogits = new Float32Array(OMNI_VOCAB_SIZE);
  private gExpAct = new Float32Array(OMNI_HIDDEN_EXP);
  private gExpPre = new Float32Array(OMNI_HIDDEN_EXP);
  private gZ1 = new Float32Array(OMNI_D_MODEL);
  private gH1Pre = new Float32Array(OMNI_D_MODEL);
  private gX = new Float32Array(OMNI_D_MODEL);
  private gXTmp = new Float32Array(OMNI_D_MODEL);
  private tmpT = new Float32Array(1024);
  private tmpAG = new Float32Array(1024);

  constructor(options?: { seed?: number; dataDir?: string; checkpointPath?: string }) {
    this.rand = mulberry32(options?.seed ?? 20261117);
    this.dataDir = options?.dataDir ?? getDefaultOmniDataDir();
    this.checkpointPath = options?.checkpointPath ?? getDefaultCheckpointPath();

    const makeKron = (rows: number, cols: number, p1: number, q1: number, kTerms: number, sigma: number): OmniKronLayer => {
      const p2 = Math.ceil(rows / p1);
      const q2 = Math.ceil(cols / q1);
      const A: Float32Array[] = [];
      const B: Float32Array[] = [];
      const sA = Math.sqrt(1 / (p1 * kTerms));
      const sB = Math.sqrt(1 / (p2 * kTerms));
      for (let k = 0; k < kTerms; k++) {
        const ak = new Float32Array(p1 * q1);
        const bk = new Float32Array(p2 * q2);
        for (let i = 0; i < ak.length; i++) ak[i] = (this.rand() - 0.5) * 2 * sA;
        for (let i = 0; i < bk.length; i++) bk[i] = (this.rand() - 0.5) * 2 * sB;
        A.push(ak);
        B.push(bk);
        this.registerAdam(ak);
        this.registerAdam(bk);
      }
      return { meta: { rows, cols, p1, q1, p2, q2, kTerms, sigma }, A, B };
    };

    this.E = makeKron(OMNI_VOCAB_SIZE, OMNI_EMB_DIM, 16, 8, 4, 0.45);
    this.W1 = makeKron(OMNI_D_MODEL, OMNI_D_MODEL, 16, 16, 4, 0.22);
    this.routerW = new Float32Array(OMNI_D_MODEL * OMNI_NUM_EXPERTS);
    for (let i = 0; i < this.routerW.length; i++) this.routerW[i] = (this.rand() - 0.5) * 0.15;
    this.registerAdam(this.routerW);

    this.experts = Array.from({ length: OMNI_NUM_EXPERTS }, () =>
      makeKron(OMNI_D_MODEL, OMNI_HIDDEN_EXP, 16, 16, 3, 0.24)
    );
    this.Wout = makeKron(OMNI_HIDDEN_EXP, OMNI_VOCAB_SIZE, 16, 16, 4, 0.24);
    this.bout = new Float32Array(OMNI_VOCAB_SIZE);
    this.registerAdam(this.bout);

    const effectiveWeights =
      OMNI_VOCAB_SIZE * OMNI_EMB_DIM +
      OMNI_D_MODEL * OMNI_D_MODEL +
      OMNI_D_MODEL * OMNI_NUM_EXPERTS +
      OMNI_NUM_EXPERTS * OMNI_D_MODEL * OMNI_HIDDEN_EXP +
      OMNI_HIDDEN_EXP * OMNI_VOCAB_SIZE +
      OMNI_VOCAB_SIZE;

    let trainableDofs = 0;
    for (const buf of this.adamState.keys()) trainableDofs += buf.length;

    this.metrics = {
      status: 'idle',
      step: 0,
      targetSteps: 800,
      epochsApprox: 0,
      bytesProcessed: 0,
      throughputBytesPerSec: 0,
      elapsedMs: 0,
      currentTrainLoss: 5.545,
      initialTrainLoss: 5.545,
      lossHistory: [],
      perDomainHeldOutPpl: {},
      expertUsagePct: [25, 25, 25, 25],
      effectiveWeights,
      trainableDofs,
      fp16EquivalentBytes: effectiveWeights * 2,
      checkpointPath: path.basename(this.checkpointPath),
      checkpointBytesOnDisk: 0,
      achievedBpw: 0,
      lastError: null,
    };

    this.refreshDatasetsFromDisk();
    this.loadSavedMetadataIfPresent();
  }

  private getMetaPath(): string {
    return `${this.checkpointPath}.meta.json`;
  }

  private getMasterPath(): string {
    return `${this.checkpointPath}.master.bin`;
  }

  private loadSavedMetadataIfPresent(): void {
    const metaPath = this.getMetaPath();
    if (!fs.existsSync(metaPath) || !fs.existsSync(this.checkpointPath)) return;
    try {
      const saved = JSON.parse(fs.readFileSync(metaPath, 'utf-8')) as Partial<OmniTrainingMetricsSnapshot>;
      if (typeof saved.step === 'number' && saved.step > 0) {
        if (!this.loadMasterStateIfPresent()) {
          this.loadWeightsFromX8dFile(this.checkpointPath);
        }
        this.metrics = {
          ...this.metrics,
          ...saved,
          status: saved.status === 'running' ? 'paused' : (saved.status ?? 'completed'),
          checkpointBytesOnDisk: fs.statSync(this.checkpointPath).size,
        };
      }
    } catch {
      // Ignore corrupt sidecar
    }
  }

  private saveMasterState(): void {
    try {
      const arrays: Float32Array[] = [];
      for (const [param, st] of this.adamState.entries()) {
        arrays.push(param, st.m, st.v);
      }
      const totalFloats = arrays.reduce((s, a) => s + a.length, 0);
      const out = new Float32Array(totalFloats);
      let cursor = 0;
      for (const a of arrays) {
        out.set(a, cursor);
        cursor += a.length;
      }
      fs.writeFileSync(this.getMasterPath(), Buffer.from(out.buffer));
    } catch {
      // Non-fatal
    }
  }

  private loadMasterStateIfPresent(): boolean {
    const mPath = this.getMasterPath();
    if (!fs.existsSync(mPath)) return false;
    try {
      const raw = fs.readFileSync(mPath);
      const floats = new Float32Array(raw.buffer, raw.byteOffset, Math.floor(raw.byteLength / 4));
      let expected = 0;
      for (const [param, st] of this.adamState.entries()) {
        expected += param.length + st.m.length + st.v.length;
      }
      if (floats.length !== expected) return false;
      let cursor = 0;
      for (const [param, st] of this.adamState.entries()) {
        param.set(floats.subarray(cursor, cursor + param.length));
        cursor += param.length;
        st.m.set(floats.subarray(cursor, cursor + st.m.length));
        cursor += st.m.length;
        st.v.set(floats.subarray(cursor, cursor + st.v.length));
        cursor += st.v.length;
      }
      return true;
    } catch {
      return false;
    }
  }

  private loadWeightsFromX8dFile(filePath: string): void {
    const stat = fs.statSync(filePath);
    const fd = fs.openSync(filePath, 'r');
    const readAt = (offset: number, length: number): Uint8Array => {
      const buf = Buffer.allocUnsafe(length);
      fs.readSync(fd, buf, 0, length, offset);
      return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    };
    try {
      const hdr = readAt(0, 16);
      const gv = new DataView(hdr.buffer, hdr.byteOffset, hdr.byteLength);
      const numSpans = gv.getUint32(0, true);
      if (numSpans !== 4 + OMNI_NUM_EXPERTS) return;
      const tail = readAt(stat.size - numSpans * 24, numSpans * 24);
      const tv = new DataView(tail.buffer, tail.byteOffset, tail.byteLength);
      const spans: Array<{ offset: number; length: number }> = [];
      for (let i = 0; i < numSpans; i++) {
        spans.push({ offset: tv.getUint32(i * 24 + 4, true), length: tv.getUint32(i * 24 + 8, true) });
      }
      const copyKron = (target: OmniKronLayer, idx: number, biasOut?: Float32Array) => {
        const sp = deserializeKronSpan(readAt(spans[idx].offset, spans[idx].length));
        for (let k = 0; k < target.A.length; k++) {
          target.A[k].set(sp.qA[k]);
          target.B[k].set(sp.qB[k]);
        }
        if (biasOut && sp.bias) biasOut.set(sp.bias);
      };
      copyKron(this.E, 0);
      copyKron(this.W1, 1);
      this.routerW.set(deserializeRouterSpan(readAt(spans[2].offset, spans[2].length)));
      for (let e = 0; e < OMNI_NUM_EXPERTS; e++) copyKron(this.experts[e], 3 + e);
      copyKron(this.Wout, numSpans - 1, this.bout);
    } finally {
      fs.closeSync(fd);
    }
  }

  private registerAdam(arr: Float32Array): void {
    this.adamState.set(arr, {
      g: new Float32Array(arr.length),
      m: new Float32Array(arr.length),
      v: new Float32Array(arr.length),
    });
  }

  public refreshDatasetsFromDisk(): boolean {
    const manifest = readOmniManifest(this.dataDir);
    if (!manifest) return false;
    this.trainChunks = {};
    this.heldOutChunks = {};
    this.activeDomains = [];
    for (const d of DOMAIN_ORDER) {
      const tr = loadDomainSplitBytes(d, 'train', this.dataDir);
      const ho = loadDomainSplitBytes(d, 'heldout', this.dataDir);
      if (tr.length > OMNI_CONTEXT_LEN + 2) {
        this.trainChunks[d] = tr;
        this.heldOutChunks[d] = ho.length > OMNI_CONTEXT_LEN + 2 ? ho : tr.slice(0, Math.min(2048, tr.length));
        this.activeDomains.push(d);
      }
    }
    return this.activeDomains.length > 0;
  }

  private quantizeSnapshot() {
    const qLayer = (l: OmniKronLayer) => ({
      qA: l.A.map((a) => {
        const o = new Float32Array(a.length);
        steQuantizeVec(a, 4, o);
        return o;
      }),
      qB: l.B.map((b) => {
        const o = new Float32Array(b.length);
        steQuantizeVec(b, 4, o);
        return o;
      }),
    });
    const qRouter = new Float32Array(this.routerW.length);
    steQuantizeVec(this.routerW, 4, qRouter);
    const qBout = new Float32Array(this.bout.length);
    steQuantizeVec(this.bout, 4, qBout);
    return {
      qE: qLayer(this.E),
      qW1: qLayer(this.W1),
      qRouter,
      qExp: this.experts.map(qLayer),
      qWout: qLayer(this.Wout),
      qBout,
    };
  }

  public saveCheckpointX8d(): number {
    const dir = path.dirname(this.checkpointPath);
    fs.mkdirSync(dir, { recursive: true });

    const spans: Uint8Array[] = [
      serializeKronSpan(this.E),
      serializeKronSpan(this.W1),
      serializeRouterSpan(this.routerW),
      ...this.experts.map((ex) => serializeKronSpan(ex)),
      serializeKronSpan(this.Wout, this.bout),
    ];

    // Magic-free container: 16-byte global header + raw spans + 24-byte tail directory per span (Issue #52)
    const globalHdr = new Uint8Array(16);
    const gv = new DataView(globalHdr.buffer);
    gv.setUint32(0, spans.length, true);
    gv.setUint32(4, this.metrics.step, true);
    gv.setUint32(8, this.metrics.effectiveWeights, true);
    gv.setUint32(12, this.metrics.trainableDofs, true);

    const tailDir = new Uint8Array(spans.length * 24);
    const tv = new DataView(tailDir.buffer);
    let offset = globalHdr.length;
    for (let i = 0; i < spans.length; i++) {
      tv.setUint32(i * 24, i, true);
      tv.setUint32(i * 24 + 4, offset, true);
      tv.setUint32(i * 24 + 8, spans[i].length, true);
      tv.setUint32(i * 24 + 12, 0, true);
      offset += spans[i].length;
    }

    const tmpPath = `${this.checkpointPath}.tmp`;
    const fd = fs.openSync(tmpPath, 'w');
    try {
      fs.writeSync(fd, globalHdr);
      for (const sp of spans) fs.writeSync(fd, sp);
      fs.writeSync(fd, tailDir);
    } finally {
      fs.closeSync(fd);
    }
    fs.renameSync(tmpPath, this.checkpointPath);

    const stat = fs.statSync(this.checkpointPath);
    this.metrics.checkpointBytesOnDisk = stat.size;
    this.metrics.achievedBpw = Number(((stat.size * 8) / this.metrics.effectiveWeights).toFixed(4));
    this.saveMasterState();
    try {
      fs.writeFileSync(this.getMetaPath(), JSON.stringify(this.metrics, null, 2));
    } catch {
      // Non-fatal metadata write
    }
    return stat.size;
  }

  public evaluateHeldOutAllDomains(maxBytesPerDomain = 512): void {
    const snap = this.quantizeSnapshot();
    const probs = new Float32Array(OMNI_VOCAB_SIZE);
    let pplSum = 0;
    let pplCount = 0;

    for (const d of this.activeDomains) {
      const ho = this.heldOutChunks[d];
      if (!ho || ho.length <= OMNI_CONTEXT_LEN + 2) continue;
      const limit = Math.min(ho.length, OMNI_CONTEXT_LEN + maxBytesPerDomain);
      let nll = 0;
      let hits = 0;
      let steps = 0;

      for (let t = OMNI_CONTEXT_LEN; t < limit; t++) {
        for (let s = 0; s < OMNI_CONTEXT_LEN; s++) {
          evalEmbedSlot(snap.qE.qA, snap.qE.qB, this.E.meta, ho[t - OMNI_CONTEXT_LEN + s], this.xBuf, s * OMNI_EMB_DIM);
        }
        kronForward(this.W1.meta, snap.qW1.qA, snap.qW1.qB, this.xBuf, this.h1Pre, this.tmpT);
        for (let c = 0; c < OMNI_D_MODEL; c++) {
          this.z1[c] = this.xBuf[c] + 0.5 * Math.tanh(this.h1Pre[c]);
        }

        let bestExp = 0;
        let bestScore = -Infinity;
        for (let e = 0; e < OMNI_NUM_EXPERTS; e++) {
          let s = 0;
          for (let c = 0; c < OMNI_D_MODEL; c++) s += this.z1[c] * snap.qRouter[c * OMNI_NUM_EXPERTS + e];
          if (s > bestScore) {
            bestScore = s;
            bestExp = e;
          }
        }

        kronForward(
          this.experts[bestExp].meta,
          snap.qExp[bestExp].qA,
          snap.qExp[bestExp].qB,
          this.z1,
          this.expOut,
          this.tmpT
        );
        for (let c = 0; c < OMNI_HIDDEN_EXP; c++) this.expAct[c] = Math.tanh(this.expOut[c]);
        kronForward(this.Wout.meta, snap.qWout.qA, snap.qWout.qB, this.expAct, this.logits, this.tmpT);
        for (let v = 0; v < OMNI_VOCAB_SIZE; v++) this.logits[v] += snap.qBout[v];

        let maxL = -Infinity;
        let argmax = 0;
        for (let v = 0; v < OMNI_VOCAB_SIZE; v++) {
          if (this.logits[v] > maxL) {
            maxL = this.logits[v];
            argmax = v;
          }
        }
        let sumE = 0;
        for (let v = 0; v < OMNI_VOCAB_SIZE; v++) {
          const e = Math.exp(this.logits[v] - maxL);
          probs[v] = e;
          sumE += e;
        }
        const pTarg = Math.max(1e-7, probs[ho[t]] / sumE);
        nll += -Math.log(pTarg);
        if (argmax === ho[t]) hits++;
        steps++;
      }

      const ppl = Number(Math.exp(nll / Math.max(1, steps)).toFixed(2));
      const acc = Number(((hits / Math.max(1, steps)) * 100).toFixed(1));
      const prevInit = this.metrics.perDomainHeldOutPpl[d]?.initPpl ?? ppl;
      this.metrics.perDomainHeldOutPpl[d] = {
        initPpl: prevInit,
        currentPpl: ppl,
        top1AccPct: acc,
      };
      pplSum += ppl;
      pplCount++;
    }

    if (pplCount > 0) {
      const meanPpl = Number((pplSum / pplCount).toFixed(2));
      this.metrics.lossHistory.push({
        step: this.metrics.step,
        trainLoss: this.metrics.currentTrainLoss,
        meanHeldOutPpl: meanPpl,
      });
      if (this.metrics.lossHistory.length > 60) {
        this.metrics.lossHistory.shift();
      }
    }
  }

  public runStepBatch(batchWindows = 24, lr = 0.045): number {
    if (this.activeDomains.length === 0) {
      if (!this.refreshDatasetsFromDisk()) {
        throw new Error('No omni dataset splits found on disk. Run dataset ingestion first.');
      }
    }

    for (const st of this.adamState.values()) st.g.fill(0);
    const snap = this.quantizeSnapshot();
    const expCounts = [0, 0, 0, 0];
    let batchLoss = 0;
    const invB = 1 / batchWindows;

    for (let b = 0; b < batchWindows; b++) {
      const domId = this.activeDomains[(this.metrics.step * batchWindows + b) % this.activeDomains.length];
      const buf = this.trainChunks[domId]!;
      const tPos = OMNI_CONTEXT_LEN + Math.floor(this.rand() * (buf.length - OMNI_CONTEXT_LEN - 1));
      const targetByte = buf[tPos];

      for (let s = 0; s < OMNI_CONTEXT_LEN; s++) {
        evalEmbedSlot(snap.qE.qA, snap.qE.qB, this.E.meta, buf[tPos - OMNI_CONTEXT_LEN + s], this.xBuf, s * OMNI_EMB_DIM);
      }

      kronForward(this.W1.meta, snap.qW1.qA, snap.qW1.qB, this.xBuf, this.h1Pre, this.tmpT);
      for (let c = 0; c < OMNI_D_MODEL; c++) {
        const th = Math.tanh(this.h1Pre[c]);
        this.h1Act[c] = th;
        this.z1[c] = this.xBuf[c] + 0.5 * th;
      }

      // Softmax router -> top-1 active expert
      let maxR = -Infinity;
      let chosenExp = 0;
      for (let e = 0; e < OMNI_NUM_EXPERTS; e++) {
        let s = 0;
        for (let c = 0; c < OMNI_D_MODEL; c++) s += this.z1[c] * snap.qRouter[c * OMNI_NUM_EXPERTS + e];
        this.rLogits[e] = s;
        if (s > maxR) {
          maxR = s;
          chosenExp = e;
        }
      }
      expCounts[chosenExp]++;

      kronForward(
        this.experts[chosenExp].meta,
        snap.qExp[chosenExp].qA,
        snap.qExp[chosenExp].qB,
        this.z1,
        this.expOut,
        this.tmpT
      );
      for (let c = 0; c < OMNI_HIDDEN_EXP; c++) this.expAct[c] = Math.tanh(this.expOut[c]);
      kronForward(this.Wout.meta, snap.qWout.qA, snap.qWout.qB, this.expAct, this.logits, this.tmpT);
      for (let v = 0; v < OMNI_VOCAB_SIZE; v++) this.logits[v] += snap.qBout[v];

      let maxL = -Infinity;
      for (let v = 0; v < OMNI_VOCAB_SIZE; v++) if (this.logits[v] > maxL) maxL = this.logits[v];
      let sumE = 0;
      for (let v = 0; v < OMNI_VOCAB_SIZE; v++) {
        const e = Math.exp(this.logits[v] - maxL);
        this.gLogits[v] = e;
        sumE += e;
      }
      const invSum = 1 / sumE;
      for (let v = 0; v < OMNI_VOCAB_SIZE; v++) this.gLogits[v] *= invSum;
      const pTarget = Math.max(1e-7, this.gLogits[targetByte]);
      batchLoss += -Math.log(pTarget);
      this.gLogits[targetByte] -= 1;

      const gBout = this.adamState.get(this.bout)!.g;
      for (let v = 0; v < OMNI_VOCAB_SIZE; v++) {
        this.gLogits[v] *= invB;
        gBout[v] += this.gLogits[v];
      }

      kronBackward(
        this.Wout.meta,
        snap.qWout.qA,
        snap.qWout.qB,
        this.expAct,
        this.gLogits,
        this.Wout.A.map((a) => this.adamState.get(a)!.g),
        this.Wout.B.map((a) => this.adamState.get(a)!.g),
        this.gExpAct,
        this.tmpT,
        this.tmpAG
      );

      for (let c = 0; c < OMNI_HIDDEN_EXP; c++) {
        this.gExpPre[c] = this.gExpAct[c] * (1 - this.expAct[c] * this.expAct[c]);
      }

      kronBackward(
        this.experts[chosenExp].meta,
        snap.qExp[chosenExp].qA,
        snap.qExp[chosenExp].qB,
        this.z1,
        this.gExpPre,
        this.experts[chosenExp].A.map((a) => this.adamState.get(a)!.g),
        this.experts[chosenExp].B.map((a) => this.adamState.get(a)!.g),
        this.gZ1,
        this.tmpT,
        this.tmpAG
      );

      // Gentle load-balancing gradient on router logits so all 4 experts stay utilized
      const gRouter = this.adamState.get(this.routerW)!.g;
      let rSumE = 0;
      const rProbs = new Float32Array(OMNI_NUM_EXPERTS);
      for (let e = 0; e < OMNI_NUM_EXPERTS; e++) {
        rProbs[e] = Math.exp(this.rLogits[e] - maxR);
        rSumE += rProbs[e];
      }
      const targetDomExp = DOMAIN_ORDER.indexOf(domId) % OMNI_NUM_EXPERTS;
      for (let e = 0; e < OMNI_NUM_EXPERTS; e++) {
        const pE = rProbs[e] / rSumE;
        const gRE = (pE - (e === targetDomExp ? 1 : 0)) * 0.08 * invB;
        for (let c = 0; c < OMNI_D_MODEL; c++) {
          gRouter[c * OMNI_NUM_EXPERTS + e] += this.z1[c] * gRE;
        }
      }

      for (let c = 0; c < OMNI_D_MODEL; c++) {
        this.gH1Pre[c] = this.gZ1[c] * 0.5 * (1 - this.h1Act[c] * this.h1Act[c]);
      }

      kronBackward(
        this.W1.meta,
        snap.qW1.qA,
        snap.qW1.qB,
        this.xBuf,
        this.gH1Pre,
        this.W1.A.map((a) => this.adamState.get(a)!.g),
        this.W1.B.map((a) => this.adamState.get(a)!.g),
        this.gXTmp,
        this.tmpT,
        this.tmpAG
      );

      for (let c = 0; c < OMNI_D_MODEL; c++) this.gX[c] = this.gZ1[c] + this.gXTmp[c];

      const { p2, q1, q2, sigma, kTerms } = this.E.meta;
      for (let s = 0; s < OMNI_CONTEXT_LEN; s++) {
        const byteId = buf[tPos - OMNI_CONTEXT_LEN + s];
        const i1 = Math.floor(byteId / p2);
        const i2 = byteId % p2;
        const xOff = s * OMNI_EMB_DIM;
        for (let k = 0; k < kTerms; k++) {
          const ak = snap.qE.qA[k];
          const bk = snap.qE.qB[k];
          const gak = this.adamState.get(this.E.A[k])!.g;
          const gbk = this.adamState.get(this.E.B[k])!.g;
          for (let j1 = 0; j1 < q1; j1++) {
            for (let j2 = 0; j2 < q2; j2++) {
              const g = this.gX[xOff + j1 * q2 + j2] * sigma;
              gak[i1 * q1 + j1] += g * bk[i2 * q2 + j2];
              gbk[i2 * q2 + j2] += g * ak[i1 * q1 + j1];
            }
          }
        }
      }
    }

    this.metrics.step++;
    this.metrics.bytesProcessed += batchWindows;
    const avgLoss = Number((batchLoss * invB).toFixed(4));
    if (this.metrics.step === 1) this.metrics.initialTrainLoss = avgLoss;
    this.metrics.currentTrainLoss = avgLoss;
    this.metrics.expertUsagePct = expCounts.map((c) => Math.round((c / batchWindows) * 100)) as [
      number,
      number,
      number,
      number,
    ];

    const stepT = this.metrics.step;
    const b1C = 1 - Math.pow(0.9, stepT);
    const b2C = 1 - Math.pow(0.99, stepT);
    for (const [param, st] of this.adamState.entries()) {
      for (let i = 0; i < param.length; i++) {
        const g = Math.max(-4, Math.min(4, st.g[i]));
        st.m[i] = 0.9 * st.m[i] + 0.1 * g;
        st.v[i] = 0.99 * st.v[i] + 0.01 * g * g;
        const mh = st.m[i] / b1C;
        const vh = st.v[i] / b2C;
        param[i] -= (lr * mh) / (Math.sqrt(vh) + 1e-7);
      }
    }

    const totalCorpusBytes = Math.max(
      1,
      this.activeDomains.reduce((s, d) => s + (this.trainChunks[d]?.length ?? 0), 0)
    );
    this.metrics.epochsApprox = Number((this.metrics.bytesProcessed / totalCorpusBytes).toFixed(3));
    if (this.startedAtMs > 0) {
      const elapsed = Math.max(1, Date.now() - this.startedAtMs);
      this.metrics.elapsedMs = elapsed;
      this.metrics.throughputBytesPerSec = Math.round((this.metrics.bytesProcessed * 1000) / elapsed);
    }
    return avgLoss;
  }

  public trainSynchronously(totalSteps = 400, batchWindows = 32): OmniTrainingMetricsSnapshot {
    if (!this.refreshDatasetsFromDisk()) {
      throw new Error('Omni datasets not found on disk. Run fetchAllOmniDatasets first.');
    }
    this.startedAtMs = Date.now();
    this.metrics.status = 'running';
    this.metrics.targetSteps = totalSteps;
    this.evaluateHeldOutAllDomains(180);

    for (let s = 1; s <= totalSteps; s++) {
      this.runStepBatch(batchWindows, 0.055);
      if (s % 80 === 0 || s === totalSteps) {
        this.evaluateHeldOutAllDomains(180);
        this.saveCheckpointX8d();
      }
    }
    this.metrics.status = 'completed';
    this.saveCheckpointX8d();
    return this.metrics;
  }

  public startBackgroundTraining(targetSteps = 1000, batchWindows = 32): void {
    if (this.timer) return;
    if (!this.refreshDatasetsFromDisk()) {
      this.metrics.status = 'error';
      this.metrics.lastError = 'No datasets found on disk. Click Fetch / Expand HF Omni Datasets first.';
      return;
    }
    this.startedAtMs = Date.now() - this.metrics.elapsedMs;
    if (this.metrics.step === 0) {
      this.evaluateHeldOutAllDomains(160);
    }
    this.metrics.status = 'running';
    this.metrics.targetSteps = Math.max(targetSteps, this.metrics.step + 240);
    this.metrics.lastError = null;

    const tick = () => {
      if (this.metrics.status !== 'running') return;
      try {
        const chunk = 8;
        for (let i = 0; i < chunk && this.metrics.step < this.metrics.targetSteps; i++) {
          this.runStepBatch(batchWindows, 0.055);
        }
        if (this.metrics.step % 64 === 0 || this.metrics.step >= this.metrics.targetSteps) {
          this.evaluateHeldOutAllDomains(160);
          this.saveCheckpointX8d();
        }
        if (this.metrics.step >= this.metrics.targetSteps) {
          this.metrics.status = 'completed';
          this.saveCheckpointX8d();
          this.stopBackgroundTraining(true);
          return;
        }
        this.timer = setTimeout(tick, 10);
      } catch (err: unknown) {
        this.metrics.status = 'error';
        this.metrics.lastError = err instanceof Error ? err.message : 'Training error';
        this.stopBackgroundTraining(false);
      }
    };

    this.timer = setTimeout(tick, 5);
  }

  public stopBackgroundTraining(markCompleted = false): void {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    if (this.metrics.step > 0) {
      this.saveCheckpointX8d();
    }
    if (!markCompleted && this.metrics.status === 'running') {
      this.metrics.status = 'paused';
    }
  }
}

/**
 * STRICT DIRECT-SEEK DISK RUNTIME (Zero mmap, zero JSON weights, zero training state):
 * Opens checkpointPath with fs.openSync('r'), seeks tail directory, seeks shared tensors + ONLY
 * the routed top-1 Expert span per generation step via fs.readSync.
 */
export function inferFromDiskX8dOnly(
  prompt: string,
  genBytesLen = 48,
  checkpointPath = getDefaultCheckpointPath()
): OmniDiskInferResult {
  if (!fs.existsSync(checkpointPath)) {
    throw new Error('No .x8D checkpoint on disk yet. Run Omni training first.');
  }
  const t0 = Date.now();
  const stat = fs.statSync(checkpointPath);
  const fd = fs.openSync(checkpointPath, 'r');
  let diskBytesSeekRead = 0;
  let diskSeeksPerformed = 0;

  const readSpanAt = (offset: number, length: number): Uint8Array => {
    const buf = Buffer.allocUnsafe(length);
    fs.readSync(fd, buf, 0, length, offset);
    diskBytesSeekRead += length;
    diskSeeksPerformed++;
    return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  };

  try {
    const globalHdr = readSpanAt(0, 16);
    const gv = new DataView(globalHdr.buffer, globalHdr.byteOffset, globalHdr.byteLength);
    const numSpans = gv.getUint32(0, true);
    const tailOffset = stat.size - numSpans * 24;
    const tailBuf = readSpanAt(tailOffset, numSpans * 24);
    const tv = new DataView(tailBuf.buffer, tailBuf.byteOffset, tailBuf.byteLength);

    const spanDir: Array<{ offset: number; length: number }> = [];
    for (let i = 0; i < numSpans; i++) {
      spanDir.push({
        offset: tv.getUint32(i * 24 + 4, true),
        length: tv.getUint32(i * 24 + 8, true),
      });
    }

    // Load ONLY shared spans [E, W1, Router, Wout] initially; Expert spans [3..6] loaded lazily via f.seek/f.read!
    const E = deserializeKronSpan(readSpanAt(spanDir[0].offset, spanDir[0].length));
    const W1 = deserializeKronSpan(readSpanAt(spanDir[1].offset, spanDir[1].length));
    const routerW = deserializeRouterSpan(readSpanAt(spanDir[2].offset, spanDir[2].length));
    const Wout = deserializeKronSpan(readSpanAt(spanDir[numSpans - 1].offset, spanDir[numSpans - 1].length));

    const expertCache = new Map<number, ReturnType<typeof deserializeKronSpan>>();
    const activeExpertsUsed = new Set<number>();

    const seed = Array.from(new TextEncoder().encode(prompt));
    const stream = [...seed];
    const xBuf = new Float32Array(OMNI_D_MODEL);
    const h1Pre = new Float32Array(OMNI_D_MODEL);
    const z1 = new Float32Array(OMNI_D_MODEL);
    const expOut = new Float32Array(OMNI_HIDDEN_EXP);
    const expAct = new Float32Array(OMNI_HIDDEN_EXP);
    const logits = new Float32Array(OMNI_VOCAB_SIZE);
    const tmpT = new Float32Array(1024);

    for (let g = 0; g < genBytesLen; g++) {
      const n = stream.length;
      for (let s = 0; s < OMNI_CONTEXT_LEN; s++) {
        const b = stream[n - OMNI_CONTEXT_LEN + s] ?? 32;
        evalEmbedSlot(E.qA, E.qB, E.meta, b, xBuf, s * OMNI_EMB_DIM);
      }
      kronForward(W1.meta, W1.qA, W1.qB, xBuf, h1Pre, tmpT);
      for (let c = 0; c < OMNI_D_MODEL; c++) z1[c] = xBuf[c] + 0.5 * Math.tanh(h1Pre[c]);

      let bestExp = 0;
      let bestScore = -Infinity;
      for (let e = 0; e < OMNI_NUM_EXPERTS; e++) {
        let s = 0;
        for (let c = 0; c < OMNI_D_MODEL; c++) s += z1[c] * routerW[c * OMNI_NUM_EXPERTS + e];
        if (s > bestScore) {
          bestScore = s;
          bestExp = e;
        }
      }
      activeExpertsUsed.add(bestExp);

      let exp = expertCache.get(bestExp);
      if (!exp) {
        const sp = spanDir[3 + bestExp];
        exp = deserializeKronSpan(readSpanAt(sp.offset, sp.length));
        expertCache.set(bestExp, exp);
      }

      kronForward(exp.meta, exp.qA, exp.qB, z1, expOut, tmpT);
      for (let c = 0; c < OMNI_HIDDEN_EXP; c++) expAct[c] = Math.tanh(expOut[c]);
      kronForward(Wout.meta, Wout.qA, Wout.qB, expAct, logits, tmpT);
      if (Wout.bias) for (let v = 0; v < OMNI_VOCAB_SIZE; v++) logits[v] += Wout.bias[v];

      let bestByte = 0;
      for (let v = 1; v < OMNI_VOCAB_SIZE; v++) if (logits[v] > logits[bestByte]) bestByte = v;
      stream.push(bestByte);
    }

    const genBytes = new Uint8Array(stream.slice(seed.length));
    const continuation = new TextDecoder().decode(genBytes);
    const outputQuantaSample = Array.from(genBytes.slice(0, 10)).map((b) => (b * 0.001).toFixed(3));

    return {
      prompt,
      continuation,
      outputQuantaSample,
      activeExpertsUsed: Array.from(activeExpertsUsed),
      diskFileSizeBytes: stat.size,
      diskBytesSeekRead,
      diskSeeksPerformed,
      latencyMs: Math.max(1, Date.now() - t0),
    };
  } finally {
    fs.closeSync(fd);
  }
}
