/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * Unified 5.00M Effective-Weight (24,320 Compact DOFs) 256-State Byte-Diffusion + NCM Trainer
 * Consumes unified Safetensors-equivalent .x8D v2 container (x8dContainerFormat.ts):
 * - Domain: strictly 256 byte states (0..255 <-> 0.000..0.255), zero BPE, zero MASK=256, zero -1 ternary.
 * - Architecture: Kronecker E + Trunk + Trainable 4-Expert System-1/System-2 Router + Kronecker Head.
 * - Objective: Real 256-state uniform-byte-corruption denoising (Issue #54) + NCM multi-pass refinement (#64).
 * - Storage: Shared layers encoded at 3-bit STE + x8d_arith (Codec 0x04), Experts at 2-bit STE + x8d_arith,
 *   indexed in a magic-free 48-byte named tensor directory with per-span CRC32 checksums.
 * - Inference: Reads strictly from .x8D v2 via fs.openSync + fs.readSync (zero mmap, zero master weights).
 */

import fs from 'fs';
import path from 'path';
import {
  OmniDomainId,
  getDefaultOmniDataDir,
  loadDomainSplitBytes,
  readOmniManifest,
} from './omniDatasetIngestor';
import {
  X8dCodecId,
  X8dDtypeId,
  X8dSeekStats,
  X8dTensorDirEntry,
  readDirectoryWithFd,
  readNamedTensorWithFd,
  writeX8dV2Container,
  x8dArithCompressBytes,
  x8dArithDecompressBytes,
} from './x8dContainerFormat';

export const OMNI_CONTEXT_LEN = 8;
export const OMNI_EMB_DIM = 64;
export const OMNI_D_MODEL = OMNI_CONTEXT_LEN * OMNI_EMB_DIM; // 512
export const OMNI_HIDDEN_EXP = 2048;
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
  qBits: number;
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
  achievedDofBpb: number;
  diffusionObjective: string;
  lastError: string | null;
}

export interface OmniDiskInferResult {
  prompt: string;
  continuation: string;
  outputQuantaSample: string[];
  activeExpertsUsed: number[];
  system1EarlyExits: number;
  system2Refinements: number;
  crcVerifiedCount: number;
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

function packNBits(codes: Uint8Array, qBits: number): Uint8Array {
  const totalBits = codes.length * qBits;
  const out = new Uint8Array(Math.ceil(totalBits / 8));
  let bitPos = 0;
  const mask = (1 << qBits) - 1;
  for (let i = 0; i < codes.length; i++) {
    const val = codes[i] & mask;
    for (let b = 0; b < qBits; b++) {
      if ((val & (1 << b)) !== 0) {
        out[bitPos >> 3] |= 1 << (bitPos & 7);
      }
      bitPos++;
    }
  }
  return out;
}

function unpackNBits(packed: Uint8Array, offset: number, count: number, qBits: number): { codes: Uint8Array; bytesRead: number } {
  const codes = new Uint8Array(count);
  const bytesRead = Math.ceil((count * qBits) / 8);
  let bitPos = 0;
  for (let i = 0; i < count; i++) {
    let val = 0;
    for (let b = 0; b < qBits; b++) {
      if ((packed[offset + (bitPos >> 3)] & (1 << (bitPos & 7))) !== 0) {
        val |= 1 << b;
      }
      bitPos++;
    }
    codes[i] = val;
  }
  return { codes, bytesRead };
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

function serializeKronArithPayload(layer: OmniKronLayer, extraBias?: Float32Array): Uint8Array {
  const { meta, A, B } = layer;
  const qBits = meta.qBits;
  const hdrLen = 16 + meta.kTerms * 16 + (extraBias ? 8 : 0);
  const chunks: Uint8Array[] = [];

  const hdr = new Uint8Array(hdrLen);
  const view = new DataView(hdr.buffer);
  view.setUint16(0, meta.rows, true);
  view.setUint16(2, meta.cols, true);
  hdr[4] = meta.p1;
  hdr[5] = meta.q1;
  hdr[6] = meta.kTerms;
  hdr[7] = qBits;
  hdr[8] = extraBias ? 1 : 0;
  view.setFloat32(12, meta.sigma, true);

  let hdrCursor = 16;
  if (extraBias) {
    const bCodes = new Uint8Array(extraBias.length);
    const dummy = new Float32Array(extraBias.length);
    const bm = steQuantizeVec(extraBias, 4, dummy, bCodes);
    view.setFloat32(hdrCursor, bm.minV, true);
    view.setFloat32(hdrCursor + 4, bm.span, true);
    hdrCursor += 8;
    chunks.push(packNBits(bCodes, 4));
  }

  for (let k = 0; k < meta.kTerms; k++) {
    const cA = new Uint8Array(A[k].length);
    const cB = new Uint8Array(B[k].length);
    const dA = new Float32Array(A[k].length);
    const dB = new Float32Array(B[k].length);
    const mA = steQuantizeVec(A[k], qBits, dA, cA);
    const mB = steQuantizeVec(B[k], qBits, dB, cB);
    view.setFloat32(hdrCursor, mA.minV, true);
    view.setFloat32(hdrCursor + 4, mA.span, true);
    view.setFloat32(hdrCursor + 8, mB.minV, true);
    view.setFloat32(hdrCursor + 12, mB.span, true);
    hdrCursor += 16;
    chunks.push(packNBits(cA, qBits), packNBits(cB, qBits));
  }

  const totalBody = chunks.reduce((s, c) => s + c.length, 0);
  const body = new Uint8Array(totalBody);
  let pos = 0;
  for (const c of chunks) {
    body.set(c, pos);
    pos += c.length;
  }
  const arithBody = x8dArithCompressBytes(body);
  const out = new Uint8Array(hdr.length + arithBody.length);
  out.set(hdr, 0);
  out.set(arithBody, hdr.length);
  return out;
}

function deserializeKronArithPayload(buf: Uint8Array): {
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
  const qBits = buf[7];
  const hasBias = buf[8] === 1;
  const sigma = view.getFloat32(12, true);
  const meta: OmniKronMeta = {
    rows,
    cols,
    p1,
    q1,
    p2: Math.ceil(rows / p1),
    q2: Math.ceil(cols / q1),
    kTerms,
    sigma,
    qBits,
  };

  let hdrCursor = 16;
  const hdrLen = 16 + kTerms * 16 + (hasBias ? 8 : 0);
  const body = x8dArithDecompressBytes(buf.subarray(hdrLen));
  let payCursor = 0;

  let bias: Float32Array | null = null;
  if (hasBias) {
    const bMin = view.getFloat32(hdrCursor, true);
    const bSpan = view.getFloat32(hdrCursor + 4, true);
    hdrCursor += 8;
    const { codes, bytesRead } = unpackNBits(body, payCursor, cols, 4);
    payCursor += bytesRead;
    bias = new Float32Array(cols);
    for (let i = 0; i < cols; i++) bias[i] = bMin + (codes[i] / 15) * bSpan;
  }

  const lenA = meta.p1 * meta.q1;
  const lenB = meta.p2 * meta.q2;
  const qMax = (1 << qBits) - 1;
  const qA: Float32Array[] = [];
  const qB: Float32Array[] = [];

  for (let k = 0; k < kTerms; k++) {
    const minA = view.getFloat32(hdrCursor, true);
    const spanA = view.getFloat32(hdrCursor + 4, true);
    const minB = view.getFloat32(hdrCursor + 8, true);
    const spanB = view.getFloat32(hdrCursor + 12, true);
    hdrCursor += 16;

    const uA = unpackNBits(body, payCursor, lenA, qBits);
    payCursor += uA.bytesRead;
    const uB = unpackNBits(body, payCursor, lenB, qBits);
    payCursor += uB.bytesRead;

    const fA = new Float32Array(lenA);
    const fB = new Float32Array(lenB);
    for (let i = 0; i < lenA; i++) fA[i] = minA + (uA.codes[i] / qMax) * spanA;
    for (let i = 0; i < lenB; i++) fB[i] = minB + (uB.codes[i] / qMax) * spanB;
    qA.push(fA);
    qB.push(fB);
  }

  return { meta, qA, qB, bias };
}

function serializeRouterArithPayload(routerW: Float32Array): Uint8Array {
  const codes = new Uint8Array(routerW.length);
  const dummy = new Float32Array(routerW.length);
  const { minV, span } = steQuantizeVec(routerW, 3, dummy, codes);
  const packed = x8dArithCompressBytes(packNBits(codes, 3));
  const buf = new Uint8Array(12 + packed.length);
  const view = new DataView(buf.buffer);
  view.setUint16(0, OMNI_D_MODEL, true);
  view.setUint16(2, OMNI_NUM_EXPERTS, true);
  view.setFloat32(4, minV, true);
  view.setFloat32(8, span, true);
  buf.set(packed, 12);
  return buf;
}

function deserializeRouterArithPayload(buf: Uint8Array): Float32Array {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const rows = view.getUint16(0, true);
  const cols = view.getUint16(2, true);
  const minV = view.getFloat32(4, true);
  const span = view.getFloat32(8, true);
  const count = rows * cols;
  const raw = x8dArithDecompressBytes(buf.subarray(12));
  const { codes } = unpackNBits(raw, 0, count, 3);
  const w = new Float32Array(count);
  for (let i = 0; i < count; i++) w[i] = minV + (codes[i] / 7) * span;
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

  // Pre-allocated CPU tile work buffers (sized for d_model=512, hidden_exp=2048, max p1*q2 = 2048)
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
  private tmpT = new Float32Array(2048);
  private tmpAG = new Float32Array(2048);

  constructor(options?: { seed?: number; dataDir?: string; checkpointPath?: string }) {
    this.rand = mulberry32(options?.seed ?? 20261121);
    this.dataDir = options?.dataDir ?? getDefaultOmniDataDir();
    this.checkpointPath = options?.checkpointPath ?? getDefaultCheckpointPath();

    const makeKron = (
      rows: number,
      cols: number,
      p1: number,
      q1: number,
      kTerms: number,
      sigma: number,
      qBits: number
    ): OmniKronLayer => {
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
      return { meta: { rows, cols, p1, q1, p2, q2, kTerms, sigma, qBits }, A, B };
    };

    // 5.00M effective weights | 24,320 compact DOFs -> ~0.008–0.011 bpw real serialized .x8D v2 file
    this.E = makeKron(OMNI_VOCAB_SIZE, OMNI_EMB_DIM, 16, 8, 2, 0.45, 3);
    this.W1 = makeKron(OMNI_D_MODEL, OMNI_D_MODEL, 32, 16, 2, 0.22, 3);
    this.routerW = new Float32Array(OMNI_D_MODEL * OMNI_NUM_EXPERTS);
    for (let i = 0; i < this.routerW.length; i++) this.routerW[i] = (this.rand() - 0.5) * 0.14;
    this.registerAdam(this.routerW);

    this.experts = Array.from({ length: OMNI_NUM_EXPERTS }, () =>
      makeKron(OMNI_D_MODEL, OMNI_HIDDEN_EXP, 32, 32, 2, 0.24, 2)
    );
    this.Wout = makeKron(OMNI_HIDDEN_EXP, OMNI_VOCAB_SIZE, 32, 16, 2, 0.24, 3);
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
      achievedDofBpb: 0,
      diffusionObjective: '256-State Uniform Corruption + NCM S1/S2 Denoising',
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
      if (typeof saved.step === 'number' && saved.step > 0 && saved.effectiveWeights === this.metrics.effectiveWeights) {
        if (!this.loadMasterStateIfPresent()) {
          this.loadWeightsFromX8dV2File(this.checkpointPath);
        }
        this.metrics = {
          ...this.metrics,
          ...saved,
          status: saved.status === 'running' ? 'paused' : (saved.status ?? 'completed'),
          checkpointBytesOnDisk: fs.statSync(this.checkpointPath).size,
        };
      }
    } catch {
      // Ignore incompatible/old v1 sidecar cleanly so v2 initializes cleanly
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

  private loadWeightsFromX8dV2File(filePath: string): void {
    const stat = fs.statSync(filePath);
    const fd = fs.openSync(filePath, 'r');
    try {
      const dir = readDirectoryWithFd(fd, stat.size);
      const byName = new Map(dir.entries.map((e) => [e.name, e]));
      const copyKron = (target: OmniKronLayer, name: string, biasOut?: Float32Array) => {
        const entry = byName.get(name);
        if (!entry) return;
        const sp = deserializeKronArithPayload(readNamedTensorWithFd(fd, entry));
        for (let k = 0; k < target.A.length; k++) {
          target.A[k].set(sp.qA[k]);
          target.B[k].set(sp.qB[k]);
        }
        if (biasOut && sp.bias) biasOut.set(sp.bias);
      };
      copyKron(this.E, 'embed.E');
      copyKron(this.W1, 'trunk.W1');
      const rEntry = byName.get('router.W');
      if (rEntry) this.routerW.set(deserializeRouterArithPayload(readNamedTensorWithFd(fd, rEntry)));
      for (let e = 0; e < OMNI_NUM_EXPERTS; e++) copyKron(this.experts[e], `expert.${e}`);
      copyKron(this.Wout, 'head.Wout', this.bout);
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
        steQuantizeVec(a, l.meta.qBits, o);
        return o;
      }),
      qB: l.B.map((b) => {
        const o = new Float32Array(b.length);
        steQuantizeVec(b, l.meta.qBits, o);
        return o;
      }),
    });
    const qRouter = new Float32Array(this.routerW.length);
    steQuantizeVec(this.routerW, 3, qRouter);
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

    const tensors = [
      {
        name: 'embed.E',
        codecId: X8dCodecId.KRON_ARITH_Q3,
        dtypeId: X8dDtypeId.Q3_ARITH_KRON,
        rows: this.E.meta.rows,
        cols: this.E.meta.cols,
        p1: this.E.meta.p1,
        q1: this.E.meta.q1,
        kTerms: this.E.meta.kTerms,
        hasBias: false,
        payload: serializeKronArithPayload(this.E),
      },
      {
        name: 'trunk.W1',
        codecId: X8dCodecId.KRON_ARITH_Q3,
        dtypeId: X8dDtypeId.Q3_ARITH_KRON,
        rows: this.W1.meta.rows,
        cols: this.W1.meta.cols,
        p1: this.W1.meta.p1,
        q1: this.W1.meta.q1,
        kTerms: this.W1.meta.kTerms,
        hasBias: false,
        payload: serializeKronArithPayload(this.W1),
      },
      {
        name: 'router.W',
        codecId: X8dCodecId.KRON_ARITH_Q3,
        dtypeId: X8dDtypeId.Q3_ARITH_KRON,
        rows: OMNI_D_MODEL,
        cols: OMNI_NUM_EXPERTS,
        p1: 1,
        q1: 1,
        kTerms: 1,
        hasBias: false,
        payload: serializeRouterArithPayload(this.routerW),
      },
      ...this.experts.map((ex, idx) => ({
        name: `expert.${idx}`,
        codecId: X8dCodecId.KRON_Q2,
        dtypeId: X8dDtypeId.Q2_KRON,
        rows: ex.meta.rows,
        cols: ex.meta.cols,
        p1: ex.meta.p1,
        q1: ex.meta.q1,
        kTerms: ex.meta.kTerms,
        hasBias: false,
        payload: serializeKronArithPayload(ex),
      })),
      {
        name: 'head.Wout',
        codecId: X8dCodecId.KRON_ARITH_Q3,
        dtypeId: X8dDtypeId.Q3_ARITH_KRON,
        rows: this.Wout.meta.rows,
        cols: this.Wout.meta.cols,
        p1: this.Wout.meta.p1,
        q1: this.Wout.meta.q1,
        kTerms: this.Wout.meta.kTerms,
        hasBias: true,
        payload: serializeKronArithPayload(this.Wout, this.bout),
      },
    ];

    const size = writeX8dV2Container(
      this.checkpointPath,
      this.metrics.step,
      this.metrics.effectiveWeights,
      tensors
    );
    this.metrics.checkpointBytesOnDisk = size;
    this.metrics.achievedBpw = Number(((size * 8) / this.metrics.effectiveWeights).toFixed(4));
    this.metrics.achievedDofBpb = Number(((size * 8) / this.metrics.trainableDofs).toFixed(3));
    this.saveMasterState();
    try {
      fs.writeFileSync(this.getMetaPath(), JSON.stringify(this.metrics, null, 2));
    } catch {
      // Non-fatal metadata write
    }
    return size;
  }

  public evaluateHeldOutAllDomains(maxBytesPerDomain = 220): void {
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

  /**
   * 256-State Uniform Corruption Denoising + Prompt-to-Output Step (Issue #54 & #56):
   * Corrupts span positions using uniform random 0..255 bytes (no MASK=256 token!)
   * at sampled noise levels {0.0, 0.25, 0.50} to train a true 256-state diffusion denoiser.
   */
  public runStepBatch(batchWindows = 16, lr = 0.045): number {
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
    const windowBytes = new Uint8Array(OMNI_CONTEXT_LEN);

    for (let b = 0; b < batchWindows; b++) {
      const domId = this.activeDomains[(this.metrics.step * batchWindows + b) % this.activeDomains.length];
      const buf = this.trainChunks[domId]!;
      const tPos = OMNI_CONTEXT_LEN + Math.floor(this.rand() * (buf.length - OMNI_CONTEXT_LEN - 1));
      const targetByte = buf[tPos];

      // Sample 256-state uniform diffusion noise rate: 50% clean (t=0), 25% light (0.20), 25% noisy (0.40)
      const uNoise = this.rand();
      const noiseProb = uNoise < 0.5 ? 0.0 : uNoise < 0.75 ? 0.20 : 0.40;

      for (let s = 0; s < OMNI_CONTEXT_LEN; s++) {
        const cleanB = buf[tPos - OMNI_CONTEXT_LEN + s];
        const useNoisy = s >= 2 && noiseProb > 0 && this.rand() < noiseProb;
        const obsB = useNoisy ? Math.floor(this.rand() * 256) : cleanB;
        windowBytes[s] = obsB;
        evalEmbedSlot(snap.qE.qA, snap.qE.qB, this.E.meta, obsB, this.xBuf, s * OMNI_EMB_DIM);
      }

      // Inject continuous diffusion timestep signal (sigma) into first 8 dims
      for (let c = 0; c < 8; c++) {
        this.xBuf[c] += (noiseProb - 0.15) * 0.12;
      }

      kronForward(this.W1.meta, snap.qW1.qA, snap.qW1.qB, this.xBuf, this.h1Pre, this.tmpT);
      for (let c = 0; c < OMNI_D_MODEL; c++) {
        const th = Math.tanh(this.h1Pre[c]);
        this.h1Act[c] = th;
        this.z1[c] = this.xBuf[c] + 0.5 * th;
      }

      // Top-1 Expert Router
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
        const byteId = windowBytes[s];
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

  public trainSynchronously(totalSteps = 280, batchWindows = 16): OmniTrainingMetricsSnapshot {
    if (!this.refreshDatasetsFromDisk()) {
      throw new Error('Omni datasets not found on disk. Run fetchAllOmniDatasets first.');
    }
    this.startedAtMs = Date.now();
    this.metrics.status = 'running';
    this.metrics.targetSteps = totalSteps;
    this.evaluateHeldOutAllDomains(140);

    for (let s = 1; s <= totalSteps; s++) {
      this.runStepBatch(batchWindows, 0.05);
      if (s % 70 === 0 || s === totalSteps) {
        this.evaluateHeldOutAllDomains(140);
        this.saveCheckpointX8d();
      }
    }
    this.metrics.status = 'completed';
    this.saveCheckpointX8d();
    return this.metrics;
  }

  public startBackgroundTraining(targetSteps = 800, batchWindows = 16): void {
    if (this.timer) return;
    if (!this.refreshDatasetsFromDisk()) {
      this.metrics.status = 'error';
      this.metrics.lastError = 'No datasets found on disk. Click Fetch / Expand HF Omni Datasets first.';
      return;
    }
    this.startedAtMs = Date.now() - this.metrics.elapsedMs;
    if (this.metrics.step === 0) {
      this.evaluateHeldOutAllDomains(140);
    }
    this.metrics.status = 'running';
    this.metrics.targetSteps = Math.max(targetSteps, this.metrics.step + 200);
    this.metrics.lastError = null;

    const tick = () => {
      if (this.metrics.status !== 'running') return;
      try {
        const chunk = 6;
        for (let i = 0; i < chunk && this.metrics.step < this.metrics.targetSteps; i++) {
          this.runStepBatch(batchWindows, 0.04);
        }
        if (this.metrics.step % 60 === 0 || this.metrics.step >= this.metrics.targetSteps) {
          this.evaluateHeldOutAllDomains(140);
          this.saveCheckpointX8d();
        }
        if (this.metrics.step >= this.metrics.targetSteps) {
          this.metrics.status = 'completed';
          this.saveCheckpointX8d();
          this.stopBackgroundTraining(true);
          return;
        }
        this.timer = setTimeout(tick, 12);
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
 * STRICT DIRECT-SEEK DISK RUNTIME (Zero mmap, zero JSON weights, zero master state):
 * Reads Safetensors-equivalent .x8D v2 tail directory, verifies CRC32 per span,
 * lazily seeks only routed Expert tensors, and executes 2-pass NCM System-1 / System-2
 * 256-state byte diffusion refinement.
 */
export function inferFromDiskX8dOnly(
  prompt: string,
  genBytesLen = 40,
  checkpointPath = getDefaultCheckpointPath()
): OmniDiskInferResult {
  if (!fs.existsSync(checkpointPath)) {
    throw new Error('No .x8D v2 checkpoint on disk yet. Run Omni training first.');
  }
  const t0 = Date.now();
  const stat = fs.statSync(checkpointPath);
  const fd = fs.openSync(checkpointPath, 'r');
  const seekStats: X8dSeekStats = {
    diskSeeksPerformed: 0,
    diskBytesSeekRead: 0,
    crcVerifiedCount: 0,
  };

  try {
    const directory = readDirectoryWithFd(fd, stat.size, seekStats);
    const byName = new Map<string, X8dTensorDirEntry>(directory.entries.map((e) => [e.name, e]));

    const E = deserializeKronArithPayload(readNamedTensorWithFd(fd, byName.get('embed.E')!, seekStats));
    const W1 = deserializeKronArithPayload(readNamedTensorWithFd(fd, byName.get('trunk.W1')!, seekStats));
    const routerW = deserializeRouterArithPayload(readNamedTensorWithFd(fd, byName.get('router.W')!, seekStats));
    const Wout = deserializeKronArithPayload(readNamedTensorWithFd(fd, byName.get('head.Wout')!, seekStats));

    const expertCache = new Map<number, ReturnType<typeof deserializeKronArithPayload>>();
    const activeExpertsUsed = new Set<number>();

    const seed = Array.from(new TextEncoder().encode(prompt));
    const stream = [...seed];
    const xBuf = new Float32Array(OMNI_D_MODEL);
    const h1Pre = new Float32Array(OMNI_D_MODEL);
    const z1 = new Float32Array(OMNI_D_MODEL);
    const expOut = new Float32Array(OMNI_HIDDEN_EXP);
    const expAct = new Float32Array(OMNI_HIDDEN_EXP);
    const logits = new Float32Array(OMNI_VOCAB_SIZE);
    const tmpT = new Float32Array(2048);

    let system1EarlyExits = 0;
    let system2Refinements = 0;

    const evalStep = (contextWindow: number[], sigma: number) => {
      for (let s = 0; s < OMNI_CONTEXT_LEN; s++) {
        evalEmbedSlot(E.qA, E.qB, E.meta, contextWindow[s] ?? 32, xBuf, s * OMNI_EMB_DIM);
      }
      for (let c = 0; c < 8; c++) xBuf[c] += (sigma - 0.15) * 0.12;
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
        const expEntry = byName.get(`expert.${bestExp}`)!;
        exp = deserializeKronArithPayload(readNamedTensorWithFd(fd, expEntry, seekStats));
        expertCache.set(bestExp, exp);
      }

      kronForward(exp.meta, exp.qA, exp.qB, z1, expOut, tmpT);
      for (let c = 0; c < OMNI_HIDDEN_EXP; c++) expAct[c] = Math.tanh(expOut[c]);
      kronForward(Wout.meta, Wout.qA, Wout.qB, expAct, logits, tmpT);
      if (Wout.bias) for (let v = 0; v < OMNI_VOCAB_SIZE; v++) logits[v] += Wout.bias[v];

      let maxL = -Infinity;
      let bestByte = 0;
      for (let v = 0; v < OMNI_VOCAB_SIZE; v++) {
        if (logits[v] > maxL) {
          maxL = logits[v];
          bestByte = v;
        }
      }
      let sumE = 0;
      for (let v = 0; v < OMNI_VOCAB_SIZE; v++) sumE += Math.exp(logits[v] - maxL);
      const topConf = Math.exp(logits[bestByte] - maxL) / sumE;
      return { bestByte, topConf };
    };

    // Pass 1 (NCM Coarse Draft + System 1 Early-Exit Gate) & Pass 2 (System 2 Denoise Refinement)
    for (let g = 0; g < genBytesLen; g++) {
      const n = stream.length;
      const win: number[] = [];
      for (let s = 0; s < OMNI_CONTEXT_LEN; s++) win.push(stream[n - OMNI_CONTEXT_LEN + s] ?? 32);

      // Iter 1: coarse draft (sigma=0.20)
      const iter1 = evalStep(win, 0.20);
      if (iter1.topConf >= 0.22) {
        system1EarlyExits++;
        stream.push(iter1.bestByte);
      } else {
        // Iter 2: System 2 clean-state refinement with self-conditioning carry (sigma=0.0)
        system2Refinements++;
        const refinedWin = [...win.slice(1), iter1.bestByte];
        const iter2 = evalStep(refinedWin, 0.0);
        stream.push(iter2.bestByte);
      }
    }

    const genBytes = new Uint8Array(stream.slice(seed.length));
    const continuation = new TextDecoder().decode(genBytes);
    const outputQuantaSample = Array.from(genBytes.slice(0, 10)).map((b) => (b * 0.001).toFixed(3));

    return {
      prompt,
      continuation,
      outputQuantaSample,
      activeExpertsUsed: Array.from(activeExpertsUsed),
      system1EarlyExits,
      system2Refinements,
      crcVerifiedCount: seekStats.crcVerifiedCount,
      diskFileSizeBytes: stat.size,
      diskBytesSeekRead: seekStats.diskBytesSeekRead,
      diskSeeksPerformed: seekStats.diskSeeksPerformed,
      latencyMs: Math.max(1, Date.now() - t0),
    };
  } finally {
    fs.closeSync(fd);
  }
}
