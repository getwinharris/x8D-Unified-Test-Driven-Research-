/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * Unified Safetensors-Equivalent Seekable .x8D Container Specification (v2):
 * Reconciles bapXai/x8Dsub-byte, bapXai/x8D-Omni-Diffusion, and this research workbench
 * into ONE magic-free, checksummed, tensor-named random-access binary format:
 *
 * Layout:
 *   [16-byte Global Header] (magic-free, Issue #52)
 *     0..3   uint32 versionAndFlags (2 = x8D v2 unified container)
 *     4..7   uint32 tensorCount
 *     8..11  uint32 trainStep
 *     12..15 uint32 effectiveWeights
 *   [Tensor Payload Spans 0..N-1] (packed in contiguous byte spans)
 *   [48-byte Tail Directory Entry × tensorCount]
 *     0..23  ASCII tensor name (null-padded, up to 24 chars, e.g. "expert.0", "trunk.W1")
 *     24     uint8  codecId  (1=raw_quanta_u8, 2=kron_q2, 3=kron_q4, 4=kron_arith_q3)
 *     25     uint8  dtypeId  (1=u8_quanta, 2=q2_kron, 3=q3_arith_kron, 4=q4_kron, 5=f32)
 *     26..27 uint16 rows
 *     28..29 uint16 cols
 *     30     uint8  p1
 *     31     uint8  q1
 *     32     uint8  kTerms
 *     33     uint8  hasBias
 *     34..35 uint16 reserved
 *     36..39 uint32 byteOffset
 *     40..43 uint32 byteLength
 *     44..47 uint32 crc32 (integrity verification on f.seek/f.read)
 */

import fs from 'fs';
import zlib from 'zlib';

export const X8D_CONTAINER_VERSION = 2;
export const X8D_GLOBAL_HEADER_BYTES = 16;
export const X8D_DIR_ENTRY_BYTES = 48;

export enum X8dCodecId {
  RAW_QUANTA_U8 = 1,
  KRON_Q2 = 2,
  KRON_Q4 = 3,
  KRON_ARITH_Q3 = 4,
}

export enum X8dDtypeId {
  U8_QUANTA = 1,
  Q2_KRON = 2,
  Q3_ARITH_KRON = 3,
  Q4_KRON = 4,
  F32 = 5,
}

export interface X8dTensorDirEntry {
  name: string;
  codecId: X8dCodecId;
  dtypeId: X8dDtypeId;
  rows: number;
  cols: number;
  p1: number;
  q1: number;
  kTerms: number;
  hasBias: boolean;
  byteOffset: number;
  byteLength: number;
  crc32: number;
}

export interface X8dNamedSpanInput {
  name: string;
  codecId: X8dCodecId;
  dtypeId: X8dDtypeId;
  rows: number;
  cols: number;
  p1: number;
  q1: number;
  kTerms: number;
  hasBias: boolean;
  payload: Uint8Array;
}

export interface X8dContainerDirectory {
  version: number;
  tensorCount: number;
  trainStep: number;
  effectiveWeights: number;
  fileSizeBytes: number;
  entries: X8dTensorDirEntry[];
}

export interface X8dSeekStats {
  diskSeeksPerformed: number;
  diskBytesSeekRead: number;
  crcVerifiedCount: number;
}

// Standard IEEE 802.3 CRC32 table
const CRC_TABLE = (() => {
  const tbl = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) {
      c = (c & 1) !== 0 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    tbl[i] = c >>> 0;
  }
  return tbl;
})();

export function computeCrc32(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    crc = CRC_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

/**
 * Fixes x8Dsub-byte's save_file(int(v) & 0xFF) truncation bug cleanly:
 * maps raw bytes 0..255 -> 0.000..0.255 quanta AND losslessly recovers 0..255 before U8 persistence.
 */
export function bytesToQuanta(bytes: Uint8Array): Float32Array {
  const out = new Float32Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) {
    out[i] = (bytes[i] & 0xff) * 0.001;
  }
  return out;
}

export function quantaToBytes(quanta: ArrayLike<number>): Uint8Array {
  const out = new Uint8Array(quanta.length);
  for (let i = 0; i < quanta.length; i++) {
    const q = Number(quanta[i]);
    if (!Number.isFinite(q)) {
      out[i] = 0;
      continue;
    }
    // If caller passes normalized 0.000..0.255 quanta (<= 0.256), invert via / 0.001; otherwise clamp 0..255
    const b = q >= 0 && q <= 0.256 ? Math.round(q * 1000) : Math.round(q);
    out[i] = Math.max(0, Math.min(255, b)) & 0xff;
  }
  return out;
}

export function x8dArithCompressBytes(raw: Uint8Array): Uint8Array {
  const deflated = new Uint8Array(zlib.deflateRawSync(raw, { level: 9 }));
  if (deflated.length + 1 < raw.length) {
    const out = new Uint8Array(deflated.length + 1);
    out[0] = 1; // Compressed flag
    out.set(deflated, 1);
    return out;
  }
  const out = new Uint8Array(raw.length + 1);
  out[0] = 0; // Raw fallback if incompressible
  out.set(raw, 1);
  return out;
}

export function x8dArithDecompressBytes(encoded: Uint8Array): Uint8Array {
  if (encoded.length === 0) return encoded;
  if (encoded[0] === 1) {
    return new Uint8Array(zlib.inflateRawSync(encoded.subarray(1)));
  }
  return encoded.subarray(1);
}

export function codecLabel(codecId: X8dCodecId): string {
  switch (codecId) {
    case X8dCodecId.RAW_QUANTA_U8:
      return '0x01 raw_quanta_u8';
    case X8dCodecId.KRON_Q2:
      return '0x02 kron_q2';
    case X8dCodecId.KRON_Q4:
      return '0x03 kron_q4';
    case X8dCodecId.KRON_ARITH_Q3:
      return '0x04 kron_arith_q3';
    default:
      return `0x0${Number(codecId).toString(16)}`;
  }
}

export function writeX8dV2Container(
  filePath: string,
  trainStep: number,
  effectiveWeights: number,
  tensors: X8dNamedSpanInput[]
): number {
  const globalHdr = new Uint8Array(X8D_GLOBAL_HEADER_BYTES);
  const gv = new DataView(globalHdr.buffer);
  gv.setUint32(0, X8D_CONTAINER_VERSION, true);
  gv.setUint32(4, tensors.length, true);
  gv.setUint32(8, trainStep, true);
  gv.setUint32(12, effectiveWeights, true);

  const tailDir = new Uint8Array(tensors.length * X8D_DIR_ENTRY_BYTES);
  const tv = new DataView(tailDir.buffer);
  const ascii = new TextEncoder();

  let offset = X8D_GLOBAL_HEADER_BYTES;
  for (let i = 0; i < tensors.length; i++) {
    const t = tensors[i];
    const base = i * X8D_DIR_ENTRY_BYTES;
    const nameBytes = ascii.encode(t.name.slice(0, 24));
    tailDir.set(nameBytes, base);
    tailDir[base + 24] = t.codecId;
    tailDir[base + 25] = t.dtypeId;
    tv.setUint16(base + 26, Math.min(65535, t.rows), true);
    tv.setUint16(base + 28, Math.min(65535, t.cols), true);
    tailDir[base + 30] = t.p1 & 0xff;
    tailDir[base + 31] = t.q1 & 0xff;
    tailDir[base + 32] = t.kTerms & 0xff;
    tailDir[base + 33] = t.hasBias ? 1 : 0;
    tv.setUint16(base + 34, 0, true);
    tv.setUint32(base + 36, offset, true);
    tv.setUint32(base + 40, t.payload.length, true);
    tv.setUint32(base + 44, computeCrc32(t.payload), true);
    offset += t.payload.length;
  }

  const tmpPath = `${filePath}.tmp`;
  const fd = fs.openSync(tmpPath, 'w');
  try {
    fs.writeSync(fd, globalHdr);
    for (const t of tensors) fs.writeSync(fd, t.payload);
    fs.writeSync(fd, tailDir);
  } finally {
    fs.closeSync(fd);
  }
  fs.renameSync(tmpPath, filePath);
  return fs.statSync(filePath).size;
}

export function readDirectoryWithFd(
  fd: number,
  fileSizeBytes: number,
  stats?: X8dSeekStats
): X8dContainerDirectory {
  const hdrBuf = Buffer.allocUnsafe(X8D_GLOBAL_HEADER_BYTES);
  fs.readSync(fd, hdrBuf, 0, X8D_GLOBAL_HEADER_BYTES, 0);
  if (stats) {
    stats.diskSeeksPerformed++;
    stats.diskBytesSeekRead += X8D_GLOBAL_HEADER_BYTES;
  }

  const gv = new DataView(hdrBuf.buffer, hdrBuf.byteOffset, hdrBuf.byteLength);
  const version = gv.getUint32(0, true);
  const tensorCount = gv.getUint32(4, true);
  const trainStep = gv.getUint32(8, true);
  const effectiveWeights = gv.getUint32(12, true);

  if (version !== X8D_CONTAINER_VERSION) {
    throw new Error(`Unsupported .x8D container version ${version} (expected ${X8D_CONTAINER_VERSION})`);
  }

  const tailSize = tensorCount * X8D_DIR_ENTRY_BYTES;
  const tailOffset = fileSizeBytes - tailSize;
  const tailBuf = Buffer.allocUnsafe(tailSize);
  fs.readSync(fd, tailBuf, 0, tailSize, tailOffset);
  if (stats) {
    stats.diskSeeksPerformed++;
    stats.diskBytesSeekRead += tailSize;
  }

  const tv = new DataView(tailBuf.buffer, tailBuf.byteOffset, tailBuf.byteLength);
  const dec = new TextDecoder();
  const entries: X8dTensorDirEntry[] = [];

  for (let i = 0; i < tensorCount; i++) {
    const base = i * X8D_DIR_ENTRY_BYTES;
    let nameEnd = 24;
    for (let j = 0; j < 24; j++) {
      if (tailBuf[base + j] === 0) {
        nameEnd = j;
        break;
      }
    }
    const name = dec.decode(tailBuf.subarray(base, base + nameEnd));
    entries.push({
      name,
      codecId: tailBuf[base + 24] as X8dCodecId,
      dtypeId: tailBuf[base + 25] as X8dDtypeId,
      rows: tv.getUint16(base + 26, true),
      cols: tv.getUint16(base + 28, true),
      p1: tailBuf[base + 30],
      q1: tailBuf[base + 31],
      kTerms: tailBuf[base + 32],
      hasBias: tailBuf[base + 33] === 1,
      byteOffset: tv.getUint32(base + 36, true),
      byteLength: tv.getUint32(base + 40, true),
      crc32: tv.getUint32(base + 44, true),
    });
  }

  return {
    version,
    tensorCount,
    trainStep,
    effectiveWeights,
    fileSizeBytes,
    entries,
  };
}

export function inspectX8dV2Directory(filePath: string): X8dContainerDirectory {
  const stat = fs.statSync(filePath);
  const fd = fs.openSync(filePath, 'r');
  try {
    return readDirectoryWithFd(fd, stat.size);
  } finally {
    fs.closeSync(fd);
  }
}

export const X8D_XAT_MAX_CONTEXT_BYTES = 65_536; // Explicit bapXai sequence-window capacity contract (64 KiB)

export function readNamedTensorWithFd(
  fd: number,
  entry: X8dTensorDirEntry,
  stats?: X8dSeekStats
): Uint8Array {
  if (entry.byteLength === 0) return new Uint8Array(0);
  const buf = Buffer.allocUnsafe(entry.byteLength);
  fs.readSync(fd, buf, 0, entry.byteLength, entry.byteOffset);
  const view = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  const actualCrc = computeCrc32(view);
  if (actualCrc !== entry.crc32) {
    throw new Error(
      `CRC32 mismatch seeking tensor "${entry.name}" (expected 0x${entry.crc32.toString(16)}, got 0x${actualCrc.toString(16)})`
    );
  }
  if (stats) {
    stats.diskSeeksPerformed++;
    stats.diskBytesSeekRead += entry.byteLength;
    stats.crcVerifiedCount++;
  }
  return view;
}

export function readBlockX8d(
  filePath: string,
  tensorName: string,
  localOffset: number,
  length: number
): Uint8Array {
  const dir = inspectX8dV2Directory(filePath);
  const entry = dir.entries.find((e) => e.name === tensorName);
  if (!entry) throw new Error(`Tensor "${tensorName}" not found in .x8D directory`);
  if (localOffset < 0 || length < 0 || localOffset + length > entry.byteLength) {
    throw new Error(`Out-of-bounds block read [${localOffset}, ${localOffset + length}) for "${tensorName}" (${entry.byteLength} B)`);
  }
  const fd = fs.openSync(filePath, 'r');
  try {
    const buf = Buffer.allocUnsafe(length);
    fs.readSync(fd, buf, 0, length, entry.byteOffset + localOffset);
    return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  } finally {
    fs.closeSync(fd);
  }
}

/**
 * x8Dsub-byte #4 Framework Storage ABI (no model.forward, no neural operators):
 * - xIn: validates 0..255 serialized stream & maps to 0.000..0.255 quanta coordinates
 * - xCompress: applies explicit codec policy (lossless quanta+arith vs factorized) & reports measured bpw
 * - xOut: restores raw 0..255 bytes (inverse / 0.001, fixing int(v)&0xFF collapse)
 */
export function xIn(rawBytes: Uint8Array): { bytes: Uint8Array; quanta: Float32Array; withinXAt: boolean } {
  return {
    bytes: new Uint8Array(rawBytes),
    quanta: bytesToQuanta(rawBytes),
    withinXAt: rawBytes.length <= X8D_XAT_MAX_CONTEXT_BYTES,
  };
}

export function xCompress(
  rawBytes: Uint8Array,
  policy: 'lossless_arith' | 'raw_quanta_u8' = 'lossless_arith'
): {
  codecId: X8dCodecId;
  lossless: boolean;
  payload: Uint8Array;
  measuredBitsPerByte: number;
} {
  if (policy === 'raw_quanta_u8') {
    const copy = new Uint8Array(rawBytes);
    return {
      codecId: X8dCodecId.RAW_QUANTA_U8,
      lossless: true,
      payload: copy,
      measuredBitsPerByte: 8.0,
    };
  }
  const payload = x8dArithCompressBytes(rawBytes);
  return {
    codecId: X8dCodecId.KRON_ARITH_Q3,
    lossless: true,
    payload,
    measuredBitsPerByte: rawBytes.length > 0 ? (payload.length * 8) / rawBytes.length : 0,
  };
}

export function xOut(
  payload: Uint8Array | ArrayLike<number>,
  codecId: X8dCodecId = X8dCodecId.RAW_QUANTA_U8
): Uint8Array {
  if (payload instanceof Uint8Array) {
    return codecId === X8dCodecId.KRON_ARITH_Q3 ? x8dArithDecompressBytes(payload) : new Uint8Array(payload);
  }
  return quantaToBytes(payload);
}

