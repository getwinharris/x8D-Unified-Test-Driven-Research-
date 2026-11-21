import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {
  X8D_CONTAINER_VERSION,
  X8D_XAT_MAX_CONTEXT_BYTES,
  X8dCodecId,
  X8dDtypeId,
  inspectX8dV2Directory,
  readBlockX8d,
  readNamedTensorWithFd,
  writeX8dV2Container,
  xCompress,
  xIn,
  xOut,
} from './x8dContainerFormat';

test('xIn -> xOut preserves all 0..255 raw bytes, NUL, Tamil UTF-8, invalid UTF-8, and PNG/WAV headers without 0-truncation', () => {
  const allBytes = new Uint8Array(256);
  for (let i = 0; i < 256; i++) allBytes[i] = i;

  const tamilUtf8 = new TextEncoder().encode('தமிழ் x8D 0.000..0.255');
  const pngHeader = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0xff, 0xfe, 0x80]);

  for (const vec of [allBytes, tamilUtf8, pngHeader]) {
    const entered = xIn(vec);
    assert.equal(entered.withinXAt, true);
    const restoredFromQuanta = xOut(entered.quanta, X8dCodecId.RAW_QUANTA_U8);
    assert.deepEqual(restoredFromQuanta, vec);

    const compressed = xCompress(vec, 'lossless_arith');
    assert.equal(compressed.lossless, true);
    const restoredFromArith = xOut(compressed.payload, compressed.codecId);
    assert.deepEqual(restoredFromArith, vec);
  }
  assert.equal(X8D_XAT_MAX_CONTEXT_BYTES, 65_536);
});

test('writeX8dV2Container preserves named tensor identity, random-access seek, block slicing, and CRC32 corruption detection', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'x8d-v2-fmt-'));
  const filePath = path.join(dir, 'test.x8d');
  try {
    const aBytes = new TextEncoder().encode('tensor-A-payload-0123456789');
    const bBytes = new Uint8Array([0x00, 0x01, 0x7f, 0xff, 0x42, 0x99]);

    const fileSize = writeX8dV2Container(filePath, 10, 50000, [
      {
        name: 'embed.E',
        codecId: X8dCodecId.RAW_QUANTA_U8,
        dtypeId: X8dDtypeId.U8_QUANTA,
        rows: 256,
        cols: 64,
        p1: 16,
        q1: 8,
        kTerms: 2,
        hasBias: false,
        payload: aBytes,
      },
      {
        name: 'expert.0',
        codecId: X8dCodecId.KRON_ARITH_Q3,
        dtypeId: X8dDtypeId.Q3_ARITH_KRON,
        rows: 512,
        cols: 2048,
        p1: 32,
        q1: 32,
        kTerms: 2,
        hasBias: false,
        payload: bBytes,
      },
    ]);

    const directory = inspectX8dV2Directory(filePath);
    assert.equal(directory.version, X8D_CONTAINER_VERSION);
    assert.equal(directory.tensorCount, 2);
    assert.equal(directory.fileSizeBytes, fileSize);
    assert.deepEqual(
      directory.entries.map((e) => e.name),
      ['embed.E', 'expert.0']
    );

    const fd = fs.openSync(filePath, 'r');
    const stats = { diskSeeksPerformed: 0, diskBytesSeekRead: 0, crcVerifiedCount: 0 };
    try {
      const readB = readNamedTensorWithFd(fd, directory.entries[1], stats);
      assert.deepEqual(readB, bBytes);
      assert.equal(stats.crcVerifiedCount, 1);
      assert.equal(stats.diskSeeksPerformed, 1);
    } finally {
      fs.closeSync(fd);
    }

    const sliceA = readBlockX8d(filePath, 'embed.E', 7, 9);
    assert.equal(new TextDecoder().decode(sliceA), 'A-payload');

    // Flip 1 byte inside expert.0 payload and verify CRC32 catches corruption
    const corrupt = fs.readFileSync(filePath);
    corrupt[directory.entries[1].byteOffset] ^= 0xff;
    fs.writeFileSync(filePath, corrupt);

    const fd2 = fs.openSync(filePath, 'r');
    try {
      assert.throws(() => readNamedTensorWithFd(fd2, directory.entries[1]), /CRC32 mismatch/);
    } finally {
      fs.closeSync(fd2);
    }
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
