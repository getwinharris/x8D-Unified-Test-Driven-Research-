export type AuditCategory =
  | 'sub1bit-0001'
  | 'mmap-latent'
  | 'ncm-reasoning'
  | 'training-groot'
  | 'tools-runtime';

export interface RepoIssueRecord {
  number: number;
  kind: 'ISSUE' | 'PR';
  state: 'OPEN' | 'CLOSED';
  category: AuditCategory;
  title: string;
  takeaway: string;
  impactOn50MPlan: string;
  proposedSolution: string;
}

export const REPO_METADATA = {
  repo: 'bapXai/x8D-Omni-Diffusion',
  canonicalName: 'x8D Omni Diffusion Language Model (x8D Omni DLM)',
  totalFiles: 138,
  totalIssues: 73,
  openIssues: 40,
  totalPRs: 1,
  openPRs: 1,
  closedPRs: 0,
  keyModules: [
    'omni_diffusion/x8d_subbyte.py',
    'omni_diffusion/x8d_arith.py',
    'omni_diffusion/x8d_mmap.py',
    'omni_diffusion/moe_disk.py',
    'omni_diffusion/x8d_byte_diffusion.py',
    'omni_diffusion/x8d_spec_decode.py',
    'tools/quantize_hf_safetensors.py',
    'tools/openai_chat_server.py',
  ],
};

export const REPO_ISSUES_AND_PRS: RepoIssueRecord[] = [
  {
    number: 74,
    kind: 'PR',
    state: 'OPEN',
    category: 'tools-runtime',
    title: 'feat(#73): deterministic System-1 calculator CLI and host tool endpoint',
    takeaway:
      'Extracts deterministic arithmetic into a stdlib Fraction parser (tools/x8d_calculator.py + POST /v1/tools/calculate) instead of burning neural diffusion passes.',
    impactOn50MPlan:
      'Keep kinematic unit conversions and exact trajectory math outside the unified model weights via a host tool boundary.',
    proposedSolution:
      'Merge PR #74 after adding bracket/exponent/trig guards, timeout caps, and typed MCP JSON schema registration inside user memory.x8D so tool calls skip prompt bloat.',
  },
  {
    number: 73,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'tools-runtime',
    title: 'feat(tools): exact terminal calculator for System-1 fast math and typed model tool calls',
    takeaway:
      'Specifies deterministic System-1 math dispatch and typed tool call framing inside the unified model harness.',
    impactOn50MPlan:
      'Routes arithmetic queries at Iteration 1 via deterministic tool dispatch in <3 ms.',
    proposedSolution:
      'Implement in Rust + Python Fraction backend wired to PR #74; log verified tool traces into local memory.x8D for zero-shot MCP invocation.',
  },
  {
    number: 72,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'tools-runtime',
    title: 'feat(tool): deterministic Rust calculator CLI for System-1 fast math and System-2 tool use',
    takeaway:
      'Provides a native compiled Rust CLI so edge/phone runtimes avoid Python startup latency on System-1 tool calls.',
    impactOn50MPlan:
      'Guarantees sub-millisecond arithmetic execution inside mobile/robotics harnesses.',
    proposedSolution:
      'Build a zero-dependency Rust crate (`x8d-calc`) with checked arbitrary-precision rationals and expose a single stdio JSON-RPC pipe usable by Rust and TypeScript.',
  },
  {
    number: 71,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'training-groot',
    title: 'research/feat: Eureka-style reproducible autonomous x8D experiment runner (isolated from model/harness)',
    takeaway:
      'Separates autonomous experiment search and reward iteration cleanly from core model weights and inference harnesses.',
    impactOn50MPlan:
      'Lets training ablations test LR, Muon schedules, and Hadamard scaling automatically without polluting production weights.',
    proposedSolution:
      'Run experiments inside ephemeral git worktrees writing metrics JSON + verified non-PII breakthroughs through the promotion gate before touching main model.x8D.',
  },
  {
    number: 70,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'mmap-latent',
    title: 'research/feat: Rust + TypeScript portable x8D runtime, CPU/NVMe baseline and CUDA-free backend tests',
    takeaway:
      'Distinguishes eliminating CUDA lock-in from still needing SRAM/DRAM tile ALUs while reading compressed spans directly from NVMe.',
    impactOn50MPlan:
      'Execute FSDP/ZeRO-2 training on clusters and stream edge inference via direct f.seek/read into L1/L2 CPU tiles.',
    proposedSolution:
      'Retire page-faulting mmap in x8d_mmap.py in favor of explicit `f.seek/f.read` + aligned tile scratchpad buffers in Rust (`std::fs::File::read_exact_at`) and Node.',
  },
  {
    number: 69,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'training-groot',
    title: 'feat(data/eval): x8D-1B raw-byte corpus, leakage-free benchmarks and staged omni-capability matrix',
    takeaway:
      'Supplies a leak-free raw-byte dataset and multi-stage capability matrix spanning text, image, audio, code, and robotics.',
    impactOn50MPlan:
      'Provides the clean validation splits used to gate 50M/150M/1B stages without data leakage.',
    proposedSolution:
      'Deduplicate via MinHash on raw byte shingles (0..255), compute Decontamination bitmasks against GSM8K/HumanEval/GR00T, and publish byte-parity loader manifests.',
  },
  {
    number: 68,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'training-groot',
    title: 'feat(train): x8D-1B from-scratch 256-byte diffusion — phased training and measured capability gates',
    takeaway:
      'Defines a clean from-scratch dense bidirectional Transformer baseline (20L, d=2048, SwiGLU 5632, vocab=256).',
    impactOn50MPlan:
      'Scale down directly to our 50M prototype (14L, d=512, SwiGLU 1536) before launching 1B+ multi-node runs.',
    proposedSolution:
      'Execute Phase 1–4 exit gates using 50M first; promote only runs beating perplexity & cosine thresholds onto the 1B and 1T Kronecker-MoE configs.',
  },
  {
    number: 67,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'ncm-reasoning',
    title: 'research/feat: hybrid latent reasoning and auditable tool/working-memory continuation',
    takeaway:
      'Combines internal continuous reasoning states with explicit auditable tool calls (arXiv:2610.18861 HybridLink).',
    impactOn50MPlan:
      'Bridges System-2 continuous latent reasoning cleanly into MCP/harness actions and user memory.x8D.',
    proposedSolution:
      'Add a lightweight HybridLink projection layer that flushes continuous latent hypotheses into explicit MCP tool frames recorded step-by-step into local memory.x8D.',
  },
  {
    number: 66,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'ncm-reasoning',
    title: 'research/feat: Uni-LaDiR-inspired shared multimodal latent diffusion with joint task training',
    takeaway:
      'Unified continuous latent space across text, vision, and robot kinematics yields +6.1% manipulation gains (arXiv:2609.19878).',
    impactOn50MPlan:
      'Shared internal workspace uniting language goals, camera patches, and motor chunks while I/O remains 0..255 bytes.',
    proposedSolution:
      'Project all modality byte blocks through a single shared RMSNorm-aligned latent projector supervised with continuation usefulness loss.',
  },
  {
    number: 65,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'ncm-reasoning',
    title: 'research/feat: trainable continuous reasoning state with causal tests for x8D diffusion',
    takeaway:
      'Mandates causal perturbation ablations so continuous latent recurrence provably contributes to downstream accuracy.',
    impactOn50MPlan:
      'Verify GR00T action precision and GSM8K reasoning degrade when continuous latent carry states are corrupted.',
    proposedSolution:
      'Add automated CI gate injecting Gaussian/byte-flip perturbations into latent carry `z_t`—fail build unless MSE/loss worsens >= 15% vs unperturbed control.',
  },
  {
    number: 64,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'ncm-reasoning',
    title: 'research/feat: NCM-inspired coarse-to-fine adaptive byte diffusion scheduling',
    takeaway:
      'Adapts Next-Concept Modeling (arXiv:2610.14162): November follow-up notes fixed 8x8 scheduling is only an x8D hypothesis, not a validated requirement.',
    impactOn50MPlan:
      'Keep fixed 8x8 solely as a control baseline; test adaptive entropy chunks, learned concept hierarchies, and KDA state under matched compute (target: >=4x hypothesis).',
    proposedSolution:
      'Run matched-FLOP A/B benchmarks comparing variable entropy chunks vs. learned NCM concepts vs. 8x8 baseline instead of hardcoding 8x8.',
  },
  {
    number: 63,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'ncm-reasoning',
    title: 'Establish canonical x8D Omni Diffusion Language Model definition and integrated algorithm contract',
    takeaway:
      'Unifies conflicting README, script, and issue descriptions and reconciles the outdated fixed-8x8 assumption.',
    impactOn50MPlan:
      'Explicitly separates 4 orthogonal layers: (1) 0..255 byte I/O, (2) trainable internal representation, (3) weight compression codec, and (4) S1/S2 scheduling.',
    proposedSolution:
      'Amend #63 ARCHITECTURE_CONTRACT.md to drop mandatory 8x8 phrasing, separate projected targets from measured results, and lock vocab=256 + f.seek/read + dual .x8D files.',
  },
  {
    number: 62,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'ncm-reasoning',
    title: 'research/feat: adapt Qwen3.8-Flash-Next to pure-byte x8D with zero-VRAM serving',
    takeaway:
      'Synthesizes Block AttnRes, 3:1 KDA/Gated DeltaNet + global MLA, and Muon+AdamW split for 75% smaller KV state.',
    impactOn50MPlan:
      'Provides the 10 KDA + 4 MLA backbone topology inside our 50M and 1T unified models.',
    proposedSolution:
      'Replace full softmax KV layers with 3:1 KDA recurrent matrix layers + Block AttnRes depth routing, and use Muon on 2D hidden matrices + AdamW on byte embeddings.',
  },
  {
    number: 61,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'sub1bit-0001',
    title: 'bug: x8d_arith (the only fractional-bit coder) is dead code — pack_subbyte_lossless never written',
    takeaway:
      'x8d_arith.py implements a valid 256-symbol range coder, yet x8d_subbyte.py imports it unused and never invokes it.',
    impactOn50MPlan:
      'Wire x8d_arith into seekable 4KB blocks so Codec 0x01 stream/embedding compression is real and bit-exact.',
    proposedSolution:
      'Implement `pack_subbyte_lossless()` in `x8d_subbyte.py` with order-1 residual predictors + 4KB seekable `x8d_arith` frames + unit roundtrip checks.',
  },
  {
    number: 60,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'sub1bit-0001',
    title: 'proposal: replace block-mean packer with measurable sub-bit transform + entropy coding',
    takeaway:
      'Demonstrates size_bits = N * H(q | context): real 0.008 bpw needs structured factorization plus entropy coding.',
    impactOn50MPlan:
      'Pairs Kronecker factors (Codec 0x03) with x8d_arith (Codec 0x01) and outlier scaling to hit 0.008 bpw cleanly.',
    proposedSolution:
      'Build Codec 0x03: Hadamard/per-channel diagonal outlier scaling -> SVD-initialized Kronecker `A ⊗ B` sub-byte cores -> `x8d_arith` range coding.',
  },
  {
    number: 59,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'sub1bit-0001',
    title: "bug: 'disk == source_bytes x 0.001' is a tautological gate — cannot detect information loss",
    takeaway:
      'Size-ratio checks alone passed corrupted checkpoints where weights collapsed to block averages.',
    impactOn50MPlan:
      'Replace file-size checks with mathematical fidelity gates before saving or loading checkpoints.',
    proposedSolution:
      'Enforce mandatory decompression CI gates: per-tensor cosine similarity >= 0.990, KL logit divergence <= 0.02, and byte stream memcmp exactness == 100%.',
  },
  {
    number: 58,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'sub1bit-0001',
    title: 'bug: BITS_PER_WEIGHT unit error — bits multiplied by value scale (0.008 bit/weight)',
    takeaway:
      'Multiplying byte[i] by 0.001 yields 256 distinct float values (still 8 bits of state cardinality, not 0.008 bits).',
    impactOn50MPlan:
      'Track state entropy/cardinality separately from the 0.000–0.255 external I/O coordinate representation.',
    proposedSolution:
      'Fix telemetry constants to measure compressed bitstream length divided by parameter count (`total_compressed_bits / num_params`), keeping `byte * 0.001` strictly as I/O value normalization.',
  },
  {
    number: 57,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'sub1bit-0001',
    title: 'bug: .x8D container cannot serve a forward pass — 1000-byte block mean discards per-weight info',
    takeaway:
      '_stream_pack collapses 1000 consecutive weights into 1 averaged byte, erasing 999 degrees of freedom per chunk.',
    impactOn50MPlan:
      'Delete chunk-mean averaging completely; retain structured Kronecker factor matrices readable directly per span.',
    proposedSolution:
      'Deprecate 1000-byte mean chunking in `x8d_subbyte.py` & `quantize_hf_safetensors.py`; replace with Kronecker+LittleBit factor spans evaluated on the fly via `f.seek/read`.',
  },
  {
    number: 56,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'training-groot',
    title: 'feat: REAL pure-byte training loop (256-state diffusion, QAT, .x8D export)',
    takeaway:
      'Converts training to strict 0..255 inputs/outputs with Straight-Through Estimator (STE) QAT and direct .x8D export.',
    impactOn50MPlan:
      'Eliminates offline PTQ collapse by letting gradients adapt weights to Hadamard-Kronecker sub-byte constraints during training.',
    proposedSolution:
      'Add STE fake-quant modules mimicking Codec 0x03 Hadamard+Kronecker during forward passes, export straight to direct-seek `.x8D` at checkpoint time.',
  },
  {
    number: 55,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'tools-runtime',
    title: 'bug: streaming quantizer dies on TimeoutError after hours — make it resumable + chunked',
    takeaway:
      'Single blocking HTTP connection stalls after hours on large HF checkpoints without range resume.',
    impactOn50MPlan:
      'Ensures multi-gigabyte/terabyte conversion jobs recover cleanly across network drops.',
    proposedSolution:
      'Use HTTP `Range:` chunked downloads (16 MiB windows) with exponential backoff retry, per-tensor `.part` checkpoint ledger, and atomic rename on finish.',
  },
  {
    number: 54,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'training-groot',
    title: 'feat: pure 256-state entropy-bound byte diffusion (uniform corruption, no MASK state)',
    takeaway:
      'Uses uniform random-byte corruption inside 0..255 and entropy-bound acceptance instead of synthetic MASK=256 IDs.',
    impactOn50MPlan:
      'Keeps external data/streams strictly within 256 unsigned bytes across text, games, audio, and robot telemetry.',
    proposedSolution:
      'Implement uniform random-byte noise kernel `q_t` on 0..255 paired with self-conditioning logit carry and Shannon entropy survival masking.',
  },
  {
    number: 52,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'mmap-latent',
    title: 'bug: x8d_export.py writes X8DGGUF1 magic + name-length headers + manifest — violates no-magic contract',
    takeaway:
      'Legacy export injects magic headers and manifests violating the raw magic-free container contract.',
    impactOn50MPlan:
      'Keep raw payload frames clean while placing deterministic span offsets in a separate or fixed-width tail index.',
    proposedSolution:
      'Strip `X8DGGUF1` magic strings and variable text manifests from payload streams; use fixed 24-byte tail footer `(tensor_hash_u64, offset_u64, len_u32, codec_u8)` for direct `f.seek`.',
  },
  {
    number: 51,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'sub1bit-0001',
    title: 'bug: 0.001 sub-byte law not applied — containers are 1 byte/param (50% off), not 99.9% smaller',
    takeaway:
      'quantize_state_dict_to_gguf stored 1 uint8 per parameter (~50% off FP16) rather than genuine sub-bit compression.',
    impactOn50MPlan:
      'Unblocks reaching true 0.008 bpw (~957 MB at 1T scale) on disk.',
    proposedSolution:
      'Route all `quantize_state_dict` outputs through Codec 0x03 Kronecker+x8d_arith encoder with verified >= 100x compression assertion.',
  },
  {
    number: 50,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'sub1bit-0001',
    title: 'bug: x8D quantization stores raw U8 bytes / scale manifest — 0.001 law not applied',
    takeaway:
      'Mixed Q8 affine scales with manifest metadata and destructive round(q*0.001) truncation.',
    impactOn50MPlan:
      'Prevents accidental precision destruction when storing weights in 0.000..0.255 quanta.',
    proposedSolution:
      'Separate raw I/O byte-quanta (`byte*0.001`, decoded `round(q/0.001)`) from weight storage Codec 0x03 (`A⊗B` + `x8d_arith`).',
  },
  {
    number: 48,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'tools-runtime',
    title: 'bug: image/audio generation returns text bytes, not real media files',
    takeaway:
      'Multimodal HTTP endpoints returned raw prompt text bytes instead of valid PNG/RIFF-WAV headers and payloads.',
    impactOn50MPlan:
      'Proves unified byte streams produce valid openable PNG/WAV assets natively.',
    proposedSolution:
      'Constrain modality byte decoders with magic-byte prefix conditioning (`89 50 4E 47` PNG / `52 49 46 46` WAV) and validate header CRC before returning API responses.',
  },
  {
    number: 45,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'mmap-latent',
    title: 'research/feat: zero-VRAM, low-RAM from-NVMe serving architecture',
    takeaway:
      'Establishes zero-VRAM low-RAM serving from compressed NVMe storage (upgraded from mmap to direct f.seek/f.read).',
    impactOn50MPlan:
      'Enables running 1T unified models on laptops and phones under <1 GB active RAM.',
    proposedSolution:
      'Replace `mmap.ACCESS_READ` page-table residency with direct `f.seek/f.read` span loading into reusable 64KB–2MB tile buffers + `io_uring` async prefetch.',
  },
  {
    number: 44,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'tools-runtime',
    title: 'docs: README + CHANGELOG + GitHub-vs-HF repo separation audit',
    takeaway:
      'Separates source code/tools (GitHub repo) from clean HF Hub model weight artifacts (minimal trust_remote_code + .x8D).',
    impactOn50MPlan:
      'Keeps HuggingFace model downloads fast and free of harness Markdown clutter.',
    proposedSolution:
      'Split publish pipeline: GitHub keeps training/web UI/tools; HF Hub receives only `model.x8D`, minimal `modeling_x8d.py`, and benchmark card.',
  },
  {
    number: 43,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'tools-runtime',
    title: 'feat(web): ChatGPT-style chat UI + OpenAI-compatible streaming endpoint (GitHub repo only)',
    takeaway:
      'Provides SSE streaming `/v1/chat/completions` endpoint and lightweight chat frontend backed by direct disk streaming.',
    impactOn50MPlan:
      'Allows direct local and mobile interaction with S1/S2 x8D models and memory.x8D context.',
    proposedSolution:
      'Wire `openai_chat_server.py` SSE generator directly to the `f.seek/read` span runner and inspectable `0.000..0.255` quanta decoder + local `memory.x8D` session mount.',
  },
  {
    number: 42,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'training-groot',
    title: 'feat: strip legacy modality tokenizers/wrappers; use real serialized bytes only',
    takeaway:
      'Removes MagViT-v2, BPE, and synthetic `IMG_START/IMG_END` special wrappers in favor of real serialized byte streams.',
    impactOn50MPlan:
      'Enforces single unified 0..255 representation across text, vision, audio, and GR00T sensor bytes.',
    proposedSolution:
      'Delete unused wrapper classes in `image_processor.py` & tokenizer utilities; feed raw UTF-8/PNG/WAV/sensor bytes straight to byte embedding table `[0..255]`.',
  },
  {
    number: 37,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'ncm-reasoning',
    title: 'feat(dspark): k-parallel byte blocks + multi-step diffusion prediction/verification',
    takeaway:
      'Adapts block-parallel multi-token generation to pure 256-byte diffusion blocks with confidence-based verification.',
    impactOn50MPlan:
      'Unlocks generating entire 8x8 (64-byte) or k-byte spans per forward pass.',
    proposedSolution:
      'Predict k=8..64 parallel bytes per pass with per-byte confidence head; accept contiguous high-confidence prefix in Iter 1 (S1) and re-noise low-confidence tail (S2).',
  },
  {
    number: 36,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'mmap-latent',
    title: 'feat(moe): SARA isolation + route-first zero-VRAM disk expert serving',
    takeaway:
      'Sparse Any-Route Architecture isolates disk byte spans per routed expert so inactive experts never consume RAM.',
    impactOn50MPlan:
      'Scales dense 50M prototype cleanly to 1T (138 routed experts, 2 active) and 10T–50T cloud tiers.',
    proposedSolution:
      'Group expert blocks contiguously inside `model.x8D` and run router logits first so `f.seek/read` fetches only the top-2 active expert byte spans per layer.',
  },
  {
    number: 34,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'training-groot',
    title: 'feat: training stack — first pure 256-state byte-diffusion forward pass',
    takeaway:
      'Establishes minimal end-to-end forward/backward pass with strictly 256 byte states and zero 264-vocab artifacts.',
    impactOn50MPlan:
      'Provides the verified gradient smoke test before launching multi-hour runs.',
    proposedSolution:
      'Create single-script CI smoke test checking input bounds `0<=x<=255`, uniform noise transition, loss decay on TinyStories bytes over 50 steps.',
  },
  {
    number: 30,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'training-groot',
    title: 'feat: integrate 256-state byte diffusion sampler + dataset pipeline',
    takeaway:
      'Connects raw-byte HF dataset streaming cleanly into the 256-state uniform-corruption sampler.',
    impactOn50MPlan:
      'Streams LeRobot, GR00T, GSM8K, and TinyStories bytes without pre-tokenization bottlenecks.',
    proposedSolution:
      'Implement iterable streaming dataloader converting HF rows to raw UTF-8/JSON byte blocks packed into 64-byte DSpark windows with prefetch queue.',
  },
  {
    number: 26,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'ncm-reasoning',
    title: 'docs(research): frontier architecture/data update — pure 256-byte diffusion + Qwen3.8 direction',
    takeaway:
      'Synchronizes architecture notes around pure 256-state diffusion and Qwen3.8 linear-recurrent hybrids.',
    impactOn50MPlan:
      'Keeps design docs consistent with 2025–2026 sub-1-bit, Bolmo, BLT-D, and NCM literature.',
    proposedSolution:
      'Update research index with BitNet b1.58, LittleBit-2, NanoQuant, BTC-LLM, Bolmo, BLT-D, and direct `f.seek/read` comparison tables.',
  },
  {
    number: 24,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'ncm-reasoning',
    title: 'docs/research: 2026 depth/context architectures — AttnRes, KDA, GDN/QSA, Gated Residual, Engram/N-gram',
    takeaway:
      'Evaluates modern recurrent state + depth-residual mechanisms suitable for strict CPU/NVMe byte-diffusion.',
    impactOn50MPlan:
      'Combines Block AttnRes depth mixing with O(1) KDA recurrent memory and local Engram n-gram tables.',
    proposedSolution:
      'Adopt 3:1 KDA:MLA + Block AttnRes (`N=8` blocks) and map N-gram/Engram lookups directly into local `memory.x8D` hash spans.',
  },
  {
    number: 21,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'training-groot',
    title: 'bug: ByteTokenizer/model paths must reject every id outside raw byte range 0-255',
    takeaway:
      'Legacy code wrapped or silently ignored IDs >= 256, causing subtle vocabulary corruption.',
    impactOn50MPlan:
      'Enforces hard invariant across all training, HF test, and inference endpoints.',
    proposedSolution:
      'Insert strict assertion `assert (ids >= 0).all() and (ids <= 255).all()` at model embedding entry and raise `ValueError` on any out-of-range integer.',
  },
  {
    number: 7,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'ncm-reasoning',
    title: 'research/feat: benchmark KDA/MLA vs Qwen3.8 GDN+QSA hybrid for byte diffusion',
    takeaway:
      'Benchmarks Kimi K3 3:1 KDA+MLA against Qwen3.8 GDN+QSA at matched byte-diffusion compute.',
    impactOn50MPlan:
      'Selects the highest throughput linear-attention ratio for edge CPU and 1T span execution.',
    proposedSolution:
      'Run 50M A/B benchmark at 8K–32K raw byte context measuring tokens/s, RAM footprint, and byte-level BPB under identical Muon+AdamW hyperparams.',
  },
  {
    number: 6,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'training-groot',
    title: 'feat: QAT/STE + optimizer ablation for compressed pure-byte model',
    takeaway:
      'Compares AdamW vs Muon+AdamW split alongside QAT/STE under Codec 0x03 sub-bit weight noise.',
    impactOn50MPlan:
      'Prevents accuracy drop when compressing trained matrices down to 0.008 bpw.',
    proposedSolution:
      'Add Hadamard rotation inside STE fake-quant + Muon spectral orthogonalization on 2D FFN/Attn weights and log BitNet-style stability curves.',
  },
  {
    number: 5,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'mmap-latent',
    title: 'feat: sparse disk-backed MoE with route-first prefetch',
    takeaway:
      'Configurable routed expert count (`E`, `top_k`) and optional shared expert aligned with direct NVMe span reads.',
    impactOn50MPlan:
      'Allows seamless scaling from dense 50M up to 138-expert 1T edge and 10T–50T cloud tiers.',
    proposedSolution:
      'Execute router + shared expert on tiny always-hot SRAM table, then dispatch async `io_uring` / `f.seek/read` solely for `top_k=2` expert slices.',
  },
  {
    number: 4,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'ncm-reasoning',
    title: 'feat: DSpark-style speculative byte-block diffusion (re-noise, no MASK state)',
    takeaway:
      'Speculative block drafting + survival acceptance strictly within 0..255 states without MASK tokens.',
    impactOn50MPlan:
      'Pairs naturally with NCM Iter-1 drafts to commit whole low-entropy code/boilerplate spans in 1 pass.',
    proposedSolution:
      'Implement self-speculative re-noising: accept prefix blocks where `p_accept >= 0.90` and re-noise only rejected tail indices uniformly over `0..255`.',
  },
  {
    number: 2,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'training-groot',
    title: 'feat: enforce pure raw-byte model domain (exactly 256 states, 0-255)',
    takeaway:
      'Canonical axiom eliminating BPE tokenizers forever: `vocab_size = 256`, `nn.Linear(d, 256)`.',
    impactOn50MPlan:
      'Guarantees universal text, game, audio, image, and robotics interoperability in one 0.000–0.255 space.',
    proposedSolution:
      'Lock `vocab_size=256` in config schema, strip every BPE tokenizer artifact from model repos, and tie byte embedding + LM head cleanly.',
  },
  {
    number: 41,
    kind: 'ISSUE',
    state: 'CLOSED',
    category: 'mmap-latent',
    title: 'research(feat): JustVugg/colibri (GLM-5.2 on 24GB RAM) + FSE/NVMe telemetry extraction',
    takeaway:
      'Demonstrated running 744B/40B active MoE inside 24–25 GB RAM from NVMe with FSE compression.',
    impactOn50MPlan:
      'Inspires sub-byte FSE/range entropy tables paired with x8D’s even leaner direct `f.seek/read` runtime (dropping Colibrì mmap overhead).',
    proposedSolution:
      'Closed reference benchmark: adopt Colibrì’s FSE/entropy lookup insight while replacing mmap page faults with deterministic `f.seek/read` tile streaming.',
  },
  {
    number: 3,
    kind: 'ISSUE',
    state: 'CLOSED',
    category: 'sub1bit-0001',
    title: 'feat: x8Dsub-byte 0.001 sub-byte weight compression + .gguf export',
    takeaway:
      'Initial prototype establishing magic-free raw binary container layout and the 0.001 coordinate concept.',
    impactOn50MPlan:
      'Keep raw binary layout discipline; supersede raw byte-mean encoding with Codec 0x01 + Codec 0x03.',
    proposedSolution:
      'Closed ancestor issue: fully superseded by open fixes #57–#61 (Kronecker + Hadamard scaling + x8d_arith range coding).',
  },
];

export interface FrontierComparisonRow {
  method: string;
  targetBpw: string;
  mechanism: string;
  executionStrategy: string;
  tradeoffVsX8D: string;
  isX8D?: boolean;
}

export const SUB_1_BIT_COMPARISON: FrontierComparisonRow[] = [
  {
    method: 'x8D Unified (Codec 0x01 + 0x03)',
    targetBpw: '0.008 bpw target (0.001 B/w)',
    mechanism: 'Hadamard channel outlier scaling + Kronecker factorization + 0.000–0.255 coordinates + x8d_arith range coding',
    executionStrategy: 'Direct f.seek / f.read span streaming into L1/L2 CPU tiles (zero mmap; projected <1 GB RAM @ 1T)',
    tradeoffVsX8D: 'Architectural target (~957 MB projected @ 1T); micro-matrix & Codec 0x01 measured locally, pending full 50M/1B checkpoint PPL eval.',
    isX8D: true,
  },
  {
    method: 'LittleBit / LittleBit-2 (NeurIPS 25 / ICML 26)',
    targetBpw: '0.10 bpw (~31x shrink)',
    mechanism: 'Low-rank latent matrix factorization (W ≈ U V^T) with binarized factors + row/col/latent scales + Joint-ITQ',
    executionStrategy: 'Custom low-bit GPU kernels (5.5x–11.6x speedup over FP16)',
    tradeoffVsX8D: 'Closest structural relative to x8D Codec 0x03, but ~12.5x larger than x8D 0.008 bpw and GPU-kernel bound.',
  },
  {
    method: 'NanoQuant (Samsung, ICML 26)',
    targetBpw: '0.50 – 0.60 bpw (~25.8x shrink)',
    mechanism: 'Calibration-only PTQ mixture-of-rank binary bases initialized via Hessian-aware ADMM',
    executionStrategy: 'Custom binary GEMV/GEMM CUDA kernels (20.1 tok/s on 70B / 8GB GPU)',
    tradeoffVsX8D: 'Strong zero-retraining PTQ, but ~60x heavier per weight than 0.008 bpw and drops on complex math/code.',
  },
  {
    method: 'BTC-LLM (ACL 26)',
    targetBpw: '0.70 – 0.90 bpw',
    mechanism: 'Flash Binary Codebook clustering +-1 vectors + learnable invertible outlier transforms',
    executionStrategy: 'Standard GPU kernels without sparse mask overhead (1.6x speedup)',
    tradeoffVsX8D: 'Highlighted critical activation outlier lesson: inspired adding Hadamard/diagonal channel scaling before x8D 0..0.255 projection.',
  },
  {
    method: 'BitNet b1.58 2B4T (Microsoft, 2025)',
    targetBpw: '1.58 bpw (ternary {-1,0,+1})',
    mechanism: 'From-scratch QAT restricting weights to {-1, 0, +1} with absmean scaling',
    executionStrategy: 'CPU lookup / bitnet.cpp SIMD int8/2-bit kernels (no FP16 mult)',
    tradeoffVsX8D: 'Nearly 200x heavier on disk than 0.008 bpw and uses ternary -1 weights that x8D explicitly replaces.',
  },
  {
    method: 'EntroLLM & NWC (NeurIPS 25 / ICML 26)',
    targetBpw: '~1.0 – 3.0 bpw',
    mechanism: 'Entropy-coded low-bit weights decoded lazily inside custom register/GPU matmul engines',
    executionStrategy: 'Lossless Huffman/ANS + CUDA matmul decode',
    tradeoffVsX8D: 'Proves entropy coding on weights is CPU/GPU decode viable; x8D extends this concept into 0.008 bpw Kronecker+arith spans.',
  },
];

export interface SpatialLiteratureItem {
  id: string;
  paper: string;
  venueOrId: string;
  mechanism: string;
  efficiencyGain: string;
  roleIn50MHybrid: string;
}

export const SPATIAL_ENCODING_LITERATURE: SpatialLiteratureItem[] = [
  {
    id: 'ncm-2026',
    paper: 'Next-Concept Modeling (NCM)',
    venueOrId: 'arXiv:2610.14162',
    mechanism:
      'Coarse-to-fine parallel generation over hierarchical residual concept grids; predicts structural outline first inside Iter 1, then refines detail levels.',
    efficiencyGain: '10-31x sampling speedup vs flat iterative diffusion',
    roleIn50MHybrid:
      'Drives NCM-in-Loop System 1 fast exit for low-entropy reflex/game blocks while System 2 iterates only high-entropy spans.',
  },
  {
    id: 'uniladir',
    paper: 'Uni-LaDiR: Unified Latent Diffusion Reasoning',
    venueOrId: 'arXiv:2609.19878',
    mechanism:
      'Shared continuous reasoning space unifying text, vision, and 3D/robot state blocks optimized for multi-step continuation usefulness.',
    efficiencyGain: '+7.3% visual reasoning, +6.1% robot manipulation accuracy',
    roleIn50MHybrid:
      'Provides the unified internal latent workspace connecting language goals, camera patches, and motor action chunks.',
  },
  {
    id: 'bolmo-bltd',
    paper: 'Bolmo & BLT-D (Byte Latent Diffusion)',
    venueOrId: '2025–2026 Tokenless Frontier',
    mechanism:
      'Replaces static BPE vocabularies with dynamic entropy-based byte patching and discrete byte diffusion directly over raw 0..255 streams.',
    efficiencyGain: 'Matches subword LMs at parity FLOPs with zero OOV/UTF-8 bugs',
    roleIn50MHybrid:
      'Validates x8D’s vocab_size=256 axiom across English, code, math digits, and binary game/robotics streams.',
  },
  {
    id: 'video-vae-dit',
    paper: 'GR00T 2 / Cosmos Spatial-Temporal Video VAE + DiT',
    venueOrId: 'NVIDIA Physical AI (2026)',
    mechanism:
      'Compresses multi-camera RGB streams 4x temporally and 8x spatially into structured latent grids conditioned via spatial cross-attention.',
    efficiencyGain: '256x pixel token reduction before transformer attention',
    roleIn50MHybrid:
      'Compresses multi-view camera bytes so backbone FLOPs focus on end-effector kinematics.',
  },
  {
    id: 'kda-gdn2',
    paper: 'Kimi Delta Attention (KDA) & Block AttnRes',
    venueOrId: 'arXiv:2510.26692 / arXiv:2603.15031',
    mechanism:
      'Associative matrix state with channel-wise forgetting interleaved 3:1 with global MLA, plus softmax depth residuals across N=8 blocks.',
    efficiencyGain: 'O(1) inference state · 75% KV reduction · 1.25x effective compute',
    roleIn50MHybrid:
      'Maximizes deep reasoning capacity within phone/CPU <1 GB RAM budgets.',
  },
];

export interface TrainingPhase {
  phase: string;
  title: string;
  objective: string;
  datasetSlice: string;
  memoryAndComputeStrategy: string;
  exitGate: string;
}

export const TRAINING_ROADMAP: TrainingPhase[] = [
  {
    phase: 'Phase 0',
    title: 'Codec 0x01/0x03 & Hadamard Outlier Gate Fix (Pre-Training)',
    objective:
      'Replace #57 block-mean packing and #59 size checks with Hadamard outlier scaling + Kronecker decomposition + 4KB seekable x8d_arith frames (no mmap).',
    datasetSlice: 'Synthetic structural tensors + GR00T calibration shards (100 MB)',
    memoryAndComputeStrategy:
      'Pure CPU / stdlib + Rust/Python container validator verifying lossless Codec 0x01 byte roundtrip & direct f.seek/read offsets.',
    exitGate: 'Cosine >= 0.994 (lossless arith = 1.000) and direct f.seek/read RSS <= 20 MB (zero mmap).',
  },
  {
    phase: 'Phase 1',
    title: '50M Spatial-Temporal Encoder & Unified S1/S2 NCM Alignment',
    objective:
      'Train the 50M backbone (10 KDA + 4 MLA layers, Block AttnRes, 0..255 I/O) from random init on GR00T, Snake/PushT reflexes, and raw text.',
    datasetSlice: 'GR00T-X-Embodiment-Sim + LeRobot PushT + GSM8K + TinyStories bytes',
    memoryAndComputeStrategy:
      'FSDP / ZeRO-2 with Muon optimizer on 2D hidden matrices, AdamW on embeddings/gates; direct pread/f.seek shard streaming.',
    exitGate: 'Iteration 1 NCM grid converges; System 1 reflex ticks hit <= 5 ms while preserving System 2 continuity.',
  },
  {
    phase: 'Phase 2',
    title: 'Adaptive Representation Ablations & Causal Latent Verification',
    objective:
      'Compare adaptive entropy chunks, learned NCM concepts, and KDA state vs. fixed 8x8 baseline (#64/#65) plus PR #74 deterministic Fraction host tool calls.',
    datasetSlice: 'Multi-embodiment manipulation + Sudoku constraints + GSM8K reasoning',
    memoryAndComputeStrategy:
      'Skip low-entropy spans after Iter 1; run matched-FLOP ablations & gradient checkpointing per AttnRes block.',
    exitGate: 'Causal ablation proves >= 15% degradation when latent carry is perturbed; measure real tokens/s vs 48-step baseline (no unverified speedup claims).',
  },
  {
    phase: 'Phase 3',
    title: '0.008 bpw Hadamard-Kronecker + Local memory.x8D Continual Sync',
    objective:
      'Export 50M–1T model.x8D at 0.008 bpw (no -1 ternary) and attach per-user on-device memory.x8D for MCP schemas, skills, files, and preferences.',
    datasetSlice: 'Agentic MCP tool trajectories + HuggingFace omni validation streams',
    memoryAndComputeStrategy:
      'Direct f.seek/read dual-file runner loading active model.x8D + memory.x8D spans under <1 GB RAM with privacy gate.',
    exitGate: '1.00T model.x8D fits in ~957 MB flash / ~180 MB active RAM; private PII stays locked in local memory.x8D.',
  },
  {
    phase: 'Phase 4',
    title: '10T–50T Cloud model.x8D Cluster + Anonymized Skill Promotion',
    objective:
      'Scale distributed cloud model.x8D (10T–50T @ 9.3–46.6 GB compressed) paired with sovereign user memory.x8D instances and anonymized breakthrough sync.',
    datasetSlice: 'Global omni-modal corpus + privacy-scrubbed verified skill breakthroughs',
    memoryAndComputeStrategy:
      'SARA route-first disk expert isolation (#36) across cloud NVMe spans + deterministic local memory.x8D continual training.',
    exitGate: 'Zero raw user PII leakage; >= 85% distributed efficiency across direct-seek NVMe nodes.',
  },
];
