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
  impactOnUnifiedModel: string;
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

export interface AlignmentAuditRow {
  dimension: string;
  repoIssuesSaying: string;
  userSayingInThread: string;
  currentWorkbenchDoing: string;
  status: 'aligned' | 'fixed-this-turn' | 'requires-training';
}

export const ALIGNMENT_AUDIT_MATRIX: AlignmentAuditRow[] = [
  {
    dimension: '1. Single Unified Model Identity',
    repoIssuesSaying:
      '#63 & #2: Define ONE canonical x8D Omni Diffusion model with vocab_size=256 (0..255 raw bytes), zero BPE, and zero MASK=256 IDs.',
    userSayingInThread:
      'Focus strictly on ONE unified x8D Omni Diffusion–NCM model. Stop mixing unrelated side models, extra products, or invented codec labels.',
    currentWorkbenchDoing:
      'Refocused the entire workbench onto ONE unified x8D model; deleted 5 unrelated panels (memory.x8D, duplicate HF/Footprint/Spatial sections) and stripped all invented codec names.',
    status: 'aligned',
  },
  {
    dimension: '2. System 1 Decision Expert Inside Mixture',
    repoIssuesSaying:
      '#72, #73, #74 (PR), #37, #4: Add deterministic System-1 math dispatch and Iter-1 fast speculation alongside deliberate System-2 multi-step diffusion.',
    userSayingInThread:
      'System 1 decision-making (robotics, games, fast control, PR #74 math) must live INSIDE the same unified mixture—not only AR, code, math, and QA, and not as a separate model.',
    currentWorkbenchDoing:
      'Unified System 1 Decision Expert (1-step Snake control, robot ticks, PR #74 Fraction math) and System 2 (9x9 Sudoku constraints, deep QA) live inside one shared weight set.',
    status: 'aligned',
  },
  {
    dimension: '3. Prompt-to-Output NCM Efficiency',
    repoIssuesSaying:
      '#64 & #65: Adapt Next-Concept Modeling (arXiv:2610.14162) coarse-to-fine scheduling without hardcoding 8x8; verify latent carry causally.',
    userSayingInThread:
      'Take default 8-bit inputs (Prompt/Dataset 0..255 -> 0.000..0.255), apply NCM efficiency inside the loop from Prompt to Output, and decode back to real UTF-8 language or actions.',
    currentWorkbenchDoing:
      'Encodes raw 8-bit bytes into 0.000..0.255 across the NCM-Diffusion loop and decodes output quanta back into readable UTF-8 text (language answers, Sudoku boards, Snake moves).',
    status: 'aligned',
  },
  {
    dimension: '4. 0.008 bpw (0.001 Sub-Byte) Storage vs. I/O',
    repoIssuesSaying:
      '#50, #51, #57, #58, #59, #60, #61: Scaling bytes x 0.001 is 8 bpw (unit error #58); 1000:1 block averaging (#57) destroys weights; x8d_arith (#61) is dead code; replace with factorization + QAT (#56) + x8d_arith (#60).',
    userSayingInThread:
      'Target 0.008 bits per full 8-bit (0.001 sub-byte) on disk without -1 ternary. Train directly in the compact representation from step 0 and measure real serialized bytes without formula shortcuts.',
    currentWorkbenchDoing:
      'Fixed all formula shortcuts: measures real serialized Uint8Array buffers across Kronecker+STE models trained from random init, comparing actual bpw & held-out PPL against BF16, INT8, INT4, and Broken #57.',
    status: 'fixed-this-turn',
  },
  {
    dimension: '5. Pretrained Checkpoints (needle3) & Native STE Training',
    repoIssuesSaying:
      '#56 & #59: Pretrained dense FP16 weights collapse under naive PTQ without QAT/STE training; measure cosine >= 0.990 and held-out perplexity instead of size-only checks.',
    userSayingInThread:
      'Test real needle3 weights parameter-to-parameter AND prove a compact model learns natively from step 0, exports to .x8D, and generates coherent output when reloaded strictly from .x8D.',
    currentWorkbenchDoing:
      'Runs both real 242 MB needle3.safetensors PTQ benchmarks AND from-scratch Dense vs. Compact Native .x8D (STE) training with strict Uint8Array reload-only held-out PPL & UTF-8 generation.',
    status: 'fixed-this-turn',
  },
  {
    dimension: '6. Direct f.seek / f.read Serving (No mmap)',
    repoIssuesSaying:
      '#45, #70, #52: Remove magic X8DGGUF1 headers (#52) and move away from OS page-fault mmap overhead toward explicit f.seek/f.read tile streaming under <1 GB RAM.',
    userSayingInThread:
      'We do not need mmap. Read active weight spans directly from the compressed .x8D model via f.seek/f.read and divide / 0.001 inside phone or small CPU RAM (<1 GB for 1T scale).',
    currentWorkbenchDoing:
      'Simulator defaults to Raw .x8D Direct (f.seek/read, no mmap) showing <1 GB disk and <1 GB working RAM at 1T scale; real C/Rust/Python streaming kernel still needs end-to-end training & export.',
    status: 'fixed-this-turn',
  },
];

export const REPO_ISSUES_AND_PRS: RepoIssueRecord[] = [
  {
    number: 74,
    kind: 'PR',
    state: 'OPEN',
    category: 'tools-runtime',
    title: 'feat(#73): deterministic System-1 calculator CLI and host tool endpoint',
    takeaway:
      'Extracts deterministic arithmetic into a stdlib Fraction parser (tools/x8d_calculator.py + POST /v1/tools/calculate) instead of burning neural diffusion passes.',
    impactOnUnifiedModel:
      'Acts as the deterministic System-1 math fast path inside the unified runtime.',
    proposedSolution:
      'Merge PR #74 with strict bracket/exponent guards and timeout caps so System-1 math executes in <3 ms without burning System-2 steps.',
  },
  {
    number: 73,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'tools-runtime',
    title: 'feat(tools): exact terminal calculator for System-1 fast math and typed model tool calls',
    takeaway:
      'Specifies deterministic System-1 math dispatch and typed tool call framing inside the unified model harness.',
    impactOnUnifiedModel:
      'Routes arithmetic queries at Iteration 1 inside the unified mixture.',
    proposedSolution:
      'Wire typed calculator calls directly to PR #74 inside the Iteration-1 System-1 expert path.',
  },
  {
    number: 72,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'tools-runtime',
    title: 'feat(tool): deterministic Rust calculator CLI for System-1 fast math and System-2 tool use',
    takeaway:
      'Provides a native compiled Rust CLI so phone/CPU runtimes avoid Python startup latency on System-1 tool calls.',
    impactOnUnifiedModel:
      'Guarantees sub-millisecond arithmetic execution inside mobile/robotics harnesses.',
    proposedSolution:
      'Compile a zero-dependency `x8d-calc` binary with checked rationals exposed over stdio JSON-RPC.',
  },
  {
    number: 71,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'training-groot',
    title: 'research/feat: Eureka-style reproducible autonomous x8D experiment runner (isolated from model/harness)',
    takeaway:
      'Separates autonomous experiment search cleanly from core model weights and inference harnesses.',
    impactOnUnifiedModel:
      'Lets training ablations test LR and QAT schedules without polluting production checkpoints.',
    proposedSolution:
      'Run ablations inside isolated git worktrees gated by perplexity, cosine, and latency checks before promoting to main `model.x8D`.',
  },
  {
    number: 70,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'mmap-latent',
    title: 'research/feat: Rust + TypeScript portable x8D runtime, CPU/NVMe baseline and CUDA-free backend tests',
    takeaway:
      'Distinguishes eliminating CUDA lock-in from still needing L1/L2 tile buffers while streaming spans from disk.',
    impactOnUnifiedModel:
      'Drives CPU/phone inference via direct `f.seek/f.read` into compact tile scratchpads under 1 GB RAM.',
    proposedSolution:
      'Retire page-faulting mmap in `x8d_mmap.py` in favor of explicit `f.seek/f.read` (`read_exact_at` in Rust) into aligned CPU tile buffers.',
  },
  {
    number: 69,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'training-groot',
    title: 'feat(data/eval): x8D-1B raw-byte corpus, leakage-free benchmarks and staged omni-capability matrix',
    takeaway:
      'Supplies a leak-free raw-byte dataset spanning text, image, audio, code, games, and robotics.',
    impactOnUnifiedModel:
      'Provides clean validation splits to verify unified System 1 + System 2 gates across 50M and 1B runs.',
    proposedSolution:
      'Deduplicate on raw byte shingles (0..255) and decontaminate GSM8K, HumanEval, Sudoku, and robotics splits.',
  },
  {
    number: 68,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'training-groot',
    title: 'feat(train): x8D-1B from-scratch 256-byte diffusion — phased training and measured capability gates',
    takeaway:
      'Defines a clean from-scratch dense bidirectional Transformer baseline (20L, d=2048, SwiGLU 5632, vocab=256).',
    impactOnUnifiedModel:
      'Scales directly from a 50M prototype (14L, d=512) up to 1B and 1T Kronecker-MoE configs.',
    proposedSolution:
      'Pass 50M exit gates under #56 QAT first before promoting weights to 1B and 1T edge exports.',
  },
  {
    number: 67,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'ncm-reasoning',
    title: 'research/feat: hybrid latent reasoning and auditable tool/working-memory continuation',
    takeaway:
      'Combines internal continuous reasoning states with explicit auditable tool calls inside one model.',
    impactOnUnifiedModel:
      'Connects System-2 latent carry cleanly to System-1 tool/action emissions.',
    proposedSolution:
      'Flush continuous latent hypotheses into deterministic byte frames validated before returning outputs.',
  },
  {
    number: 66,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'ncm-reasoning',
    title: 'research/feat: Uni-LaDiR-inspired shared multimodal latent diffusion with joint task training',
    takeaway:
      'Shared internal latent workspace across language, vision, and robot kinematics while external I/O remains 0..255 bytes.',
    impactOnUnifiedModel:
      'Lets one unified mixture jointly handle games, robotics, code, math, and language QA.',
    proposedSolution:
      'Project all modality byte blocks through a single RMSNorm-aligned latent trunk supervised end-to-end.',
  },
  {
    number: 65,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'ncm-reasoning',
    title: 'research/feat: trainable continuous reasoning state with causal tests for x8D diffusion',
    takeaway:
      'Mandates causal perturbation tests so continuous latent carry provably helps downstream accuracy.',
    impactOnUnifiedModel:
      'Verifies System-2 latent carry genuinely improves Sudoku, GSM8K, and grasp accuracy.',
    proposedSolution:
      'Inject controlled perturbations into latent carry states in CI and require measurable degradation vs. unperturbed control.',
  },
  {
    number: 64,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'ncm-reasoning',
    title: 'research/feat: NCM-inspired coarse-to-fine adaptive byte diffusion scheduling',
    takeaway:
      'Adapts Next-Concept Modeling (arXiv:2610.14162) from Prompt to Output; notes fixed 8x8 scheduling is only a control baseline, not mandatory.',
    impactOnUnifiedModel:
      'Drives Prompt-to-Output coarse pass at Iter 1 (System 1 exit) and selective System-2 refinement.',
    proposedSolution:
      'Run matched-FLOP A/B benchmarks comparing adaptive entropy chunks vs. learned NCM concepts vs. fixed 8x8 baseline.',
  },
  {
    number: 63,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'ncm-reasoning',
    title: 'Establish canonical x8D Omni Diffusion Language Model definition and integrated algorithm contract',
    takeaway:
      'Separates 4 orthogonal layers: (1) 0..255 byte I/O, (2) internal sequence structure, (3) weight compression, and (4) S1/S2 scheduling.',
    impactOnUnifiedModel:
      'Locks single-model clarity across repo docs, training scripts, and runtime inference.',
    proposedSolution:
      'Keep vocab=256 and direct `f.seek/f.read` `.x8D` execution as the non-negotiable canonical contract.',
  },
  {
    number: 62,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'ncm-reasoning',
    title: 'research/feat: adapt Qwen3.8-Flash-Next to pure-byte x8D with zero-VRAM serving',
    takeaway:
      'Pairs 3:1 KDA recurrent matrix layers with global MLA and Block AttnRes for O(1) inference state.',
    impactOnUnifiedModel:
      'Keeps working KV/latent state tiny enough to run inside <1 GB phone/CPU RAM.',
    proposedSolution:
      'Use 3:1 KDA:MLA layers + Block AttnRes depth routing, trained with Muon on 2D matrices and AdamW on byte embeddings.',
  },
  {
    number: 61,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'sub1bit-0001',
    title: 'bug: x8d_arith (the only fractional-bit coder) is dead code — pack_subbyte_lossless never written',
    takeaway:
      '`x8d_arith.py` implements a valid 256-symbol range coder, yet `x8d_subbyte.py` imports it unused.',
    impactOnUnifiedModel:
      'Activating `x8d_arith` provides real entropy compression over structured factors and byte streams.',
    proposedSolution:
      'Wire `x8d_arith` into 4KB seekable frames inside `x8d_subbyte.py` with strict roundtrip unit tests.',
  },
  {
    number: 60,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'sub1bit-0001',
    title: 'proposal: replace block-mean packer with measurable sub-bit transform + entropy coding',
    takeaway:
      'Shows genuine 0.008 bpw requires structured factorization plus entropy coding over factor coordinates.',
    impactOnUnifiedModel:
      'Establishes the non-destructive path to 0.008 bits per full 8-bit without -1 ternary.',
    proposedSolution:
      'Apply per-channel scaling -> SVD/Kronecker factor decomposition `A x B` in 0.000..0.255 coordinates -> `x8d_arith` range coding.',
  },
  {
    number: 59,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'sub1bit-0001',
    title: "bug: 'disk == source_bytes x 0.001' is a tautological gate — cannot detect information loss",
    takeaway:
      'Size-ratio checks alone passed corrupted checkpoints where weights collapsed to flat block means.',
    impactOnUnifiedModel:
      'Protects model quality with real cosine, SQNR, and logit KL fidelity gates.',
    proposedSolution:
      'Require per-tensor cosine >= 0.990, logit KL <= 0.02, and 100% lossless byte stream roundtrip on export.',
  },
  {
    number: 58,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'sub1bit-0001',
    title: 'bug: BITS_PER_WEIGHT unit error — bits multiplied by value scale (0.008 bit/weight)',
    takeaway:
      'Multiplying byte[i] by 0.001 produces 256 states (still 8 bits of state cardinality, not 0.008 bits).',
    impactOnUnifiedModel:
      'Clarifies that `byte * 0.001` is I/O normalization, whereas 0.008 bpw measures compressed bitstream length / params.',
    proposedSolution:
      'Measure `total_compressed_bits / num_params` separately from `0.000..0.255` input/output coordinate scaling.',
  },
  {
    number: 57,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'sub1bit-0001',
    title: 'bug: .x8D container cannot serve a forward pass — 1000-byte block mean discards per-weight info',
    takeaway:
      '`_stream_pack` collapses 1000 consecutive weights into 1 averaged byte, erasing 999 degrees of freedom per chunk.',
    impactOnUnifiedModel:
      'Root cause of broken forward passes in legacy prototype containers.',
    proposedSolution:
      'Delete 1000-byte mean averaging completely; replace with Kronecker + `x8d_arith` factor spans read directly via `f.seek/f.read`.',
  },
  {
    number: 56,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'training-groot',
    title: 'feat: REAL pure-byte training loop (256-state diffusion, QAT, .x8D export)',
    takeaway:
      'Trains directly on 0..255 bytes with Straight-Through Estimator (STE) QAT and native .x8D export.',
    impactOnUnifiedModel:
      'Necessary bridge to hit 0.008 bpw with high fidelity (since offline PTQ cannot factorize dense full-rank weights).',
    proposedSolution:
      'Embed STE fake-quantization for Kronecker + 0.000..0.255 factor constraints during training and export directly to `.x8D`.',
  },
  {
    number: 55,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'tools-runtime',
    title: 'bug: streaming quantizer dies on TimeoutError after hours — make it resumable + chunked',
    takeaway:
      'Single blocking HTTP stream stalls on large HF checkpoints without range resume.',
    impactOnUnifiedModel:
      'Ensures multi-gigabyte conversions and real-weight benchmarks complete reliably.',
    proposedSolution:
      'Stream via 16 MiB HTTP `Range:` chunks with exponential backoff and atomic `.part` manifests.',
  },
  {
    number: 54,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'training-groot',
    title: 'feat: pure 256-state entropy-bound byte diffusion (uniform corruption, no MASK state)',
    takeaway:
      'Uses uniform random-byte corruption inside 0..255 without synthetic MASK=256 tokens.',
    impactOnUnifiedModel:
      'Keeps every input, intermediate state, and output strictly within 256 raw byte states.',
    proposedSolution:
      'Pair uniform 0..255 noise kernels with self-conditioning carry and entropy-gated Iter-1 commits.',
  },
  {
    number: 52,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'mmap-latent',
    title: 'bug: x8d_export.py writes X8DGGUF1 magic + name-length headers + manifest — violates no-magic contract',
    takeaway:
      'Legacy exporter injected magic strings violating the raw magic-free `.x8D` binary contract.',
    impactOnUnifiedModel:
      'Keeps `.x8D` payload spans pure while indexing byte offsets cleanly for `f.seek`.',
    proposedSolution:
      'Remove `X8DGGUF1` magic strings and move seek offsets into a fixed-width 24-byte tail directory.',
  },
  {
    number: 51,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'sub1bit-0001',
    title: 'bug: 0.001 sub-byte law not applied — containers are 1 byte/param (50% off), not 99.9% smaller',
    takeaway:
      '`quantize_state_dict_to_gguf` stored 1 uint8 per parameter (~8 bpw) instead of true 0.008 bpw.',
    impactOnUnifiedModel:
      'Required fix so disk usage drops 99.9% vs uint8 (957 MB for 1T params).',
    proposedSolution:
      'Route all `quantize_state_dict` exports through the Kronecker + `x8d_arith` sub-bit pipeline with >= 100x compression assertions.',
  },
  {
    number: 50,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'sub1bit-0001',
    title: 'bug: x8D quantization stores raw U8 bytes / scale manifest — 0.001 law not applied',
    takeaway:
      'Mixed Q8 affine scales with manifest metadata and destructive truncation.',
    impactOnUnifiedModel:
      'Distinguishes raw byte normalization (`byte * 0.001`) from continuous parameter normalization (`(w/scale) * 0.001`).',
    proposedSolution:
      'Apply parameter-to-parameter `(w/scale)*0.001` cleanly for continuous weights and compress structured factors via `x8d_arith`.',
  },
  {
    number: 45,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'mmap-latent',
    title: 'research/feat: zero-VRAM, low-RAM from-NVMe serving architecture',
    takeaway:
      'Direct `open("rb") + f.seek/f.read` span loading keeps active RAM <1 GB even at 1T scale without `mmap` page-table bloat.',
    impactOnUnifiedModel:
      'Allows running trillion-scale unified models on standard phones and small CPUs.',
    proposedSolution:
      'Replace `mmap` residency with explicit `f.seek/f.read` into reusable CPU L1/L2 tile buffers.',
  },
  {
    number: 42,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'training-groot',
    title: 'feat: strip legacy modality tokenizers/wrappers; use real serialized bytes only',
    takeaway:
      'Removes BPE, MagViT-v2, and wrapper IDs in favor of raw 0..255 bytes.',
    impactOnUnifiedModel:
      'Unifies text, games, code, audio, and robotics inside one vocab=256 stream.',
    proposedSolution:
      'Feed UTF-8, sensor structs, and actions straight into the shared `0.000..0.255` input projection.',
  },
  {
    number: 37,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'ncm-reasoning',
    title: 'feat(dspark): k-parallel byte blocks + multi-step diffusion prediction/verification',
    takeaway:
      'Parallel k-byte span generation with confidence-gated Iter-1 acceptance.',
    impactOnUnifiedModel:
      'Lets System 1 commit low-entropy spans in 1 pass while System 2 re-noises hard spans.',
    proposedSolution:
      'Emit parallel draft blocks from the NCM coarse pass and commit high-confidence prefixes immediately.',
  },
  {
    number: 36,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'mmap-latent',
    title: 'feat(moe): SARA isolation + route-first zero-VRAM disk expert serving',
    takeaway:
      'Groups routed experts contiguously in `.x8D` so only active experts are read via `f.seek/f.read`.',
    impactOnUnifiedModel:
      'Enables scaling the single unified mixture to 1T parameters while keeping active RAM <1 GB.',
    proposedSolution:
      'Execute router weights first in compact L1/L2 buffer, then seek only active expert byte spans from disk.',
  },
  {
    number: 2,
    kind: 'ISSUE',
    state: 'OPEN',
    category: 'training-groot',
    title: 'feat: enforce pure raw-byte model domain (exactly 256 states, 0-255)',
    takeaway:
      'Core invariant replacing 50k+ BPE vocabularies with 256 universal byte states.',
    impactOnUnifiedModel:
      'Allows Prompt-to-Output NCM efficiency across language, math, Sudoku, Snake, and robotics.',
    proposedSolution:
      'Lock `vocab_size=256` across config, dataloaders, and LM heads with zero out-of-range IDs.',
  },
  {
    number: 41,
    kind: 'ISSUE',
    state: 'CLOSED',
    category: 'mmap-latent',
    title: 'research(feat): tiered disk/NVMe entropy streaming telemetry extraction',
    takeaway:
      'Showed entropy-coded disk weights can serve large MoE models far below full FP16 RAM.',
    impactOnUnifiedModel:
      'Validates sub-byte disk streaming while x8D replaces mmap with leaner direct `f.seek/f.read`.',
    proposedSolution:
      'Closed reference study: adopt entropy-coded disk spans + drop mmap in favor of `f.seek/f.read`.',
  },
  {
    number: 3,
    kind: 'ISSUE',
    state: 'CLOSED',
    category: 'sub1bit-0001',
    title: 'feat: x8Dsub-byte 0.001 sub-byte weight compression + raw export prototype',
    takeaway:
      'Initial prototype establishing raw magic-free container goals and the 0.001 sub-byte direction.',
    impactOnUnifiedModel:
      'Ancestor concept superseded by real structural factorization + `x8d_arith` (#57–#61).',
    proposedSolution:
      'Closed ancestor issue: fully superseded by #56 (QAT) and #57–#61 (Kronecker + `x8d_arith`).',
  },
];

export interface ExecutionGatePhase {
  phase: string;
  title: string;
  objective: string;
  exitGate: string;
}

export const UNIFIED_EXECUTION_GATES: ExecutionGatePhase[] = [
  {
    phase: 'Phase 0',
    title: 'Fix #57/#58/#61 Container & Direct f.seek/read',
    objective:
      'Delete 1000:1 block-mean averaging (#57), strip X8DGGUF1 headers (#52), wire dead x8d_arith.py (#61), and serve spans via open("rb") + f.seek/read.',
    exitGate: '100% lossless byte roundtrip + cosine >= 0.990 on Kronecker+arith tiles with <= 20 MB runtime RSS.',
  },
  {
    phase: 'Phase 1',
    title: 'Train Unified S1 + S2 0..255 Model with #56 QAT',
    objective:
      'Jointly train NCM Prompt-to-Output coarse pass, System 1 decision expert (games, robotics, PR #74), and System 2 diffusion carry under STE sub-bit constraints.',
    exitGate: 'Iter-1 System 1 exit <= 5 ms; causal perturbation test (#65) proves latent carry improves System 2 accuracy.',
  },
  {
    phase: 'Phase 2',
    title: 'Verify 0.008 bpw .x8D Scaling (50M -> 1T <1 GB RAM)',
    objective:
      'Export QAT weights into 0.008 bpw .x8D with SARA route-first seek directory (#36) and benchmark real perplexity/accuracy vs BF16, INT8, INT4, and .cact.',
    exitGate: '1.00T .x8D < 1 GB disk and < 1 GB working RAM on CPU/phone with zero -1 ternary weights.',
  },
];
