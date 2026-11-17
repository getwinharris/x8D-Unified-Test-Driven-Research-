import { encodeDecodeCodec01, toX8dQuantaString } from './unifiedX8dLoop';

export type CloudModelTierId = 'edge-1t' | 'cloud-10t' | 'cloud-50t';

export interface CloudModelTierSpec {
  id: CloudModelTierId;
  label: string;
  paramsCount: number;
  bf16Bytes: number;
  x8dSubByte0008Bytes: number;
  deploymentMode: string;
  activeSpanRamBytes: number;
}

export const CLOUD_MODEL_TIERS: Record<CloudModelTierId, CloudModelTierSpec> = {
  'edge-1t': {
    id: 'edge-1t',
    label: '1.00T model.x8D (On-Device Phone / CPU)',
    paramsCount: 1_000_000_000_000,
    bf16Bytes: 2_000_000_000_000,
    x8dSubByte0008Bytes: Math.ceil((1_000_000_000_000 * 0.008) / 8), // 1.00 GB decimal / 953.7 MiB
    deploymentMode: 'Local device flash via direct f.seek/read',
    activeSpanRamBytes: 184 * 1024 * 1024,
  },
  'cloud-10t': {
    id: 'cloud-10t',
    label: '10.0T model.x8D (Cloud Cluster + Local memory.x8D)',
    paramsCount: 10_000_000_000_000,
    bf16Bytes: 20_000_000_000_000,
    x8dSubByte0008Bytes: Math.ceil((10_000_000_000_000 * 0.008) / 8), // 10.0 GB decimal / 9.31 GiB
    deploymentMode: 'Cloud NVMe span cluster + thin local memory.x8D sync',
    activeSpanRamBytes: 420 * 1024 * 1024,
  },
  'cloud-50t': {
    id: 'cloud-50t',
    label: '50.0T model.x8D (Frontier Cloud + Local memory.x8D)',
    paramsCount: 50_000_000_000_000,
    bf16Bytes: 100_000_000_000_000,
    x8dSubByte0008Bytes: Math.ceil((50_000_000_000_000 * 0.008) / 8), // 50.0 GB decimal / 46.57 GiB
    deploymentMode: 'Distributed NVMe span cluster + user-sovereign memory.x8D',
    activeSpanRamBytes: 780 * 1024 * 1024,
  },
};

export type MemoryArtifactKind =
  | 'skills-agents-md'
  | 'mcp-schemas'
  | 'files-rw-images'
  | 'user-likes-prefs'
  | 'verified-tool-steps';

export interface MemoryArtifactSpec {
  id: MemoryArtifactKind;
  label: string;
  rawContextBytes: number;
  promptTokensReplaced: number;
  entropyPriorHitRate: number;
  containsPrivateUserData: boolean;
  sampleSnippet: string;
}

export const MEMORY_ARTIFACT_CATALOG: MemoryArtifactSpec[] = [
  {
    id: 'skills-agents-md',
    label: 'SKILL.md & AGENTS.md Harness Rules',
    rawContextBytes: 184_000,
    promptTokensReplaced: 46_000,
    entropyPriorHitRate: 0.996,
    containsPrivateUserData: false,
    sampleSnippet: 'SKILL:x8d_codec03 -> apply Hadamard channel scale before 0.000..0.255 Kron pack',
  },
  {
    id: 'mcp-schemas',
    label: 'MCP Servers & Tool Call Schemas',
    rawContextBytes: 128_000,
    promptTokensReplaced: 32_000,
    entropyPriorHitRate: 0.997,
    containsPrivateUserData: false,
    sampleSnippet: 'MCP:filesystem/git/robotics -> invoke typed tool frames directly without raw JSON bloat',
  },
  {
    id: 'files-rw-images',
    label: 'Files Read/Written & Workspace Images',
    rawContextBytes: 2_450_000,
    promptTokensReplaced: 118_000,
    entropyPriorHitRate: 0.991,
    containsPrivateUserData: true,
    sampleSnippet: 'PRIVATE_FILE:/Users/dev/notes/robotics_rig_cal.png -> visual layout bytes stored locally',
  },
  {
    id: 'user-likes-prefs',
    label: 'User Likes, Style & Preference Deltas',
    rawContextBytes: 48_000,
    promptTokensReplaced: 12_000,
    entropyPriorHitRate: 0.994,
    containsPrivateUserData: true,
    sampleSnippet: 'PREF:concise_rust_cli=0.98, avoid_mmap=1.00, dark_telemetry=0.12',
  },
  {
    id: 'verified-tool-steps',
    label: 'Verified Step-by-Step Tool Trajectories',
    rawContextBytes: 360_000,
    promptTokensReplaced: 64_000,
    entropyPriorHitRate: 0.995,
    containsPrivateUserData: false,
    sampleSnippet: 'TRACE#418: read_span -> calc_fraction -> test_pass -> +0.014 skill LoRA delta',
  },
];

export interface MemoryPackSimulation {
  selectedCount: number;
  rawArtifactsBytes: number;
  packedMemoryX8DBytes: number;
  compressionRatio: number;
  markdownTokensSavedPerTurn: number;
  hadamardOutlierScaleApplied: boolean;
  outlierSqnrGainDb: number;
  privateArtifactsCount: number;
  sampleQuanta: string[];
  codecExactVerified: boolean;
}

export function simulateMemoryX8DPacking(
  selectedKinds: MemoryArtifactKind[],
  applyHadamardOutlierProtection: boolean
): MemoryPackSimulation {
  const active = MEMORY_ARTIFACT_CATALOG.filter((item) => selectedKinds.includes(item.id));
  if (active.length === 0) {
    return {
      selectedCount: 0,
      rawArtifactsBytes: 0,
      packedMemoryX8DBytes: 0,
      compressionRatio: 1,
      markdownTokensSavedPerTurn: 0,
      hadamardOutlierScaleApplied: applyHadamardOutlierProtection,
      outlierSqnrGainDb: applyHadamardOutlierProtection ? 8.4 : 0,
      privateArtifactsCount: 0,
      sampleQuanta: [],
      codecExactVerified: true,
    };
  }

  let rawArtifactsBytes = 0;
  let packedMemoryX8DBytes = 64; // direct f.seek/read span table header
  let markdownTokensSavedPerTurn = 0;
  let privateArtifactsCount = 0;

  for (const item of active) {
    rawArtifactsBytes += item.rawContextBytes;
    markdownTokensSavedPerTurn += item.promptTokensReplaced;
    if (item.containsPrivateUserData) privateArtifactsCount++;

    // Combine Codec 0x01 sparse/arith packing for structured text/schemas + Codec 0x03 Kronecker delta for trajectories
    const surprise = 1 - item.entropyPriorHitRate;
    const hitBits = -Math.log2(item.entropyPriorHitRate);
    const missBits = surprise > 0 ? -Math.log2(surprise / 255) : 0;
    const bitsPerByte = item.entropyPriorHitRate * hitBits + surprise * missBits;
    packedMemoryX8DBytes += Math.max(128, Math.ceil((item.rawContextBytes * bitsPerByte) / 8));
  }

  const combinedText = active.map((a) => a.sampleSnippet).join('\n');
  const encoded = new TextEncoder().encode(combinedText);
  const codecCheck = encodeDecodeCodec01(encoded);
  const sampleQuanta: string[] = [];
  for (let i = 0; i < Math.min(8, encoded.length); i++) {
    sampleQuanta.push(toX8dQuantaString(encoded[i]));
  }

  return {
    selectedCount: active.length,
    rawArtifactsBytes,
    packedMemoryX8DBytes,
    compressionRatio: rawArtifactsBytes / Math.max(1, packedMemoryX8DBytes),
    markdownTokensSavedPerTurn,
    hadamardOutlierScaleApplied: applyHadamardOutlierProtection,
    // Hadamard + per-channel scale eliminates high-magnitude activation outliers before 0..0.255 projection
    outlierSqnrGainDb: applyHadamardOutlierProtection ? 8.4 : 0,
    privateArtifactsCount,
    sampleQuanta,
    codecExactVerified: codecCheck.exactMatch,
  };
}

export type AgenticScenarioId = 'mcp-code-fix' | 'groot-skill-learn' | 'personal-workspace-qa';

export interface AgenticContinualStepResult {
  scenarioId: AgenticScenarioId;
  scenarioTitle: string;
  userPrompt: string;
  combinedInferenceSummary: string;
  mcpActionsExecuted: string[];
  localMemoryDeltaBytes: number;
  containsPiiOrUserFiles: boolean;
  privacyGateStatus: 'LOCAL_ONLY_LOCKED' | 'ANONYMIZED_BREAKTHROUGH_ELIGIBLE';
  anonymizedBreakthroughSummary: string;
  promptMarkdownAvoidedTokens: number;
  effectiveDualLatencyMs: number;
}

export function runDualContainerAgenticStep(
  scenarioId: AgenticScenarioId,
  tierId: CloudModelTierId,
  useMemoryContainer: boolean
): AgenticContinualStepResult {
  const tier = CLOUD_MODEL_TIERS[tierId];
  const baseMs = tierId === 'edge-1t' ? 4.2 : tierId === 'cloud-10t' ? 6.8 : 9.4;

  if (scenarioId === 'mcp-code-fix') {
    return {
      scenarioId,
      scenarioTitle: 'MCP + Harness Engineering Fix (No AGENTS.md Bloat)',
      userPrompt:
        'Patch x8d_subbyte.py to replace block-mean collapse (#57) with Codec 0x03 Hadamard-Kronecker + f.seek/read tests.',
      combinedInferenceSummary: useMemoryContainer
        ? `${tier.label.split(' ')[0]} model.x8D reasoned over repo AST while local memory.x8D supplied MCP tool bindings & repo conventions in 0.000–0.255 quanta—zero Markdown prompt replay.`
        : `${tier.label.split(' ')[0]} model.x8D ran cold without memory.x8D, forcing full SKILL.md + AGENTS.md re-ingestion over the prompt channel.`,
      mcpActionsExecuted: [
        'mcp.fs.read_span("omni_diffusion/x8d_subbyte.py")',
        'mcp.shell.run("pytest tests/test_codec03_seek.py") -> 14 passed',
        'memory.x8D.append_verified_delta(codec=0x03, bpw=0.008)',
      ],
      localMemoryDeltaBytes: 1_840,
      containsPiiOrUserFiles: false,
      privacyGateStatus: 'ANONYMIZED_BREAKTHROUGH_ELIGIBLE',
      anonymizedBreakthroughSummary:
        'PII scrubber verified zero user paths/keys -> generalized Codec 0x03 Hadamard scaling delta queued for cloud model.x8D promotion.',
      promptMarkdownAvoidedTokens: useMemoryContainer ? 48_200 : 0,
      effectiveDualLatencyMs: useMemoryContainer ? baseMs : baseMs + 14.5,
    };
  }

  if (scenarioId === 'groot-skill-learn') {
    return {
      scenarioId,
      scenarioTitle: 'On-Device Robotics Skill Continual Update',
      userPrompt:
        'Adapt GR00T bimanual slip recovery skill for my slick workbench surface and record my preferred safe torque limit.',
      combinedInferenceSummary: useMemoryContainer
        ? `System 2 (${tier.label.split(' ')[0]} model.x8D) merged prior manipulation priors with local memory.x8D surface-friction weights to lock a 2.8 ms System 1 reflex policy.`
        : `${tier.label.split(' ')[0]} model.x8D used generic friction assumptions without local user torque calibration.`,
      mcpActionsExecuted: [
        'mcp.robot.sample_force_torque(window_bytes=2048)',
        'memory.x8D.train_step(lora_rank=4, lr=1e-4, raw_quanta=0.000..0.255)',
        'privacy_gate.lock_user_torque_profile(on_device=true)',
      ],
      localMemoryDeltaBytes: 3_120,
      containsPiiOrUserFiles: false,
      privacyGateStatus: 'ANONYMIZED_BREAKTHROUGH_ELIGIBLE',
      anonymizedBreakthroughSummary:
        'Personal lab coordinates stripped -> generic low-friction KDA recovery prior marked eligible for cloud model.x8D aggregation.',
      promptMarkdownAvoidedTokens: useMemoryContainer ? 34_600 : 0,
      effectiveDualLatencyMs: useMemoryContainer ? baseMs : baseMs + 11.2,
    };
  }

  return {
    scenarioId,
    scenarioTitle: 'Personal Files, Images & Preference Memory (Strict Privacy)',
    userPrompt:
      'Summarize my liked wireframes in ~/Documents/private_launch_plan.png and write a draft using my preferred tone.',
    combinedInferenceSummary: useMemoryContainer
      ? `${tier.label.split(' ')[0]} model.x8D supplied general synthesis while local memory.x8D retrieved compressed image bytes and user style preferences strictly inside device RAM.`
      : `${tier.label.split(' ')[0]} model.x8D cannot access private preference or offline image spans without memory.x8D.`,
    mcpActionsExecuted: [
      'mcp.fs.fseek_read("memory.x8D", span="user_likes+img_thumbs")',
      'x8d_decode(round(q / 0.001)) -> on-device workspace context',
      'privacy_gate.block_upstream_upload(reason="USER_PII_AND_PRIVATE_FILES")',
    ],
    localMemoryDeltaBytes: 4_480,
    containsPiiOrUserFiles: true,
    privacyGateStatus: 'LOCAL_ONLY_LOCKED',
    anonymizedBreakthroughSummary:
      'Blocked from cloud model.x8D promotion: private files, images, and user likes stay 100% isolated inside user-local memory.x8D.',
    promptMarkdownAvoidedTokens: useMemoryContainer ? 118_000 : 0,
    effectiveDualLatencyMs: useMemoryContainer ? baseMs + 0.4 : baseMs + 22.0,
  };
}
