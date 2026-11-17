import express from 'express';
import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI, ThinkingLevel } from '@google/genai';
import { runNeedle3RealBenchmark } from './src/lib/needle3BenchmarkEngine';
import { NEEDLE3_TENSOR_SPECS, Needle3BenchmarkReport, Needle3TensorId } from './src/lib/needle3Types';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = 3000;
const MODEL_NAME = 'gemini-4-hydrogen';

const SYSTEM_INSTRUCTION = `You are a principal ML systems engineer advising on ONE unified x8D Omni Diffusion-NCM Hybrid model audited directly against bapXai/x8D-Omni-Diffusion and arXiv:2610.14162 (NCM).

Ground every technical answer strictly in these verified facts (never invent extra codecs, side products, or mmap layers):
1. Native 0.000..0.255 Prompt-to-Output Flow + Unified System 1 Decision Expert Mixture:
   - Default 8-bit inputs (from prompts, text, code, math, QA, games, audio, and robotics) map bijectively via byte * 0.001 into 0.000..0.255 states (vocab_size = 256, no BPE, no MASK=256 state), and decode back via / 0.001 into language/actions.
   - Next-Concept Modeling (NCM, arXiv:2610.14162) runs directly inside x8D's diffusion loop from Prompt to Output.
   - System 1 Fast Decision Expert lives INSIDE the unified mixture alongside AR/code/math/QA capabilities: confident game ticks, robot actions, and PR #74 Fraction math exit at Iteration 1, while high-entropy reasoning/constraints continue iterating inside the same shared KDA/MLA backbone.
2. Sub-1-Bit .x8D Disk Target (0.008 bits per 8-bit / 0.001 sub-byte target) & Direct f.seek/read (No mmap):
   - Zero {-1, 0, +1} ternary. Avoid #57's broken 1000:1 block-mean averaging and #52 magic headers.
   - For pretrained continuous floats (like Cactus-Compute/needle3), parameter-to-parameter scaling maps each weight individually (w * 0.001) rather than clamping to 0..255, while true 0.008 bpw requires structural factorization (e.g. Kronecker) + QAT/STE (#56) + x8d_arith range coding (#61).
   - Runtime streams active spans directly from the compressed .x8D file via open("rb") + f.seek/f.read into CPU cache tiles without OS mmap, keeping active working RAM well under 1 GB on phone or standard CPU.`;

function createGeminiClient(): GoogleGenAI | null {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') {
    return null;
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '1mb' }));

  app.get('/api/health', (_req, res) => {
    const hasKey = Boolean(process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY !== 'MY_GEMINI_API_KEY');
    res.json({
      status: 'ok',
      model: MODEL_NAME,
      thinkingLevel: 'HIGH',
      geminiConfigured: hasKey,
    });
  });

  const needle3Cache = new Map<Needle3TensorId, Needle3BenchmarkReport>();
  app.get('/api/bench/needle3', (req, res) => {
    const rawTensor = typeof req.query.tensor === 'string' ? req.query.tensor : 'self_attn_out_proj';
    const tensorId: Needle3TensorId =
      rawTensor in NEEDLE3_TENSOR_SPECS ? (rawTensor as Needle3TensorId) : 'self_attn_out_proj';
    const forceRefresh = req.query.refresh === '1';

    try {
      let report = !forceRefresh ? needle3Cache.get(tensorId) : undefined;
      if (!report) {
        report = runNeedle3RealBenchmark(tensorId);
        needle3Cache.set(tensorId, report);
      }
      res.json(report);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to run Cactus-Compute/needle3 benchmark.';
      res.status(404).json({ error: msg });
    }
  });

  app.get('/api/hf/sample', async (req, res) => {
    const ALLOWED_DATASETS: Record<string, { config: string; split: string; mode: 'rows' | 'tree' }> = {
      'lerobot/pusht': { config: 'default', split: 'train', mode: 'rows' },
      'lerobot/aloha_sim_insertion_human': { config: 'default', split: 'train', mode: 'rows' },
      'openai/gsm8k': { config: 'main', split: 'train', mode: 'rows' },
      'roneneldan/TinyStories': { config: 'default', split: 'train', mode: 'rows' },
      'Salesforce/wikitext': { config: 'wikitext-2-raw-v1', split: 'train', mode: 'rows' },
      'nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim': { config: 'default', split: 'train', mode: 'tree' },
    };

    const dataset = typeof req.query.dataset === 'string' ? req.query.dataset : 'lerobot/pusht';
    const rowIdxRaw = Number(req.query.row ?? 0);
    const rowIdx = Number.isFinite(rowIdxRaw) ? Math.max(0, Math.min(99, Math.floor(rowIdxRaw))) : 0;
    const spec = ALLOWED_DATASETS[dataset];

    if (!spec) {
      res.status(400).json({ error: `Unsupported dataset preset: ${dataset}` });
      return;
    }

    try {
      if (spec.mode === 'tree') {
        const treeUrl = `https://huggingface.co/api/datasets/${dataset}/tree/main`;
        const hfResp = await fetch(treeUrl, { headers: { 'User-Agent': 'x8d-workbench/1.0' } });
        if (!hfResp.ok) {
          throw new Error(`HuggingFace Hub API returned HTTP ${hfResp.status}`);
        }
        const entries = (await hfResp.json()) as Array<{ path: string; type: string; oid?: string }>;
        const tasks = entries.filter((e) => e.type === 'directory').map((e) => e.path);
        const selectedTask = tasks[rowIdx % Math.max(1, tasks.length)] || 'bimanual_panda_gripper.Threading';
        // Also fetch file listing inside the selected GR00T simulation task folder
        const taskUrl = `https://huggingface.co/api/datasets/${dataset}/tree/main/${encodeURIComponent(selectedTask)}`;
        const taskResp = await fetch(taskUrl, { headers: { 'User-Agent': 'x8d-workbench/1.0' } });
        const taskFiles = taskResp.ok
          ? ((await taskResp.json()) as Array<{ path: string; size?: number }>).slice(0, 8)
          : [];
        const payloadObj = {
          dataset,
          taskEnvironment: selectedTask,
          availableTasksCount: tasks.length,
          sampleTasks: tasks.slice(0, 10),
          taskTreeArtifacts: taskFiles,
        };
        res.json({
          dataset,
          rowIndex: rowIdx % Math.max(1, tasks.length),
          sourceEndpoint: treeUrl,
          rawRecord: payloadObj,
          serializedText: JSON.stringify(payloadObj),
        });
        return;
      }

      const hfUrl = `https://datasets-server.huggingface.co/first-rows?dataset=${encodeURIComponent(dataset)}&config=${encodeURIComponent(spec.config)}&split=${encodeURIComponent(spec.split)}`;
      const hfResp = await fetch(hfUrl, { headers: { 'User-Agent': 'x8d-workbench/1.0' } });
      if (!hfResp.ok) {
        throw new Error(`HuggingFace Datasets Server returned HTTP ${hfResp.status}`);
      }
      const data = (await hfResp.json()) as { rows?: Array<{ row_idx: number; row: unknown }> };
      const rows = data.rows ?? [];
      if (rows.length === 0) {
        throw new Error('No rows returned from HuggingFace dataset.');
      }
      const picked = rows[rowIdx % rows.length];
      res.json({
        dataset,
        rowIndex: picked.row_idx,
        totalCachedRows: rows.length,
        sourceEndpoint: hfUrl,
        rawRecord: picked.row,
        serializedText: JSON.stringify(picked.row),
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to fetch live HuggingFace dataset sample.';
      res.status(502).json({ error: msg });
    }
  });

  app.post('/api/research/analyze', async (req, res) => {
    const query = typeof req.body?.query === 'string' ? req.body.query.trim() : '';
    const contextSummary = typeof req.body?.contextSummary === 'string' ? req.body.contextSummary.trim() : '';

    if (!query) {
      res.status(400).json({ error: 'Please enter a technical question or architecture scenario to analyze.' });
      return;
    }

    const ai = createGeminiClient();
    if (!ai) {
      res.status(503).json({
        error:
          'GEMINI_API_KEY is not configured in Settings > Secrets. Add a valid key in the Secrets panel to run live high-thinking architectural analysis.',
      });
      return;
    }

    const prompt = contextSummary
      ? `Active Simulator Configuration:\n${contextSummary}\n\nDeveloper Question:\n${query}`
      : query;

    try {
      const response = await ai.models.generateContent({
        model: MODEL_NAME,
        contents: prompt,
        config: {
          systemInstruction: SYSTEM_INSTRUCTION,
          thinkingConfig: {
            thinkingLevel: ThinkingLevel.HIGH,
          },
        },
      });

      const text = response.text;
      if (!text) {
        res.status(502).json({ error: 'Model returned an empty response. Please try refining your technical prompt.' });
        return;
      }

      res.json({
        model: MODEL_NAME,
        thinkingLevel: 'HIGH',
        analysis: text,
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unexpected server error calling Gemini API.';
      const status = message.includes('403') || message.includes('API_KEY_INVALID')
        ? 403
        : message.includes('429') || message.includes('RESOURCE_EXHAUSTED')
          ? 429
          : 500;
      res.status(status).json({ error: message });
    }
  });

  const isProd = process.env.NODE_ENV === 'production';
  if (!isProd) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('/{*splat}', (_req, res) => {
      const indexHtml = path.join(distPath, 'index.html');
      if (fs.existsSync(indexHtml)) {
        res.sendFile(indexHtml);
      } else {
        res.status(404).send('Build output not found.');
      }
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
