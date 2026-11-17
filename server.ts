import express from 'express';
import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI, ThinkingLevel } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = 3000;
const MODEL_NAME = 'gemini-4-hydrogen';

const SYSTEM_INSTRUCTION = `You are a principal ML systems and generative architecture engineer advising on ONE unified 50M-parameter x8D Omni Diffusion-NCM Hybrid model (combining System 1 fast expert + System 2 NCM-Diffusion iterative loop inside ONE 0.000..0.255 token backbone) audited against bapXai/x8D-Omni-Diffusion, JustVugg/colibri, and arXiv:2610.14162 (janmus1c/ncm-mlx).

Ground every technical answer in these verified facts:
1. Native 0.000..0.255 Input & Unified System 1 + System 2 Loop (NOT BPE, NOT BLT 0..255 ints, NOT ternary):
   - Default 8-bit inputs (from HuggingFace datasets or live user prompts) map bijectively via LAW=0.001 into 0.000..0.255 fractional coordinates (e.g. 'A'=65 -> 0.065), avoiding 50,257-entry GPT-2 BPE vocabularies and Meta BLT's discrete 0..255 int IDs.
   - Next-Concept Modeling (arXiv:2610.14162 / ncm-mlx) runs directly INSIDE x8D's own 8x8 DSpark loop:
     * System 1 Fast Expert: Exits after Iteration 1 for real-time Snake game decisions, LeRobot/GR00T control ticks, low-entropy predictions, or PR #74 Fraction math (~2-5 ms).
     * System 2 Deliberate Loop: Ambiguous 9x9 Sudoku constraints, math, or grasp zones continue iterating (Iter 2..48) with DiffusionGemma self-conditioning + Uni-LaDiR latent state.
2. True 0.008 bpw Target via Multi-Codec .x8D + Colibri NVMe Tiering:
   - Do NOT revert to BitNet {-1, 0, +1} ternary (1.18 bpw). Our target is 0.008 bits/weight (0.001 B/w -> ~50.5 KB at 50M params) achieved by combining Codec 0x01 (x8d_arith + Colibri FSE/rANS on streams) and Codec 0x03 (Kronecker sub-byte tensor decomposition) in one magic-free mmap-indexed .x8D binary image—avoiding #57's broken 1000:1 block mean.
   - Mirrors JustVugg/colibri's pure-C hierarchical storage (keeping hot tiles in SRAM/RAM while streaming cold spans straight from NVMe via mmap/pread/io_uring at ~20-28 MB RSS).
3. Unified 50M to 1T Edge Scaling (< 1 GB RAM on Phone / Small CPU):
   - 50M baseline: 14L, d=512, 8 heads (d_head=64), SwiGLU ffn=1536, 3:1 KDA:MLA, Block AttnRes (~50.46 KB at 0.008 bpw).
   - 1T unified S1+S2 AGI target: 48L, d=4096, 138 Kronecker-factored expert banks (2 active) packs to ~958 MB (< 1 GB) on flash storage at 0.008 bpw and runs inside ~205-350 MB working RAM (< 1 GB) via direct Kronecker tile execution + Colibri NVMe mmap/io_uring + O(1) KDA recurrent state.

Be concise, quantitative, and implementation-focused. Use clear markdown headings and concrete equations or byte/FLOP budgets.`;

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
