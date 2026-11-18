import { fetchAllOmniDatasets } from '../src/lib/omniDatasetIngestor';

const grootEnvs = Number(process.env.GROOT_ENVS ?? 6);
const episodesPerEnv = Number(process.env.GROOT_EPISODES ?? 4);
const rowsPerDataset = Number(process.env.HF_ROWS ?? 400);

console.log(`Fetching real HF Omni datasets (GR00T envs=${grootEnvs}, eps/env=${episodesPerEnv}, rows=${rowsPerDataset})...`);

fetchAllOmniDatasets(
  { grootEnvs, episodesPerEnv, rowsPerDataset },
  (msg) => console.log(`[omni-fetch] ${msg}`)
)
  .then((manifest) => {
    console.log('\n==============================================================');
    console.log(`INGESTED REAL OMNI CORPUS: ${(manifest.totalTrainBytes / 1024).toFixed(1)} KB train | ${(manifest.totalHeldOutBytes / 1024).toFixed(1)} KB held-out`);
    for (const d of Object.values(manifest.domains)) {
      console.log(
        ` - ${d.id.padEnd(20)} | train=${d.trainBytes.toLocaleString().padStart(9)} B (${d.recordsTrain} recs) | held=${d.heldOutBytes.toLocaleString().padStart(8)} B`
      );
    }
  })
  .catch((err) => {
    console.error('Omni dataset ingestion failed:', err);
    process.exit(1);
  });
