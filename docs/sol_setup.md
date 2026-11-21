# ASU SOL High-Performance Computing (HPC) Setup Guide

This guide documents the procedure for executing the PlotGraph **Phase 3D Qwen2.5-7B-Instruct 30-Movie Pilot** on Arizona State University's SOL supercomputer using the known-working scheduler configuration and the existing `vllm312` GPU environment.

---

## 1. Known-Working SLURM Configuration

The batch script (`scripts/sol/kg_l_qwen_pilot.sbatch`) uses the verified SLURM parameters for this account:

| Directive | Value | Purpose |
|---|---|---|
| `#SBATCH -p general` | `general` | Partition |
| `#SBATCH -q class` | `class` | Quality of Service (QOS) |
| `#SBATCH -G 1` | `1` | Request 1 GPU |
| `#SBATCH --nodes=1` | `1` | 1 physical compute node |
| `#SBATCH --ntasks=1` | `1` | 1 task |
| `#SBATCH --cpus-per-task=4` | `4` | 4 CPU cores for data loading and vLLM workers |
| `#SBATCH --mem=32G` | `32 GB` | Host RAM |
| `#SBATCH --time=01:00:00` | `1 hour` | Wallclock limit (30 movies take ~2–4 minutes) |

> **Note**: Do not assume speculative account directives, other partitions (e.g., `gpu`), or other queues (e.g., `wildfire`).

---

## 2. Environment: Existing `vllm312` Conda Environment

This account on ASU SOL already has a verified, working Conda environment named **`vllm312`** with a fully compatible GPU stack. **Do not create another GPU environment.**

### Environment Specifications
- **Conda Environment**: `vllm312`
- **Python Version**: `3.12.12`
- **vLLM**: `0.10.2+cu129`
- **PyTorch**: `2.8.0`
- **CUDA Build**: `12.9`
- **PlotGraph**: Installed in editable mode (`pip install -e .`) with all unit tests passing.

The batch script automatically activates `vllm312` by default. You can override the target environment if desired by exporting `PLOTGRAPH_CONDA_ENV`:
```bash
export PLOTGRAPH_CONDA_ENV="vllm312"
```

---

## 3. Interactive Connection & Verification Workflow

### Step 3.1: Connect to ASU SOL via SSH

```bash
ssh <ASURITE>@sol.asu.edu
```

### Step 3.2: Clone or Sync the Repository

Clone or pull the latest repository branch on scratch storage:
```bash
cd /scratch/${USER}
git clone https://github.com/<your-repo>/plotgraph-movie-recommender.git
cd plotgraph-movie-recommender
```

### Step 3.3: Activate Existing `vllm312` Environment

Initialize Conda from your Miniconda installation and activate `vllm312`:
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm312
```

If you ever need to re-install or update the repository in editable mode within this environment:
```bash
pip install -e .
```

### Step 3.4: Configure Hugging Face Cache on Scratch

Model weights for `Qwen/Qwen2.5-7B-Instruct` are ~15 GB. **Do not download them to your home directory (`~/.cache`)**, as `$HOME` has strict storage quotas.

Set your cache to `/scratch/${USER}`:
```bash
export SCRATCH_DIR="/scratch/${USER}"
export HF_HOME="${SCRATCH_DIR}/.cache/huggingface"
mkdir -p "${HF_HOME}"
```
*(The batch script `scripts/sol/kg_l_qwen_pilot.sbatch` also exports this automatically.)*

### Step 3.5: Verify Python & GPU Stack Interactively

Before submitting a batch job, verify that PyTorch, CUDA detection, and vLLM import cleanly inside the `vllm312` environment:
```bash
python -c "
import sys
print('Python:', sys.version.split()[0])
import torch
print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())
import vllm
print('vLLM:', vllm.__version__)
"
```

---

## 4. Submitting the SLURM Pilot Jobs

### 4.1 Phase 3D Semantic+Emotion Pilot (Combined)
Submit the semantic pilot batch script:
```bash
sbatch scripts/sol/kg_l_qwen_pilot.sbatch
```

### 4.2 Phase 3E Dedicated Emotion Pilot (Evidence-Grounded)
Submit the dedicated emotion extraction batch script:
```bash
sbatch scripts/sol/kg_l_qwen_emotion_pilot.sbatch
```

The scheduler will print the job ID:
```
Submitted batch job <JOB_ID>
```

---

## 5. Monitoring the Job

### Check Queue Status
```bash
squeue -u ${USER}
```

### Tail the Output Logs
Logs are directed to `logs/sol/`:
```bash
# Follow stdout in real time
tail -f logs/sol/plotgraph_qwen_pilot_<JOB_ID>.out

# Inspect stderr if needed
cat logs/sol/plotgraph_qwen_pilot_<JOB_ID>.err
```

---

## 6. Reviewing Pilot Artifacts

When the job completes, verify the artifacts generated in `data/processed/ml-1m/kg_l/pilot_v1_1/` (or `pilot/` if running v1.0.0):

| Artifact | Format | Description |
|---|---|---|
| `raw_outputs.jsonl` | JSON Lines | Complete log of every raw response from Qwen for every attempt. |
| `validated_extractions.jsonl` | JSON Lines | Strictly validated `MoviePlotExtraction` records complying with the Phase 3C contract. |
| `failures.jsonl` | JSON Lines | Records of any extractions that exhausted retries, categorized by error type. |
| `pilot_summary.json` | JSON | Summary metrics: total sample count (30), success count, retries, runtime, and success rate. |
| `pilot_review.csv` | CSV | Human-readable inspection sheet with extracted themes, settings, devices, protagonists, source materials, continuous Plutchik emotions, and prominent emotions (`score > 0.5`). |

### Comparing Pilot Prompt Versions (v1.0.0 vs v1.1.0)
To compare completed pilot runs side by side across schema validity, continuous emotion distributions, prominent emotion frequencies, and semantic entity counts:
```bash
python scripts/compare_pilots.py \
    --dir-a data/processed/ml-1m/kg_l/pilot \
    --dir-b data/processed/ml-1m/kg_l/pilot_v1_1 \
    --name-a "Prompt v1.0.0" \
    --name-b "Prompt v1.1.0"
```

### Quick Verification Commands
```bash
# Check summary metrics for v1.1 pilot
cat data/processed/ml-1m/kg_l/pilot_v1_1/pilot_summary.json

# Check first few rows of the human review table
head -n 5 data/processed/ml-1m/kg_l/pilot_v1_1/pilot_review.csv

# Verify 30 successfully validated records
wc -l data/processed/ml-1m/kg_l/pilot_v1_1/validated_extractions.jsonl

# Confirm zero unhandled failures
wc -l data/processed/ml-1m/kg_l/pilot_v1_1/failures.jsonl
```

### 6.2 Reviewing Phase 3E Dedicated Emotion Artifacts

When the dedicated emotion job completes, verify the artifacts generated in `data/processed/ml-1m/kg_l/emotion_pilot_v1/`:

| Artifact | Format | Description |
|---|---|---|
| `raw_emotion_outputs.jsonl` | JSON Lines | Complete log of every raw response from Qwen for every emotion attempt. |
| `validated_emotion_evidence.jsonl` | JSON Lines | Strictly validated `MovieEmotionExtraction` records with text evidence and calibrated scores. |
| `emotion_vectors.jsonl` | JSON Lines | Derived clean 8-dimensional emotion vectors (`movie_id`, `movie_idx`, `title`, 8 emotion scores, `prominent_emotions`) ready for downstream KG-L graph building. |
| `emotion_failures.jsonl` | JSON Lines | Records of any emotion extractions that exhausted retries. |
| `emotion_pilot_summary.json` | JSON | Summary metrics: sample count (30), success count, retries, runtime, and success rate. |
| `emotion_review.csv` | CSV | Human-readable inspection sheet with overview, per-emotion evidence lists, continuous scores, and prominent emotions (`score > 0.5`). |

### 6.3 Three-Way Side-by-Side Emotion Comparison
To compare emotion distributions and grounding across Combined v1.0, Combined v1.1, and Dedicated Emotion v1.0:
```bash
python scripts/compare_emotion_pilots.py \
    --dir-v1-0 data/processed/ml-1m/kg_l/pilot \
    --dir-v1-1 data/processed/ml-1m/kg_l/pilot_v1_1 \
    --dir-emotion-v1 data/processed/ml-1m/kg_l/emotion_pilot_v1
```

---

## 7. Local Development & Mock Testing (macOS / No GPU)

For local testing without GPU or model downloads:
```bash
# Run Phase 3D semantic pilot with default refined prompt (v1.1.0):
python scripts/run_phase3d_qwen_pilot.py --backend mock

# Or explicitly select prompt version v1.0.0 outputting to pilot:
python scripts/run_phase3d_qwen_pilot.py --backend mock --prompt-version v1.0.0

# Run Phase 3E dedicated emotion pilot with evidence-grounded mock backend:
python scripts/run_phase3e_emotion_pilot.py --backend mock
```
This runs the complete safety guard, checkpointing, retry logic, summary generation, derived vector export, and review CSV export locally using mock responses.
