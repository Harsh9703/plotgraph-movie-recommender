# ASU SOL High-Performance Computing (HPC) Setup Guide

This guide documents the procedure for executing the PlotGraph **Phase 3D Qwen2.5-7B-Instruct 30-Movie Pilot** on Arizona State University's SOL supercomputer using the known-working scheduler configuration for this account.

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

## 2. Interactive Environment Inspection & Setup

Do not install packages inside batch jobs. Inspect the actual available SOL environment interactively on the login node or an interactive session before running the batch job.

### Step 2.1: Connect to ASU SOL via SSH

```bash
ssh <ASURITE>@sol.asu.edu
```

### Step 2.2: Clone or Sync the Repository

Clone or pull the latest repository branch on scratch storage:
```bash
cd /scratch/${USER}
git clone https://github.com/<your-repo>/plotgraph-movie-recommender.git
cd plotgraph-movie-recommender
```

### Step 2.3: Inspect Interactive Environment & Set Up Conda

Inspect your existing Miniconda or Conda installation on SOL:
```bash
# Check if conda or miniconda is present in your home directory
which conda
ls -d ~/miniconda3 ~/miniconda ~/.conda 2>/dev/null
```

If you have Miniconda installed, initialize your shell session:
```bash
source ~/miniconda3/etc/profile.d/conda.sh
```

Create and activate the `plotgraph` Conda environment:
```bash
# Create dedicated Python 3.11 environment
conda create -y -n plotgraph python=3.11
conda activate plotgraph

# Upgrade pip
pip install --upgrade pip

# Install project base and GPU dependencies (vLLM and PyTorch with CUDA)
pip install -e ".[gpu]"
```

### Step 2.4: Configure Hugging Face Cache on Scratch

Model weights for `Qwen/Qwen2.5-7B-Instruct` are ~15 GB. **Do not download them to your home directory (`~/.cache`)**, as `$HOME` has strict storage quotas.

Set your cache to `/scratch/${USER}`:
```bash
export SCRATCH_DIR="/scratch/${USER}"
export HF_HOME="${SCRATCH_DIR}/.cache/huggingface"
mkdir -p "${HF_HOME}"
```
*(The batch script `scripts/sol/kg_l_qwen_pilot.sbatch` also exports this automatically.)*

### Step 2.5: Verify Python & GPU Stack Interactively

Before submitting a batch job, verify that PyTorch, CUDA detection, and vLLM import cleanly inside the `plotgraph` environment:
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

## 3. Submitting the SLURM Pilot Job

Once the `plotgraph` environment is configured and verified interactively, submit the pilot batch script:

```bash
sbatch scripts/sol/kg_l_qwen_pilot.sbatch
```

The scheduler will print the job ID:
```
Submitted batch job <JOB_ID>
```

---

## 4. Monitoring the Job

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

## 5. Reviewing Pilot Artifacts

When the job completes, verify the artifacts generated in `data/processed/ml-1m/kg_l/pilot/`:

| Artifact | Format | Description |
|---|---|---|
| `raw_outputs.jsonl` | JSON Lines | Complete log of every raw response from Qwen for every attempt. |
| `validated_extractions.jsonl` | JSON Lines | Strictly validated `MoviePlotExtraction` records complying with the Phase 3C contract. |
| `failures.jsonl` | JSON Lines | Records of any extractions that exhausted retries, categorized by error type. |
| `pilot_summary.json` | JSON | Summary metrics: total sample count (30), success count, retries, runtime, and success rate. |
| `pilot_review.csv` | CSV | Human-readable inspection sheet with extracted themes, settings, devices, protagonists, source materials, continuous Plutchik emotions, and prominent emotions (`score > 0.5`). |

### Quick Verification Commands
```bash
# Check summary metrics
cat data/processed/ml-1m/kg_l/pilot/pilot_summary.json

# Check first few rows of the human review table
head -n 5 data/processed/ml-1m/kg_l/pilot/pilot_review.csv

# Verify 30 successfully validated records
wc -l data/processed/ml-1m/kg_l/pilot/validated_extractions.jsonl

# Confirm zero unhandled failures
wc -l data/processed/ml-1m/kg_l/pilot/failures.jsonl
```

---

## 6. Local Development & Mock Testing (macOS / No GPU)

For local testing without GPU or model downloads:
```bash
python scripts/run_phase3d_qwen_pilot.py --backend mock
```
This runs the complete safety guard, checkpointing, retry logic, summary generation, and review CSV export locally using mock responses.
