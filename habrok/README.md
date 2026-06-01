# Running the UTKFace evaluation on Habrok

Cheat-sheet for taking the code in `code/` and running it on the University of
Groningen's Habrok HPC cluster. Tested on Habrok's Python module + venv
workflow (https://wiki.hpc.rug.nl/habrok/start).

## 1. Get the code and data onto Habrok

From your local machine, copy the project to `/scratch/$USER/BSc` on Habrok.
Replace `<user>` with your Habrok username and `<login>` with whichever login
node you use (e.g. `login1.hb.hpc.rug.nl`):

```bash
# from a local shell that can ssh to Habrok
rsync -avh --exclude tf-env --exclude __pycache__ \
    ~/Documents/GitHub/BSc/ <user>@<login>:/scratch/<user>/BSc/
```

Notes:
- We exclude `tf-env/` (the Windows venv) — Habrok will get its own.
- The `utkface/` folder (~1.3 GB) and `appa-real-release/` come with the copy.
- The trained `.keras` files in `code/trained_models_by_fraction/` are
  platform-independent, so they go as-is.

## 2. Build the venv (one-time)

SSH into Habrok and run:

```bash
cd /scratch/$USER/BSc/habrok
bash setup_env.sh
```

This loads a Python module, creates `/scratch/$USER/BSc/venv/`, and pip-installs
the deps in `requirements.txt`. Takes ~5 min.

## 3. Submit the prediction job

```bash
cd /scratch/$USER/BSc/habrok
sbatch run_utkface.slurm
```

This launches a SLURM array job: 9 array tasks, one per (method, fraction)
configuration. Each task asks for one GPU, writes its CSV to
`code/utkface_results/`, and logs to `habrok/logs/`.

Monitor with:
```bash
squeue --me
tail -f logs/utkface-*.out
```

Expected wall time per task on an A100: ~30-60 min for 23705 images x 20 MC
passes. Total clock time for the array (running in parallel): ~1 hour.

## 4. Run the analysis after predictions land

Once all 9 array tasks have finished, run the analysis script (no GPU needed —
it's just plotting and metric aggregation):

```bash
sbatch run_analyze.slurm
```

Outputs land in `code/utkface_results/figures/`.

## 5. Pull the results back

From your local machine:

```bash
rsync -avh <user>@<login>:/scratch/<user>/BSc/code/utkface_results/ \
    ~/Documents/GitHub/BSc/code/utkface_results/
```

## Troubleshooting

- **Module not found** — `module avail Python` on Habrok and pick a 3.11.x
  version matching the one in `setup_env.sh`. Update both if needed.
- **CUDA / GPU not detected** — confirm `nvidia-smi` works on the node, and
  that the venv's TF is the GPU build (`pip install tensorflow[and-cuda]==2.19`).
- **OOM on a node** — fall back to a node with bigger GPU memory:
  `#SBATCH --gpus-per-node=a100:1` already grabs A100 (40 GB), which is plenty
  for DenseNet-121 inference at batch 32. If you still OOM, drop `--batch_size`
  to 16 in `run_utkface.slurm`.
