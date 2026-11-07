# 1.5B wrapped Route-SFT: frozen data and exact handoff

This companion uses the same frozen wrapped dataset as the 3B Stage-1 run,
not 3B weights and not new 1.5B rollouts. The tokenizer file was byte-compared
across pinned public Qwen revisions. The only new trainable starting weights
are the pinned original Qwen2.5-1.5B-Instruct. Use the same 800-update budget;
do not turn the second stage's 0-step evaluation into a new Stage-1 baseline.

## Data contract

The [minimal frozen v4 bundle](https://github.com/leiye302/rethink-searchqa-qwen25-1.5b/releases/tag/sft-data-v4)
contains only token IDs visible to the model, route/suffix masks, the exact
one-pass batch order and 200 unlabeled H probes. All file hashes are in
`configs/sft_data.json`. The assets are not stored in Git history.
Source IDs, private annotation artifacts, unfiltered trajectories, gold verifier
records and credentials are not exported.

Construction used model-native rollouts from the Search-QA training pool and
native visible history. Good reasoning has no clearly visible unresolved error;
the Good target stays the original natural action. Bad requires a material,
visible, unresolved reasoning error that would mislead the decision.
Ambiguous states are not training examples. Bad gets a marker and a reviewed
correction: a search query is grounded in the question and visible clues;
an answer is used only when there is sufficient visible evidence.
After curation, use the identical wrapped tool instruction and tokenize
the full H/R/target on canonical boundaries. The current release freezes
the vetted rows and does not issue new annotation API calls.

H and the entire existing reasoning R (including closing format) are masked
for CE. Only the route/action/EOS suffix is supervised. A Good target begins
with the action shell; Bad begins with one registered token `<RETHINK>`.
BCE uses the full-vocabulary probability of that token at the position just
before the suffix; it does not supervise the first think token.

```math
L=\frac1{8}\sum_i\left[
 \frac1{|u_i|}\sum_{t\in u_i}-\log p_\theta(u_{i,t}\mid H_i,R_i,u_{i,<t})
 +0.1\,\mathrm{BCE}(p_\theta(\mathrm{RETHINK}\mid H_i,R_i),b_i)
\right].
```

## Run on a single allocated GPU

Prepare the pinned SDAR source, shared Search-QA QA files and retriever as in
the README. Use a separate Python 3.12 SFT environment on Linux (CUDA 12.8).
Use one BF16 GPU with at least 40 GiB free and at least 24 GiB free disk for
this admission profile. No claim of memory fit on smaller cards is made.
The retriever must already serve the complete pinned E5/Wiki18 top-3 corpus;
reserve resources for it separately or account for them before launching.

```bash
python3.12 -m venv .venv-sft
.venv-sft/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.venv-sft/bin/python -m pip install -r configs/requirements-sft.txt
# Downloads the pinned original public 1.5B model, never a 3B model.
python src/setup.py --download-model 1.5b
PYTHONPATH=src .venv-sft/bin/python -m rethink_searchqa.sft prepare
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv-sft/bin/python -m rethink_searchqa.sft train --authorize-training
```

One epoch: 800 updates, every global batch 5 Good + 3 Bad and microbatch 1.
All FP32 model parameters train with BF16 autocast, non-reentrant gradient
checkpointing, and GPU-resident FP32 AdamW (LR 2e-6, betas .9/.95, decay .01,
gradient clip 1). Seed 0; keep the original 150-step warmup and cosine horizon
900, stopping the one-pass data order at update 800. Two disposable batches
check ordinary/worst-length inputs before loading a fresh exact parent.

Use no step-0 SFT test. At steps 400 and 800, perform a real greedy benchmark
on the common ordered 448 questions, four actions, history four, 4096/512
tokens and native exact match. The retriever service is the same pinned native
E5/Wiki18 interface used by the Stage-2 releases. No scores from another
retrieval runtime are represented as new results. At 800 also run 200 greedy
generations with an empty assistant prefix and count marker IDs/valid boundary.
Publish a model+optimizer/scheduler/RNG checkpoint before the final evaluation.
Stage 2 verifies the Stage-1 manifest, dataset hash, backbone size, tokenizer,
recipe and model file hashes, then initializes only the Actor weights.

The CPU checks validate packaging, actual native search/reward and loss masks;
they do not constitute a completed 1.5B GPU training run.
