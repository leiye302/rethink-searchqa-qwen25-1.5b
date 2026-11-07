# Search-QA evaluation: balanced monitoring, final full benchmark

Source pin: SDAR `29dcb12a55d8fc17ba6ae03fd0a69048f43051f3`.
The native launcher, Trainer, SearchEnv scorer, environment manager, collector,
retriever and preprocessor are SHA-locked in configs/upstream.json.
The evaluation adaptation below is explicit; it is not described as the
original initial/final-full-only schedule.

## Fixed task sets and cadence

The pinned PeterJinGo/nq_hotpotqa_train test file contains51,713 questions
across NQ, TriviaQA, PopQA, HotpotQA,2WikiMultiHopQA,MuSiQue and Bamboogle.
Both model sizes use identical public data revisions, source hashes and budgets.

| Update | Scope | Questions |
| --- | --- | ---: |
| 0,20,40,60,80,100,120,140 | Same fixed source-balanced monitoring subset | 448 = 7x64 |
| 150 | Complete original test, once in the normal run | 51,713 |

configs/validation_subset.json commits every selected original test-row index,
question ID and their ordered SHA256. Selection is made once using seed2026
and question-text SHA256 ranking, with64 unique questions per source and
round-robin source order. It excludes known exact normalized training-question
matches and duplicate subset question text. It does not use model outputs,
rewards, success rates or estimated difficulty.

The native StatefulDataLoader still uses shuffle=false/drop_last=false.
Its batch size512 is capacity/concurrency, not task count. The subset fits one
batch; full evaluation traverses101 complete batches and a final singleton.
No600-second cutoff, skipped final batch, automatic resampling or score-driven
subset resizing is allowed.

The completed update150 checkpoint is atomically published and SHA-verified
before the full test begins. If that test is interrupted, resume restores the
same final update and runs native validation-only, never another optimizer
update. The normal native post-validation save re-verifies rather than
overwriting the already-complete final checkpoint.

The ordinary training run already makes the final full pass. Export/report
commands do not run another full test. Independent eval defaults to full and
requires a separate explicitly requested run; --eval-scope subset is available
for a deliberately requested extra lightweight check.

## Preparation and safe loader switching

src/data.py --prepare --verify preserves the original train/test Parquets and
derives a separate validation_subset.parquet. Only extra_info.index is rebased
to subset-local0..447; prompts, question text, env_kwargs, verifier targets and
other native fields are copied unchanged. Original test indices remain in the
shared manifest and generated evaluation records. Hashes and row-by-row content
are checked. Existing complete data can be reused without downloading it again.

SearchTrainer initializes the monitor loader for training and lazily constructs
the full validation loader at update150. It swaps only val_dataset and
val_dataloader while calling the original super()._validate, then restores them
even after an error. It never recreates the training dataset/sampler or changes
GRPO, optimizer, gradients, rollout batch or checkpoint retention.

The new evaluation protocol, manifest and prepared subset receipt are part of
run/resume identity. Old full-test-only runs are not silently migrated.
New comparisons should use a new run directory for this named recipe.

## Unchanged generation and scoring

Each question gets one greedy trajectory: do_sample=false, temperature0,n1,
max4 actions, history4,4096 prompt tokens,512 response tokens/action and top3
retrieved Wiki18 passages. No Teacher, skill, external model API or verifier
labels are supplied to Actor prompts.

Native SearchEnv extracts the last complete answer block from executed history,
normalizes case, punctuation, articles and whitespace, and accepts any alias
in ground_truth.target. Missing answer is0; an explanation merely containing
the short answer is not normalized exact match. The original action projection
and validity flags remain unchanged.

## Reporting and weighting

The native reward manager repeats episode reward over active decision rows.
Original native test_score/success metrics may therefore reflect decision-row
or batch weighting. Their existing keys and values are retained, not relabeled
as deduplicated question EM.

The read-only adapter additionally reports each selected question once:

```math
\mathrm{EM}_d=\frac{1}{N_d}\sum_{i\in d}r_i,\qquad
\mathrm{EM}_{\mathrm{macro}}=\frac{1}{7}\sum_d\mathrm{EM}_d,\qquad
\mathrm{EM}_{\mathrm{micro}}=\frac{1}{N}\sum_i r_i.
```

N is448 for subset monitoring and51,713 for full evaluation. The balanced subset
has equal source weights, so its micro-EM equals its seven-source macro-EM.
Full-test micro-EM weights larger sources more. Do not join differently weighted
subset/full micro-EM points into an unlabeled curve.

Added metrics use eval/subset/... or eval/full/... namespaces. Summary JSON/CSV
records scope, count, ordered-ID SHA, per-source EM and wall_seconds. Native
collector local indices are mapped back to validation-input indices, including
the final partial batch; subset records also retain original_test_index.
Native validation does not pass the dataset's index field into gen_batch.
The adapter first verifies the actual question/target/source IDs and their
order against the selected receipt, then derives scope-local positions from
the number already seen. It never requires an absent input index or inserts
one into the native collector's inputs.
Missing, replaced or duplicated input indices fail complete-scope reporting.

Each attempt has a separate directory. An interrupted attempt without a complete
summary is not a full result. No evaluation result influences rewards, loss,
learning rate, early stopping or which questions are selected.

## Runtime target, not a fabricated timing result

448 questions is the starting fixed lightweight specification aimed at roughly
10 minutes on four A10080GB cards. This is NOT measured GPU timing. Model size,
answer length, search frequency, real E5/FAISS throughput and warm-up affect it.
A complete pass records actual wall_seconds and questions/second. If calibration
shows a different size is needed, version/refreeze one shared manifest for both
backbones before the comparison; never truncate a running evaluation at600s
or change questions independently by model/update.

## Original split overlap and verification boundary

The original downloaded train/full-test files share10 question strings under
exactly str(question).casefold().strip(). The original169,615/51,713 splits are
preserved and that limited exact overlap is reported. The monitoring subset
excludes such training-text matches. No full semantic-disjointness claim is made.

CPU/source/data/HTTP-fixture tests do not measure Actor benchmark performance,
GPU throughput, real E5/FAISS retrieval quality, A100 memory fit or150-update
results. They validate protocol/identity/aggregation/storage, not model quality.
No score table is prefilled.
