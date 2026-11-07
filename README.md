# RETHINK Search-QA — Qwen2.5-1.5B-Instruct

Built with Qwen. This is the **1.5B method** arm: original 1.5B base →
wrapped Route-SFT800 → native GRPO+OPSD with local Teacher hints. Stage 2
requires the **local same-size SFT800 Actor**; the companion 3B Actor, the
original 1.5B base and intermediate SFT checkpoints are not substitutes.
No new 1.5B training results or trained weights are claimed here.

Stage 1 starts from the pinned `Qwen/Qwen2.5-1.5B-Instruct`. The source
tokenizer is byte-identical to the 3B source tokenizer. Reuse the same
reviewed H/R and action targets, order and masks (4,000 natural Good +
2,400 reviewed Bad, 5:3, sample-mean suffix CE + 0.1 marker BCE,
no reasoning KL). See the [Stage-1 data and run contract](docs/STAGE1.md).
Both SFT and Stage 2 use native history, the same explicit optional
`<RETHINK>` instruction, and the following current-action shell:

```text
<think>reasoning</think><action>search[query]</action>
<think>reasoning</think><RETHINK><action>answer[answer]</action>
```

The tool adapter unwraps the action into the native search/answer tags before
the original projector. It does not require or force a reasoning block.
During training, only an actual Actor-generated marker ID requests a local hint.
All trajectories retain native GRPO and beta5 OPSD. Static public search skills
and the extra local hint are Teacher-only; the same current Actor scores the
same sampled Student tokens without gradients. No gold answer is sent to the
hint provider, Student or Teacher. Evaluation needs no hint API.

## Stage 2 on a clean Linux host

Allocate one node with four A100 80GB GPUs, CUDA 12.8-compatible driver,
Python 3.12 and Git. Plan at least 32 logical CPUs, 256 GiB host RAM and 350 GiB
free disk (400–500 GiB is safer for downloads, index and full saves).
The retriever separately uses Python 3.10 and CUDA; use only allocated devices.

```bash
git clone https://github.com/leiye302/rethink-searchqa-qwen25-1.5b.git
cd rethink-searchqa-qwen25-1.5b
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python src/setup.py --framework
python -m pip install -r configs/requirements.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install -e .deps/sdar -c configs/requirements.txt
python -m pip install -e .deps/sdar/agent_system/environments/env_package/search/third_party -c configs/requirements.txt
python -m pip check
python src/data.py --prepare --verify
python scripts/check_release.py
python -m pytest -q
bash scripts/train.sh --dry-run
```

Prepare the original retriever in a separate environment. The full index is a
large download; configure your own disks and GPU IDs. Keep the HTTP endpoint
on a private network.

```bash
conda create -n searchqa-retriever python=3.10 -y
conda activate searchqa-retriever
conda install numpy=1.26.4 faiss-gpu=1.8.0 -c pytorch -c nvidia -y
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r configs/requirements-retriever.txt
python src/retrieval.py --prepare
bash scripts/start_retriever.sh --gpus 0,1,2,3
```

Before Stage 2, prepare the model and data and run Stage 1 with the separately
pinned SFT environment. See [Stage 1](docs/STAGE1.md) for the full commands.
The default verified handoff is `runs/sft/checkpoint_000800/model`.

In the training environment, configure your own OpenAI-compatible endpoint
serving the pinned `glm-5.3-flash`. These are placeholders, not included credentials.
Zero-trigger batches use the unchanged static Teacher and make no requests.

```bash
source .venv/bin/activate
export HINT_API_BASE="https://your-provider.example/v1"
export HINT_MODEL="glm-5.3-flash"
read -rsp 'Hint API key: ' HINT_API_KEY; echo
export HINT_API_KEY
PYTHONPATH=src python -m rethink_searchqa.hint_api --check
bash scripts/train.sh --allow-api-cost
# Resume only the same run after the old process has exited:
# bash scripts/train.sh --resume --allow-api-cost
python src/run.py export --size 1.5b
```

## Comparison contract

The [previous Search-QA GRPO+OPSD release](https://github.com/leiye302/grpo-opsd-searchqa-qwen25-1.5b)
is the control. Both use the same pinned SDAR source, 150 updates, 128 questions
× 8 rollouts/update, 4 actions, history 4, top-3 Wiki18, 4096/512 tokens, and
native exact match. The common fixed 448 questions (64/source) are evaluated
at 0/20/40/60/80/100/120/140; the complete 51,713 test is run only at 150,
after the final checkpoint has been verified. Step 0 here is **Stage 2 step 0**
(the SFT800 Actor), not an untrained baseline. The checkpoint schedule and
resume behavior match the existing Search-QA repositories.

See [exact method deltas](docs/UPSTREAM_DIFF.md), [evaluation](docs/EVALUATION.md)
and [source licenses](THIRD_PARTY.md). CPU checks and the fixture retriever
do not establish real E5/FAISS, four-GPU vLLM/FSDP, or full 150-update results.
