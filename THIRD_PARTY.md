# Sources and licensing

- Native framework: [ZJU-REAL/SDAR](https://github.com/ZJU-REAL/SDAR), pinned
  commit29dcb12a55d8fc17ba6ae03fd0a69048f43051f3, Apache-2.0. It builds on
  veRL and verl-agent and vendors Search-R1/SkyRL components with their own
  notices. Upstream source is fetched intact; preserve its copyright/license
  files and per-file notices when redistributing it.
- Search-QA records: [PeterJinGo/nq_hotpotqa_train](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train),
  fixed revision in configs/data.json. Respect the original QA datasets'
  licenses and restrictions. Dataset weights/records are not bundled here.
- Retrieval: [wiki-18-e5-index](https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index),
  [wiki-18-corpus](https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus),
  [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2).
  Exact revisions and artifact checksums are in configs/retrieval.json.
  Respect the corpus/model licenses; this code does not relicense them.
- Backbones: [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
  (Apache-2.0) and [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
  (Qwen Research License; consult the pinned model LICENSE before use).
- The wrapper and storage utilities in this release are Apache-2.0. They do
  not grant a different license to model weights, upstream third-party code,
  corpora or QA datasets. Public model/assets are downloaded, not republished
  in the Git repository. No credentials or private experiment data are included.
