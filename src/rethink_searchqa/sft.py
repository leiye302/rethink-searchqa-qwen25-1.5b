"""Exact frozen wrapped Stage-1 data and objective on the pinned 1.5B base."""
import argparse
from collections import Counter
import gzip
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

from common import ROOT, atomic_json, load, sha
from .retriever_guard import RetrievalFailure
from .shell import INTRO_DIGEST, MARKER_ID, STYLE, assert_token
from .stage1_assets import OBJECTIVE, fetch_asset

SIZE='1.5b'
STEPS=800


def read(path):
    return json.loads(Path(path).read_bytes())


def compressed_rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def check_data(path):
    lock=load('sft_data.json')
    if (lock['schema']!='searchqa-sft-v4-minimal-v1' or lock['steps']!=STEPS
            or lock['intro_digest']!=INTRO_DIGEST or lock['token_id']!=MARKER_ID):
        raise ValueError('Different Stage-1 contract')
    for name,expected in lock['files'].items():
        f=path/name
        if f.is_symlink() or not f.is_file() or f.stat().st_size!=expected['bytes'] or sha(f)!=expected['sha256']:
            raise ValueError('Missing or changed frozen SFT asset: '+name)
    train=compressed_rows(path/'train.jsonl.gz')
    probes=compressed_rows(path/'probes200.jsonl.gz')
    order=read(path/'batch_order.json')
    counts=Counter()
    for x in train:
        if set(x)!={'input_ids','route_label','history_tokens','suffix_start'}:
            raise ValueError('Unexpected fields in minimal SFT row')
        ids,y,h,s=(x[k] for k in ('input_ids','route_label','history_tokens','suffix_start'))
        if (not 0<h<=4096 or not h<s<len(ids)<=8192 or len(ids)-h>512 or y not in (0,1)
                or ids[-1]!=lock['eos_id'] or ids[:h].count(MARKER_ID)!=2
                or ids[:s].count(MARKER_ID)!=2 or ids[s:].count(MARKER_ID)!=y
                or (ids[s]==MARKER_ID)!=bool(y)):
            raise ValueError('SFT mask or marker/EOS boundary changed')
        counts[y]+=1
    if (counts!={0:4000,1:2400} or len(order)!=STEPS
            or sorted(i for b in order for i in b)!=list(range(len(train)))
            or any(len(b)!=8 or sum(train[i]['route_label'] for i in b)!=3 for b in order)):
        raise ValueError('SFT must consume 4000 Good + 2400 Bad exactly once, every batch 5:3')
    if len(probes)!=200 or any(set(x)!={'prompt_ids'} or not 0<len(x['prompt_ids'])<=4096
                              or x['prompt_ids'].count(MARKER_ID)!=2 for x in probes):
        raise ValueError('Wrong frozen empty-prefill probe set')
    return train,order,[x['prompt_ids'] for x in probes]


def prepare(path):
    lock=load('sft_data.json')
    path.mkdir(parents=True,exist_ok=True)
    if path.is_symlink():
        raise ValueError('Use a real data directory')
    base=f"https://github.com/{lock['repository']}/releases/download/{lock['tag']}"
    for name,expected in lock['files'].items():
        fetch_asset(f'{base}/{name}',path/name,expected)
    return check_data(path)


def verify_base(path):
    spec=load('models.json')[SIZE]
    receipt=read(path/'download_receipt.json')
    if any(receipt[k]!=spec[k] for k in ('model_id','revision')):
        raise ValueError('SFT must start from the pinned original 1.5B model')
    for name,entry in receipt['files'].items():
        if (Path(name).name!=name or (path/name).stat().st_size!=entry['bytes']
                or sha(path/name)!=entry['sha256']):
            raise ValueError('Original model file differs')
    if sha(path/'tokenizer.json')!=load('sft_data.json')['source_tokenizer_sha256']:
        raise ValueError('The 1.5B tokenizer must byte-match the frozen 3B data tokenizer')
    return spec


def update(model,opt,rows,indices,core):
    import torch
    torch.cuda.synchronize()
    t=time.monotonic()
    opt.begin()
    accum={'loss':0.,'suffix_ce':0.,'route_bce':0.}
    for i in indices:
        row=rows[i]
        ids=torch.tensor(row['input_ids'],dtype=torch.long,device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(input_ids=ids[None],attention_mask=torch.ones_like(ids)[None],
                         use_cache=False,return_dict=True).logits[0]
            parts=core.losses(logits,ids,row['suffix_start'],row['route_label'],MARKER_ID,route_lambda=.1)
            loss=parts['loss']/8
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite SFT loss')
        loss.backward()
        for k in accum:
            accum[k]+=float(parts[k].detach())/8
        del logits,parts,loss,ids
    norm=opt.finish(clip=1.)
    torch.cuda.synchronize()
    return dict(accum,gradient_norm_before_clip=norm,seconds=time.monotonic()-t)


def save(model,tok,opt,scheduler,order,out,parent):
    import torch
    if opt.updates!=STEPS:
        raise ValueError('Only the finished 800-step Actor can initialize Stage 2')
    pending=out/'checkpoint_000800.pending'
    final=out/'checkpoint_000800'
    if final.exists():
        raise FileExistsError('Never overwrite the final Stage-1 checkpoint')
    pending.mkdir()
    model.save_pretrained(pending/'model',safe_serialization=True,max_shard_size='2GB')
    tok.save_pretrained(pending/'model')
    torch.save({'optimizer':opt.optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'python_rng':random.getstate(),'cpu_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all(),'next_update':STEPS,'batch_order':order,
                'data_lock_sha256':sha(ROOT/'configs/sft_data.json')},pending/'training_state.pt')
    assets={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in (pending/'model').iterdir() if p.is_file()}
    atomic_json(pending/'stage1_manifest.json',{
        'schema':'searchqa-wrapped-sft800-local-v1','size':SIZE,
        'model_id':parent['model_id'],'revision':parent['revision'],'completed_updates':STEPS,
        'protocol':STYLE,'intro_digest':INTRO_DIGEST,'marker_id':MARKER_ID,'loss':OBJECTIVE,
        'data_mix':{'good':4000,'bad':2400},'data_lock_sha256':sha(ROOT/'configs/sft_data.json'),
        'training_state_sha256':sha(pending/'training_state.pt'),'files':assets})
    pending.rename(final)
    from .stage1_assets import verify
    verify(final/'model',SIZE)
    return final


def run(args,train,order,probes):
    if not args.authorize_training or sys.platform!='linux':
        raise ValueError('Stage 1 requires Linux and explicit --authorize-training')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from setup import verify as verify_framework
    from data import verify as verify_data
    from run import base_model, check_retriever
    from retrieval import verify as verify_retrieval
    from .gpu_resident import GpuResidentAdamW
    from .sft_eval import diagnostic, monitor
    from pyarrow import parquet

    verify_framework()
    verify_data(args.data_dir)
    verify_retrieval(args.retrieval_dir)
    check_retriever(args.search_url)
    expected=read(args.data_dir/'validation_subset_receipt.json')['rows']
    questions=parquet.read_table(args.data_dir/'validation_subset.parquet',columns=['env_kwargs']).column(0).to_pylist()
    base=base_model(SIZE)
    parent=verify_base(base)
    if (torch.__version__.split('+')[0]!='2.8.0' or torch.version.cuda!='12.8'
            or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported()
            or torch.cuda.mem_get_info()[0]<40*2**30):
        raise ValueError('Select one allocated BF16 CUDA GPU with 40 GiB free and the pinned torch runtime')
    sys.path.insert(0,str(ROOT/'src/sft_reference'))
    from route_bce import core
    if args.run_dir.exists():
        raise FileExistsError('Preserve the existing run; use a new run directory')
    args.run_dir.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(args.run_dir.parent).free<24*2**30:
        raise ValueError('Need 24 GiB free for model, training state, and outputs')
    args.run_dir.mkdir()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    tok=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False)
    def fresh():
        random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,trust_remote_code=False,
                          dtype=torch.float32,attn_implementation='sdpa',low_cpu_mem_usage=True)
        core.register_token(tok,model=model); assert_token(tok)
        if tok.eos_token_id!=151645:
            raise ValueError('Unexpected EOS')
        model.config.use_cache=False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        return model.train().to('cuda')

    # Disposable full-model updates, then reload the exact starting weights.
    model=fresh()
    opt=GpuResidentAdamW(model)
    longest=[i for label,n in ((0,5),(1,3)) for i in sorted(
        (j for j,r in enumerate(train) if r['route_label']==label),
        key=lambda j:len(train[j]['input_ids']),reverse=True)[:n]]
    for batch in (order[0],longest):
        print(json.dumps({'disposable_preflight':True,**update(model,opt,train,batch,core)}),flush=True)
    del model,opt
    torch.cuda.empty_cache()
    model=fresh()
    opt=GpuResidentAdamW(model)
    # Keep the original 900-step cosine schedule, ending the data pass at 800.
    factor=lambda s: s/150 if s<150 else .5*(1.+math.cos(math.pi*(s-150)/750))
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt.optimizer,factor)

    def monitor_at(step):
        try:
            metrics=monitor(model,tok,questions,expected,args.search_url,step)
        except RetrievalFailure as error:
            atomic_json(args.run_dir/f'eval{step}_invalid.json',
                        {**error.receipt(),'scope':'monitor448','completed_updates':step})
            raise
        atomic_json(args.run_dir/f'eval{step}.json',metrics)

    with (args.run_dir/'training.jsonl').open('x',encoding='utf-8') as f:
        for step,batch in enumerate(order,1):
            lr=opt.optimizer.param_groups[0]['lr']
            result=update(model,opt,train,batch,core)
            scheduler.step()
            row=json.dumps(dict(result,step=step,lr_used=lr),allow_nan=False)
            f.write(row+'\n'); f.flush()
            if step%20==0:
                print(row,flush=True)
            if step==400:
                opt.release_gradients()
                monitor_at(step)
    opt.release_gradients()
    save(model,tok,opt,scheduler,order,args.run_dir,parent)
    del opt,scheduler
    torch.cuda.empty_cache()
    monitor_at(800)
    atomic_json(args.run_dir/'final200.json',diagnostic(model,tok,probes))
    print('Complete: fixed 800 updates, checkpoint, 400/800 monitor and final200.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','check','train'))
    p.add_argument('--sft-data',type=Path,default=ROOT/'data/stage1_v4')
    p.add_argument('--run-dir',type=Path,default=ROOT/'runs/sft')
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/searchqa')
    p.add_argument('--retrieval-dir',type=Path,default=ROOT/'data/retrieval')
    p.add_argument('--search-url',default=os.environ.get('SEARCH_URL','http://127.0.0.1:8000/retrieve'))
    p.add_argument('--authorize-training',action='store_true')
    a=p.parse_args()
    train,order,probes=prepare(a.sft_data) if a.command=='prepare' else check_data(a.sft_data)
    if a.command=='train':
        run(a,train,order,probes)
    else:
        print('Verified 6400 reviewed rows, exact 5:3 order at 800 steps, 200 unlabeled probes.')


if __name__=='__main__':
    main()
