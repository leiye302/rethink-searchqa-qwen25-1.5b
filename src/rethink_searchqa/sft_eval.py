"""Single-GPU greedy Stage-1 diagnostics and the real fixed Search-QA monitor."""
from contextlib import contextmanager
from collections import Counter
import re
import sys
from common import UPSTREAM, question_id
from evaluation_contract import summarize_questions
from . import action_wrapper, shell


@contextmanager
def evaluating(model):
    import torch
    training=model.training
    checkpointing=model.is_gradient_checkpointing
    model.eval()
    if checkpointing:
        model.gradient_checkpointing_disable()
    try:
        with torch.random.fork_rng(devices=[0]),torch.inference_mode():
            yield
    finally:
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.train(training)


def greedy(model,tok,inputs):
    import torch
    result=[]
    eos=model.generation_config.eos_token_id or tok.eos_token_id
    eos=[eos] if isinstance(eos,int) else list(eos)
    for start in range(0,len(inputs),2):
        group=inputs[start:start+2]
        width=max(map(len,group))
        ids=torch.tensor([[tok.pad_token_id]*(width-len(p))+p for p in group],device='cuda')
        masks=torch.tensor([[0]*(width-len(p))+[1]*len(p) for p in group],device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=model.generate(input_ids=ids,attention_mask=masks,do_sample=False,max_new_tokens=512,
                               eos_token_id=eos,pad_token_id=tok.pad_token_id,use_cache=True)
        for seq in out[:,width:].tolist():
            cut=next((i for i,x in enumerate(seq) if x in eos),None)
            result.append(seq if cut is None else seq[:cut+1])
    return result


def diagnostic(model,tok,probes):
    result=[]
    counts=Counter({'reasoning':0,'marker':0,'valid_marker':0,'strict':0})
    with evaluating(model):
        for i,ids in enumerate(greedy(model,tok,probes)):
            text=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            m=action_wrapper.generation_metrics(text,ids,shell.MARKER_ID,[tok.eos_token_id])
            r=re.search(r'<think>([\s\S]*?)</think>',text)
            counts.update({'reasoning':bool(r and r[1].strip()),'marker':m['marker_present'],
                           'valid_marker':m['marker_at_valid_boundary'],'strict':m['strict_protocol']})
            result.append({'probe':i,'generated_ids':ids,'text':text,**m})
    return {'n':len(result),'assistant_prefill':'','summary_counts':dict(counts),'generations':result}


def monitor(model,tok,questions,expected,url,step):
    """No model API or hint capture; original projector, retrieval and reward."""
    if step not in (400,800) or len(questions)!=448:
        raise ValueError('Stage 1 only monitors the locked 448 at 400 and 800')
    sys.path.insert(0,str(UPSTREAM))
    from omegaconf import OmegaConf
    from agent_system.environments.env_manager import SearchEnvironmentManager
    from agent_system.environments.env_package.search import build_search_envs, search_projection
    cfg=OmegaConf.create({'env':{'history_length':4,'max_steps':4,
        'search':{'search_url':url,'topk':3,'timeout':60,'log_requests':False}}})
    ids=[question_id(q) for q in questions]
    if ids!=expected['question_ids']:
        raise ValueError('Different monitor questions/order')
    records=[]
    with evaluating(model):
        for start in range(0,448,24):
            group=questions[start:start+24]
            envs=build_search_envs(seed=2026,env_num=len(group),group_n=1,
                                   is_train=True,env_config=cfg.env)
            manager=shell.wire_environment(SearchEnvironmentManager(envs,search_projection,cfg),capture_hints=False)
            try:
                obs,_=manager.reset(group)
                active=list(range(len(group)))
                for _ in range(4):
                    if not active:
                        break
                    ps=[tok.encode(tok.apply_chat_template([{'role':'user','content':obs['text'][i]}],
                          add_generation_prompt=True,tokenize=False),add_special_tokens=False)[-4096:]
                        for i in active]
                    actions=['']*len(group)
                    for i,generated in zip(active,greedy(model,tok,ps),strict=True):
                        actions[i]=tok.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    obs,rewards,dones,_=manager.step(actions)
                    for i in active:
                        if bool(dones[i]):
                            index=start+i
                            records.append({'index':index,'question_id':ids[index],
                                            'data_source':group[i]['data_source'],
                                            'episode_reward':float(rewards[i])})
                    active=[i for i in active if not bool(dones[i])]
                if active:
                    raise RuntimeError('Incomplete native episodes')
            finally:
                envs.close()
    return dict(summarize_questions(records,ids,expected['source_counts']),
                completed_updates=step,scope='monitor448',hint_api_calls=0)
