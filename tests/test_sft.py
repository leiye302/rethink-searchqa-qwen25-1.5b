import json
import os
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import UPSTREAM, load
from rethink_searchqa import shell, sft


def test_1p5b_has_distinct_verified_stage1_handoff():
    m=load('sft_data.json')
    assert load('release.json')['size']=='1.5b'
    assert load('release.json')['default_sft_actor']=='runs/sft/checkpoint_000800/model'
    assert m['steps']==800 and m['token_id']==shell.MARKER_ID and m['intro_digest']==shell.INTRO_DIGEST
    assert set(m['files'])=={'batch_order.json','train.jsonl.gz','probes200.jsonl.gz'}
    assert load('models.json')['1.5b']['revision']=='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'


def test_stage1_masked_reasoning_and_post_reasoning_bce():
    torch=pytest.importorskip('torch')
    sys.path.insert(0,str(ROOT/'src/sft_reference'))
    from route_bce import core
    marker=7
    ids=torch.tensor([1,2,3,4,marker,5,6])
    logits=torch.zeros(7,8,requires_grad=True)
    loss=core.losses(logits,ids,4,1,marker,route_lambda=.1)
    torch.testing.assert_close(loss['loss'],loss['suffix_ce']+.1*loss['route_bce'])
    loss['loss'].backward()
    assert not logits.grad[:3].any() and logits.grad[3:6].abs().sum()>0 and not logits.grad[6].any()


def test_wrapped_real_native_search_feedback_and_exact_match():
    from omegaconf import OmegaConf
    from env_smoke import fixture_retriever
    sys.path.insert(0,str(UPSTREAM))
    from agent_system.environments.env_manager import SearchEnvironmentManager
    from agent_system.environments.env_package.search import build_search_envs, search_projection
    with fixture_retriever() as (url,calls):
        cfg=OmegaConf.create({'env':{'history_length':4,'max_steps':4,
            'search':{'search_url':url,'topk':3,'timeout':5,'log_requests':False}}})
        e=build_search_envs(seed=2026,env_num=1,group_n=1,is_train=True,env_config=cfg.env)
        manager=shell.wire_environment(SearchEnvironmentManager(e,search_projection,cfg),capture_hints=False)
        try:
            obs,_=manager.reset([{'question':'What is the capital of France?',
                                  'ground_truth':{'target':['Paris']},'data_source':'nq'}])
            assert obs['text'][0].count(shell.RETHINK_INTRO)==1
            obs,r,done,infos=manager.step(['<think>Check the source.</think><RETHINK><action>search[capital of France]</action>'])
            assert len(calls)==1 and calls[0]['topk']==3 and not done[0] and float(r[0])==0
            assert '<search>capital of France</search>' in obs['text'][0] and 'Paris' in obs['text'][0]
            assert infos[0]['postprocessed_action']=='<search>capital of France</search>'
            _,r,done,infos=manager.step(['<action>answer[PARIS!]</action>'])
            assert done[0] and float(r[0])==1 and infos[0]['is_action_valid']
        finally:
            e.close()


def test_exact_public_data_when_provided():
    path=os.environ.get('RETHINK_SFT_TEST_DATA')
    if not path:
        pytest.skip('Pinned public SFT assets not supplied to this CPU test')
    train,order,probes=sft.check_data(Path(path))
    assert len(train)==6400 and len(order)==800 and len(probes)==200
