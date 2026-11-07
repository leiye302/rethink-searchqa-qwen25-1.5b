import importlib.util
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import pickle
from pathlib import Path
import re
import sys
import threading
from types import ModuleType, SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from common import UPSTREAM, load
from rethink_searchqa import hints, hint_api, shell
from rethink_searchqa.retriever_guard import RetrievalFailure, failure_reason


def test_shell_uses_exact_instruction_and_retains_native_parser():
    from rethink_searchqa.action_wrapper import NATIVE_TAIL, WRAPPED_TAIL
    from rethink_searchqa.protocol import digest
    assert digest(shell.RETHINK_INTRO) == shell.INTRO_DIGEST
    prefix='Task and visible history: <search>previous query</search>\n'
    native=prefix+NATIVE_TAIL+'\n'
    projector=UPSTREAM/'agent_system/environments/env_package/search/projection.py'
    spec=importlib.util.spec_from_file_location('pinned_search_projection',projector)
    proj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proj)
    env=SimpleNamespace(build_text_obs=lambda *a,**kw:[native],projection_f=proj.search_projection,
                        step=lambda *a,**kw: None)
    shell.wire_environment(env,capture_hints=False)
    assert env.build_text_obs([])==[prefix+shell.RETHINK_INTRO+WRAPPED_TAIL+'\n']
    values=['<action>answer[Paris]</action>',
            '<think>check</think><RETHINK><action>search[author\'s birth city]</action>',
            '<action>search[q]</action><action>answer[Paris]</action>',
            '<answer>native compatible</answer>']
    acts,valids=env.projection_f(values)
    assert acts==['<answer>Paris</answer>',"<search>author's birth city</search>",
                  '<search>q</search>','<answer>native compatible</answer>']
    assert valids==[1,1,0,1]
    assert not env._rethink_capture_hints


class MarkerTokenizer:
    pad_token_id = 0

    def encode(self,text,add_special_tokens=False):
        return [shell.MARKER_ID] if text==shell.MARKER else [ord(c)+1 for c in text]
    def convert_ids_to_tokens(self,i):
        return shell.MARKER if i==shell.MARKER_ID else ''
    def decode(self,ids,skip_special_tokens=False,**kwargs):
        return ''.join(shell.MARKER if int(i)==shell.MARKER_ID else chr(int(i)-1)
                       for i in ids if int(i)>0)


def test_active_generated_marker_only_not_prompt_padding_or_annotator(monkeypatch,tmp_path):
    torch=pytest.importorskip('torch')
    m=shell.MARKER_ID
    response=torch.tensor([[72,0],[m,0],[71,m],[m,0]])
    prompt=torch.tensor([[m,65],[m,66],[m,67],[m,68]])
    batch=SimpleNamespace(batch={'responses':response,
        'input_ids':torch.cat((prompt,response),1),
        'attention_mask':torch.tensor([[1,1,1,0]]*4)},
        non_tensor_batch={'ground_truth':['VERIFIER_ONLY_SENTINEL']*4})
    episodes=[[] for _ in range(4)]
    info=[{'postprocessed_action':'<search>query</search>'} for _ in episodes]
    hints.capture(episodes,MarkerTokenizer(),batch,{'anchor':['snippet']*4},info,[1,1,1,0],0)
    rows=[[{}] for _ in episodes]
    hints.attach(episodes,rows,['Synthetic question']*4)
    assert [bool(r and r[0].get('rethink_triggered')) for r in rows]==[False,True,False,False]
    payload=json.loads(rows[1][0]['rethink_episode_json'])
    assert 'VERIFIER_ONLY_SENTINEL' not in json.dumps(payload)
    assert rows[0][0]['rethink_episode_json']==''
    # Exactly one Actor trigger, even after reordering and duplication.
    calls=[]
    def fetch(e,cache,**kw):
        calls.append(e)
        return {0:'Reasoning skill: Search the correct entity.'},{}
    monkeypatch.setattr(hints,'fetch',fetch)
    order=[rows[i][0] for i in (1,0,1,2)]
    batch.non_tensor_batch={k:[r[k] for r in order] for k in order[0]}
    values,reasons=hints.collect_hints(batch,tmp_path)
    assert len(calls)==1 and reasons==['applied','not_triggered','applied','not_triggered']
    assert values[0]==values[2] and values[1] is values[3] is None


def test_search_hint_native_schema_and_fallback():
    episode={'benchmark':'search','task':'Who wrote the work?',
        'steps':[{'step':0,'triggered':True,'observation':'Visible document','reasoning':'r',
                  'action':'<search>work</search>','feedback':'retrieved evidence'}]}
    group=hint_api.prepare_group(episode)
    assert set(group['targets'])=={'0'} and group['episode']['benchmark']=='search'
    raw={'hints':{'0':"Next command: '<search>work author</search>'. Check the attribution."}}
    v,e=hint_api.validate_hints(raw,episode)
    assert set(v)=={0} and not e
    raw['hints']['0']="Next command: 'click[buy]'. Not a native action."
    v,e=hint_api.validate_hints(raw,episode)
    assert not v and e[0]=='invalid_hint_command_or_length'


@pytest.mark.parametrize('overflow',[False,True])
def test_teacher_hint_only_one_row_same_native_responses_and_grpo(monkeypatch,tmp_path,overflow):
    torch=pytest.importorskip('torch')
    from test_sdar import provider, teacher_functions
    env=ModuleType('verl.trainer.ppo.rlsd_ray_trainer')
    env.build_teacher_batch=teacher_functions(torch)['build_teacher_batch']
    monkeypatch.setitem(sys.modules,env.__name__,env)
    tok=MarkerTokenizer()
    head='Synthetic native user prompt\nYour question: KEEP_QUESTION\n'+hints.HISTORY_INTRO
    tail=' RECENT_EVIDENCE'+hints.TURN_INTRO+' KEEP_TURN_INSTRUCTIONS'
    prompt=head+('OLDEST_HISTORY'+' visible context '*600)[:8192-len(head)-len(tail)]+tail if overflow else 'User: question'
    p=torch.tensor([tok.encode(prompt)]*2)
    responses=torch.tensor([[75,0],[76,0]])
    ids=torch.cat((p,responses),1)
    mask=(ids!=0).long()
    class Batch:
        def __init__(self,t,n):self.batch=t;self.non_tensor_batch=n
        def select_idxs(self,inds):
            return Batch({k:v[inds] for k,v in self.batch.items()},
                         {k:[v[i] for i in inds] for k,v in self.non_tensor_batch.items()})
    batch=Batch({'input_ids':ids,'attention_mask':mask,'responses':responses},
                {'data_source':['nq','hotpotqa']})
    before={key:value.clone() for key,value in batch.batch.items()}
    cfg=SimpleNamespace(data=SimpleNamespace(max_prompt_length=8192,get=lambda k,d:d))
    captured=[]
    def same_actor(teacher):
        captured.append(teacher)
        return SimpleNamespace(batch={'old_log_probs':torch.zeros_like(responses).float()})
    trainer=SimpleNamespace(config=cfg,tokenizer=tok,skill_provider=provider(),global_steps=1,
                           actor_rollout_wg=SimpleNamespace(compute_log_prob=same_actor))
    monkeypatch.setattr(hints,'collect_hints',lambda *a,**kw:([None,'LOCAL_SENTINEL'],['not_triggered','applied']))
    x,report=hints.score_teacher(trainer,batch,tmp_path,native_teacher=lambda b:pytest.fail('unexpected fallback'))
    teacher=captured[0].batch
    assert x.shape==responses.shape and report['applied']==1
    assert report['teacher_context_repacked']==int(overflow)
    assert all(torch.equal(batch.batch[key],value) for key,value in before.items())
    assert teacher['input_ids'].shape[1]==8192+2
    if overflow:
        fitted=tok.decode(teacher['input_ids'][1])
        assert 'KEEP_QUESTION' in fitted and 'RECENT_EVIDENCE' in fitted and 'KEEP_TURN_INSTRUCTIONS' in fitted
        assert 'OLDEST_HISTORY' not in fitted
        assert 'OLDEST_HISTORY' in tok.decode(batch.batch['input_ids'][1])
        assert fitted.startswith('[Privileged Skill Information]')
    assert torch.equal(teacher['responses'],responses)
    assert 'LOCAL_SENTINEL' not in tok.decode(teacher['input_ids'][0])
    assert 'LOCAL_SENTINEL' in tok.decode(teacher['input_ids'][1])
    assert 'LOCAL_SENTINEL' not in tok.decode(batch.batch['input_ids'][1])
    assert torch.equal(teacher['attention_mask'][:,-2:],mask[:,-2:])
    monkeypatch.setattr(hints,'collect_hints',lambda *a,**kw:([None,None],['not_triggered']*2))
    native=torch.ones_like(responses)
    value,report=hints.score_teacher(trainer,batch,tmp_path,native_teacher=lambda b:(native,{}))
    assert value is native and len(captured)==1 and report['actor_marker_positions']==0
    if overflow:
        trainer.config.data.max_prompt_length=4
        monkeypatch.setattr(hints,'collect_hints',lambda *a,**kw:([None,'LOCAL_SENTINEL'],['not_triggered','applied']))
        with pytest.raises(RuntimeError,match='budget cannot fit'):
            hints.score_teacher(trainer,batch,tmp_path,native_teacher=lambda b:pytest.fail('unexpected fallback'))
        assert len(captured)==1


@pytest.mark.parametrize('body,reason',[
    ('Search error: test HTTP 503','transport_or_http_error'),
    ('Error processing search results: invalid record','malformed_search_results'),
    ('Search request failed or timed out after retries.','unavailable_retriever'),
    ('Unknown API state (no response and no error message).','unavailable_retriever'),
    ('Doc 1: a page quoting Search error: test HTTP 503',None),
    ('No search results found.',None),
])
def test_only_native_error_envelopes_are_rejected(body,reason):
    assert failure_reason('<information>'+json.dumps({'result':body})+'</information>')==reason
    assert failure_reason(body) is None
    error=pickle.loads(pickle.dumps(RetrievalFailure('fixture',2)))
    assert error.receipt()=={'status':'invalid_retrieval_failure','reason':'fixture','row_in_batch':2}


@contextmanager
def failing_retriever(monkeypatch):
    from agent_system.environments.env_package.search.third_party.skyrl_gym.tools import search
    monkeypatch.setattr(search,'MAX_RETRIES',1)  # Only shorten the CPU fixture.
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(503)
            self.end_headers()
        def log_message(self,*args): pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/retrieve'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_native_http_failure_never_publishes_final_score(monkeypatch,tmp_path):
    from common import question_id
    from omegaconf import OmegaConf
    monkeypatch.syspath_prepend(str(UPSTREAM))
    from agent_system.environments.env_manager import SearchEnvironmentManager
    from agent_system.environments.env_package.search import build_search_envs, search_projection
    question={'question':'Synthetic fixture','ground_truth':{'target':['Paris']},'data_source':'nq'}
    events=[]
    class Native:
        def _validate(self):
            self.traj_collector.multi_turn_loop(gen_batch=SimpleNamespace(non_tensor_batch={'env_kwargs':[question]}))
            pytest.fail('An invalid retrieval must interrupt before the native score')
    original=ModuleType('verl.trainer.ppo.skillsd_ray_trainer')
    original.SkillSDRayTrainer=Native
    monkeypatch.setitem(sys.modules,original.__name__,original)
    def import_file(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    adapter=import_file('grpo_eval',ROOT/'src/grpo_eval.py')
    monkeypatch.setitem(sys.modules,'grpo_eval',adapter)
    method=import_file('rethink_searchqa.test_actual_trainer',ROOT/'src/rethink_searchqa/trainer.py')
    receipt=tmp_path/'test_receipt.json'
    receipt.write_text(json.dumps({'files':{'test':{'rows':{'count':1,
        'question_ids':[question_id(question)],'source_counts':{'nq':1}}}}}))
    monkeypatch.setenv('GRPO_RUN_DIR',str(tmp_path/'run'))
    monkeypatch.setenv('QA_DATA_RECEIPT',str(receipt))
    monkeypatch.setenv('QA_EVAL_MODE','auto')
    with failing_retriever(monkeypatch) as url:
        cfg=OmegaConf.create({'env':{'history_length':4,'max_steps':4,
            'search':{'search_url':url,'topk':3,'timeout':4,'log_requests':False}},
            'trainer':{'val_only':False,'total_training_steps':150}})
        env=build_search_envs(seed=0,env_num=1,group_n=1,is_train=True,env_config=cfg.env)
        manager=shell.wire_environment(SearchEnvironmentManager(env,search_projection,cfg),capture_hints=False)
        try:
            def collect(**kwargs):
                events.append('retrieve')
                manager.reset([question])
                manager.step(['<action>search[fixture query]</action>'])
                pytest.fail('Expected the real HTTP 503 to be rejected')
            trainer=method.RethinkSearchTrainer.__new__(method.RethinkSearchTrainer)
            trainer.config=cfg
            trainer.global_steps=150
            trainer._save_checkpoint=lambda:events.append('save')
            trainer.val_dataset,trainer.val_dataloader=object(),object()
            trainer._searchqa_evaluation_loaders={'full':(trainer.val_dataset,trainer.val_dataloader)}
            trainer.traj_collector=SimpleNamespace(multi_turn_loop=collect)
            with pytest.raises(RetrievalFailure,match='transport_or_http_error'):
                trainer._validate()
            assert trainer.traj_collector.multi_turn_loop is collect
        finally:
            env.close()
    assert events==['save','retrieve']
    output=tmp_path/'run/evaluation'
    assert not list(output.rglob('summary.json')) and not (output/'latest.json').exists()
    invalid=list(output.rglob('invalid-*.json'))
    assert len(invalid)==1 and json.loads(invalid[0].read_text())=={
        'status':'invalid_retrieval_failure','reason':'transport_or_http_error',
        'row_in_batch':0,'scope':'full','update':150}


def test_start_pin_has_only_model_files_and_licenses():
    m=load('sft800_3b.json')
    assert m['completed_updates']==800 and m['protocol']==shell.STYLE
    assert m['intro_digest']==shell.INTRO_DIGEST and m['marker_id']==shell.MARKER_ID
    assert len([n for n in m['files'] if n.endswith('.safetensors')])==7
    assert all(0<s['bytes']<2**31 and re.fullmatch('[a-f0-9]{64}',s['sha256']) for s in m['files'].values())
    assert 'training_state.pt' not in m['files'] and set(m['notices'])=={'LICENSE-QWEN-3B.txt','NOTICE.txt'}
