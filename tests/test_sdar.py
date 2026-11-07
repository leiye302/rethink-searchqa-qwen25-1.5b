"""Execute pinned SDAR functions on tiny CPU tensors, without model weights."""
import ast
import copy
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Dict, List, Optional
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from common import UPSTREAM, load
from run import config, assert_sdar_only


def functions(relative, names, namespace):
    source = UPSTREAM/relative
    nodes = [n for n in ast.parse(source.read_text(encoding='utf-8')).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


def losses(torch):
    scope = {'torch':torch, 'np':np,
        'verl_F':SimpleNamespace(masked_mean=lambda x,m:(x*m).sum()/m.sum())}
    functions('verl/trainer/ppo/core_algos.py', ['agg_loss','compute_policy_loss'], scope)
    return functions('verl/trainer/ppo/sdar_utils.py', ['compute_sdar_loss'], scope)


@pytest.mark.parametrize('field,value', [('sdar_coef',0.), ('gate_beta',0.), ('skill_all',True)])
def test_native_searchqa_sdar_recipe_not_silently_changed(tmp_path, field, value):
    cfg = config('1.5b', tmp_path/'model', tmp_path/'run')
    cfg.algorithm.sdar[field] = value
    with pytest.raises(RuntimeError, match='native Search-QA GRPO\\+OPSD'):
        assert_sdar_only(cfg)


def test_grpo_term_cannot_be_disabled(tmp_path):
    from omegaconf import OmegaConf
    cfg = config('3b', tmp_path/'model', tmp_path/'run')
    OmegaConf.update(cfg, 'actor_rollout_ref.actor.pg_loss_coef', 0., force_add=True)
    with pytest.raises(RuntimeError, match='GRPO policy loss'):
        assert_sdar_only(cfg)


def test_native_gate_and_teacher_detach_with_padding_and_real_update():
    torch = pytest.importorskip('torch')
    native = losses(torch)['compute_sdar_loss']
    student = torch.nn.Parameter(torch.tensor([[-2.,-2.,-2.,-2.]]))
    teacher = torch.tensor([[-1.,-3.,-2.,-1.]], requires_grad=True)
    mask = torch.tensor([[1.,1.,1.,0.]])
    gate = torch.sigmoid(5*(teacher.detach()-student.detach()))
    loss, metrics = native(student, teacher, mask, gate_beta=5.)
    expected = (gate*(teacher.detach()-student)*mask).sum()/mask.sum()
    assert torch.allclose(loss, expected)
    assert gate[0,0] > .99 and gate[0,1] < .01 and gate[0,2] == .5
    loss.backward()
    assert teacher.grad is None
    assert torch.allclose(student.grad, -gate*mask/mask.sum())
    assert student.grad[0,3] == 0
    old = student.detach().clone()
    torch.optim.SGD([student], lr=.1).step()
    assert student[0,0] > old[0,0] and student[0,3] == old[0,3]
    assert metrics['sdar/gate_active_ratio'] == pytest.approx(1/3)


def test_actual_actor_branch_adds_sdar_without_replacing_grpo(monkeypatch):
    torch = pytest.importorskip('torch')
    native = losses(torch)
    student = torch.nn.Parameter(torch.full((2,3),-1.))
    teacher = torch.tensor([[-.8,-1.2,-.7],[-1.1,-.9,-1.3]], requires_grad=True)
    mask = torch.tensor([[1.,1.,0.],[1.,1.,0.]])
    advantages = torch.tensor([[1.,1.,0.],[-1.,-1.,0.]])
    pg = native['compute_policy_loss'](student.detach().clone(),student,advantages,mask,cliprange=.2)[0]
    pg_grad = torch.autograd.grad(pg, student, retain_graph=True)[0]
    module = ModuleType('verl.trainer.ppo.sdar_utils')
    module.compute_sdar_loss = native['compute_sdar_loss']
    monkeypatch.setitem(sys.modules,'verl.trainer.ppo.sdar_utils',module)
    path = UPSTREAM/'verl/workers/actor/dp_actor.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    branch = next(n for n in ast.walk(tree) if isinstance(n,ast.If)
        and ast.unparse(n.test) == "self.config.get('use_sdar_loss', False)")
    scope = {'self':SimpleNamespace(config={'use_sdar_loss':True,'sdar_gate_beta':5.,'sdar_loss_coef':.01}),
        'data':{'teacher_log_probs':teacher}, 'log_prob':student, 'response_mask':mask,
        'loss_agg_mode':'token-mean', 'policy_loss':pg, 'metrics':{}}
    exec(compile(ast.Module(body=[branch],type_ignores=[]),str(path),'exec'),scope)
    scope['policy_loss'].backward()
    expected = pg_grad - .01*torch.sigmoid(5*(teacher.detach()-student.detach()))*mask/mask.sum()
    assert torch.allclose(student.grad,expected)
    assert teacher.grad is None and scope['metrics']['sdar/coef'] == .01


def provider():
    scope = {'os':os, 'json':json, 'Dict':Dict, 'List':List, 'Optional':Optional}
    functions('verl/trainer/ppo/rlsd_utils.py',
        ['load_skill_mapping','load_skill_content','SkillProvider'],scope)
    return scope['SkillProvider'](str(UPSTREAM/'skills/search'),skill_all=False)


def test_official_static_skill_loading_and_data_source_selection():
    skills = provider()
    assert len(skills.skill_contents) == 5
    general = skills.skill_contents['general_skills']
    selected = skills.get_privileged_info_from_data_source('hotpotqa','Question')
    assert selected.startswith(general)
    assert skills.skill_contents['multi_hop_reasoning'] in selected
    assert skills.skill_contents['direct_retrieval'] not in selected
    assert skills.skill_contents['entity_attribute_lookup'] in skills.get_privileged_info_from_data_source('popqa','Question')
    for source in ('nq', 'triviaqa'):
        assert skills.skill_contents['direct_retrieval'] in skills.get_privileged_info_from_data_source(source, 'Question')
    for source in ('2wikimultihopqa', 'musique', 'bamboogle'):
        assert skills.skill_contents['multi_hop_reasoning'] in skills.get_privileged_info_from_data_source(source, 'Question')
    assert skills.get_privileged_info_from_data_source('unknown','Unmatched question') == general
    lock = load('upstream.json')
    assert all('skills/search/'+name in lock['unchanged_files']
        for name in skills.skill_mapping['skill_files'].values())
    assert 'skills/search/skill_mapping.json' in lock['unchanged_files']


class ByteTokenizer:
    pad_token_id = 0
    def encode(self, text, add_special_tokens=False):
        return [x+1 for x in text.encode('utf-8')]
    def decode(self, ids, skip_special_tokens=False):
        return bytes(int(x)-1 for x in ids if int(x)>0).decode('utf-8')


def teacher_functions(torch):
    class Data:
        @staticmethod
        def from_dict(tensors): return SimpleNamespace(batch=tensors)
    scope = {'torch':torch,'DataProto':Data,'SkillProvider':object}
    functions('verl/utils/model.py',['compute_position_id_with_mask'],scope)
    functions('verl/trainer/ppo/rlsd_ray_trainer.py',['build_teacher_batch'],scope)
    path = UPSTREAM/'verl/trainer/ppo/rlsd_ray_trainer.py'
    cls = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.ClassDef) and n.name=='RLSDRayTrainer')
    method = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_compute_teacher_log_probs')
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),scope)
    return scope


@pytest.mark.parametrize('budget',[32,8192])
def test_native_teacher_only_prefix_current_actor_and_response_alignment(budget):
    torch = pytest.importorskip('torch')
    from omegaconf import OmegaConf
    codec = ByteTokenizer()
    prompt = 'User: who discovered penicillin?'
    p = codec.encode(prompt)
    response = torch.tensor([[900,901,0]])
    ids = torch.tensor([[0,0]+p+response[0].tolist()])
    mask = torch.tensor([[0,0]+[1]*len(p)+[1,1,0]])
    batch = SimpleNamespace(batch={'input_ids':ids,'attention_mask':mask,'responses':response},
        non_tensor_batch={'gamefile':np.array([None],dtype=object),'data_source':np.array(['nq']),
                          'env_kwargs':np.array([{'ground_truth': {'target': ['VERIFIER_ONLY_SENTINEL']}}], dtype=object)})
    original = {key:value.clone() for key,value in batch.batch.items()}
    skill_provider = provider()
    scope = teacher_functions(torch)
    calls = []
    actor = SimpleNamespace(version=1.)
    def score(teacher_batch):
        calls.append(teacher_batch)
        return SimpleNamespace(batch={'old_log_probs':torch.full_like(response, actor.version, dtype=torch.float)})
    actor.compute_log_prob = score
    trainer = SimpleNamespace(actor_rollout_wg=actor, skill_provider=skill_provider, tokenizer=codec,
        config=SimpleNamespace(data=OmegaConf.create({'max_prompt_length':budget,'truncation':'error'})))
    first = scope['_compute_teacher_log_probs'](trainer,batch)
    actor.version = 2.
    second = scope['_compute_teacher_log_probs'](trainer,batch)
    assert torch.all(first==1.) and torch.all(second==2.)  # Not a fixed reference checkpoint.
    expected_text = '[Privileged Skill Information]\n'+skill_provider.get_privileged_info_from_data_source('nq', prompt)+'\n\n'+prompt
    expected = codec.encode(expected_text)[-budget:]
    t = calls[0].batch
    actual = t['input_ids'][0,:budget][t['attention_mask'][0,:budget].bool()].tolist()
    assert actual == expected
    assert 'VERIFIER_ONLY_SENTINEL' not in codec.decode(actual)
    assert t['responses'] is response
    assert torch.equal(t['input_ids'][:,-3:],response)
    assert torch.equal(t['attention_mask'][:,-3:],mask[:,-3:])
    assert all(torch.equal(batch.batch[key],value) for key,value in original.items())


def test_validation_is_native_unprivileged_rollout_not_teacher_scoring():
    path = UPSTREAM/'verl/trainer/ppo/ray_trainer.py'
    cls = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.ClassDef) and n.name=='RayPPOTrainer')
    method = ast.unparse(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_validate'))
    assert 'multi_turn_loop' in method and 'is_train=False' in method
    assert 'teacher_log_probs' not in method and 'skill_provider' not in method
