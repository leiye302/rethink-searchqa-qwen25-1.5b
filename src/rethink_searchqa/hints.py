"""Small SDAR hooks. Route selection is ONLY generated marker token IDs."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import re
import time
from types import SimpleNamespace
from common import atomic_json
from .hint_api import fetch, HintAPIHold, HintAPIFatal
from .history_display import INTRO as HISTORY_INTRO
from .shell import MARKER_ID, assert_token

HINT_BEGIN='[Actor-triggered local hint]'
HINT_END='[End local hint]'
TURN_INTRO="\nNow it's your turn to respond for the current step.\n"


def capture(records, tokenizer, batch, next_obs, infos, active, step):
    assert_token(tokenizer)
    responses=batch.batch['responses']
    length=responses.shape[1]
    attention=batch.batch['attention_mask'].bool()
    masks=attention[:,-length:]
    prompts=batch.batch['input_ids'][:,:-length]
    prompt_masks=attention[:,:-length]
    for i in range(len(records)):
        if not bool(active[i]):
            continue
        ids=responses[i][masks[i].bool()].detach().cpu().tolist()
        text=tokenizer.decode(ids,skip_special_tokens=False)
        reasoning=re.search(r'<think>([\s\S]*?)</think>',text)
        # Read the effective, possibly truncated H and the actual response IDs.
        # Do not pass env_kwargs/ground_truth or a detector's judgement.
        h=tokenizer.decode(prompts[i][prompt_masks[i]].detach().cpu().tolist(),skip_special_tokens=False)
        records[i].append({'step':step,'reasoning':reasoning[1].strip() if reasoning else text,
              'triggered':MARKER_ID in ids,'admissible_actions':[],
              'action':str(infos[i]['postprocessed_action']),
              'observation':h,'feedback':str(next_obs['anchor'][i])})


def attach(records, batch_list, tasks):
    for index,(episode,rows) in enumerate(zip(records,batch_list,strict=True)):
        if not episode:
            continue
        task=tasks[index]
        if not isinstance(task,str):
            raise TypeError('Search-QA hint task must be the visible question only')
        payload=(json.dumps({'steps':episode,'benchmark':'search','task':task},
                           ensure_ascii=False,separators=(',',':'))
                 if any(r['triggered'] for r in episode) else '')
        by_step={r['step']:r for r in episode}
        for step,row in enumerate(rows):
            row['rethink_episode_json']=payload
            row['rethink_step']=step
            row['rethink_triggered']=bool(by_step.get(step,{}).get('triggered',False))


def collect_hints(batch,cache,timeout=120,retries=5,workers=4):
    # Metadata follows each row through SDAR's duplication/balancing permutation.
    groups={}; mapping=[]
    for raw,step,triggered in zip(batch.non_tensor_batch['rethink_episode_json'],
            batch.non_tensor_batch['rethink_step'],batch.non_tensor_batch['rethink_triggered']):
        if triggered:
            groups.setdefault(str(raw),json.loads(str(raw)))
        mapping.append((str(raw),int(step),bool(triggered)))
    if workers<=0:
        raise HintAPIFatal('api_invalid_worker_count',{'stage':'configuration'})
    results={};started=time.perf_counter()
    def status(state,**details):
        value={'status':state,'time':time.time(),
               'elapsed_seconds':round(time.perf_counter()-started,3),
               'episode_count':len(groups),'completed_episodes':len(results),
               'same_batch_retained':True,'actor_updated':False,**details}
        print('[RETHINK API batch] '+json.dumps(value,sort_keys=True),flush=True)
        try:atomic_json(Path(cache)/'api_wait_status.json',value)
        except OSError:
            raise HintAPIFatal('api_status_write_error',{'stage':'cache_write','actor_updated':False}) from None
    while len(results)<len(groups):
        # A writable status/cache directory is checked BEFORE issuing requests.
        # Successful episodes survive another episode's retry in memory as well
        # as on disk; neither cached hints nor the rollout are regenerated.
        status('requesting')
        failure=None
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending={pool.submit(fetch,episode,cache,timeout=timeout,retries=retries):raw
                     for raw,episode in groups.items() if raw not in results}
            try:
                while pending:
                    done,_=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
                    if not done:
                        status('waiting_inflight' if failure else 'waiting_requests',
                               pending_episodes=len(pending),
                               reason=failure.reason if failure else None)
                    for future in done:
                        raw=pending.pop(future)
                        try:results[raw]=future.result()
                        except HintAPIHold as exc:
                            if failure is None or exc.fatal:failure=exc
                            status('failed' if exc.fatal else 'waiting_inflight',
                                   reason=exc.reason,fatal=exc.fatal,diagnostics=exc.diagnostics,
                                   pending_episodes=len(pending))
                            # Cancel queued work, but join already-running HTTP
                            # requests; never leave a second pool in the background.
                            for other in list(pending):
                                if other.cancel():pending.pop(other)
            finally:
                for future in pending:future.cancel()
        if failure is not None:
            if failure.fatal:
                status('failed',reason=failure.reason,fatal=True,diagnostics=failure.diagnostics)
                raise failure from None
            delay=60 if failure.temporary else 300
            status('waiting_retry' if failure.temporary else 'waiting_quota_or_auth',
                   reason=failure.reason,diagnostics=failure.diagnostics,retry_after_s=delay)
            time.sleep(delay)
    if groups:status('ready')
    hints=[];reasons=[]
    for raw,step,triggered in mapping:
        if not triggered:
            hints.append(None); reasons.append('not_triggered'); continue
        values,failures=results[raw]
        hints.append(values.get(step))
        reasons.append('applied' if step in values else failures.get(step,failures.get('episode','missing_step')))
    return hints,reasons


class RowSkillProvider:
    """Delegate native skill selection unchanged, append only this row's hint."""
    def __init__(self,parent,hint):
        self.parent=parent; self.hint=hint; self.candidate=None

    def __getattr__(self,name):
        original=getattr(self.parent,name)
        if not name.startswith('get_privileged_info'):
            return original
        def wrapped(*args,**kwargs):
            if self.candidate is None:
                static=original(*args,**kwargs)
                self.candidate=static+'\n\n'+HINT_BEGIN+'\n'+self.hint+'\n'+HINT_END if self.hint else static
            return self.candidate
        return wrapped


def complete_hint(teacher, tokenizer, response_len, hint):
    mask=teacher.batch['attention_mask'][0,:-response_len].bool()
    ids=teacher.batch['input_ids'][0,:-response_len][mask]
    return HINT_BEGIN+'\n'+hint+'\n'+HINT_END in tokenizer.decode(ids,skip_special_tokens=False)


def fit_teacher_history(row, tokenizer, budget, skill_text):
    """Reserve the native static+hint prefix by trimming only Teacher's old H."""
    import torch
    responses=row.batch['responses']
    width=responses.shape[1]
    mask=row.batch['attention_mask']
    prompt=row.batch['input_ids'][0,:-width][mask[0,:-width].bool()]
    text=tokenizer.decode(prompt,skip_special_tokens=False)
    # The visible question precedes history in the pinned prompt. Preserve it
    # (if still present) and the current turn instructions, not just the tail.
    head,history_intro,rest=text.partition(HISTORY_INTRO)
    if history_intro:
        head+=history_intro
        history,turn,tail=rest.rpartition(TURN_INTRO)
    else:
        head=''
        history,turn,tail=text.rpartition(TURN_INTRO)
    if not turn:
        head,history,tail='',text,''
    if history.startswith('Your question: ') or '\nYour question: ' in history:
        raise RuntimeError('Native Teacher budget cannot fit without removing the visible question; no update')
    tail=turn+tail
    # Exactly the prefix used by the pinned native build_teacher_batch.
    prefix=f'[Privileged Skill Information]\n{skill_text}\n\n'
    old=tokenizer.encode(history,add_special_tokens=False)
    keep=min(len(old),budget-len(tokenizer.encode(prefix+head+tail,add_special_tokens=False)))
    while keep>=0:
        recent=tokenizer.decode(old[-keep:],skip_special_tokens=False) if keep else ''
        fitted=tokenizer.encode(head+recent+tail,add_special_tokens=False)
        # Verify exactly what the native builder will re-tokenize at the join.
        decoded=tokenizer.decode(fitted,skip_special_tokens=False)
        excess=len(tokenizer.encode(prefix+decoded,add_special_tokens=False))-budget
        if fitted and excess<=0:
            return SimpleNamespace(batch={
                'input_ids':torch.cat((prompt.new_tensor([fitted]),responses),dim=1),
                'attention_mask':torch.cat((mask.new_ones((1,len(fitted))),mask[:,-width:]),dim=1),
                'responses':responses},non_tensor_batch=row.non_tensor_batch)
        keep-=max(1,excess)
    raise RuntimeError('Native Teacher budget cannot fit the complete skill, hint and prompt; no update')


def score_teacher(trainer,batch,audit_dir,*,native_teacher):
    import torch
    from verl.trainer.ppo.rlsd_ray_trainer import build_teacher_batch
    cache=audit_dir/'cache'
    begin=time.perf_counter()
    hints,reasons=collect_hints(batch,cache,timeout=120,retries=5,workers=4)
    if len(hints) != len(batch.batch['responses']):
        raise ValueError('Actor hint rows do not match the native training batch')
    repacked=0
    if not any(hints):
        # Includes zero-trigger updates and complete schema/content fallback.
        result,_=native_teacher(batch)
    else:
        budget=trainer.config.data.max_prompt_length
        truncation=trainer.config.data.get('truncation','left')
        build=lambda source,provider:build_teacher_batch(
            source,provider,trainer.tokenizer,budget,truncation=truncation)
        teacher_batch=build(batch,trainer.skill_provider)
        response_len=batch.batch['responses'].shape[1]
        for i,hint in enumerate(hints):
            if not hint:
                continue
            row=batch.select_idxs([i])
            provider=RowSkillProvider(trainer.skill_provider,hint)
            teacher=build(row,provider)
            if not complete_hint(teacher,trainer.tokenizer,response_len,hint):
                # Replay the skill selected from the original H; do not select
                # a different static skill based on the cropped Teacher input.
                fitted=fit_teacher_history(row,trainer.tokenizer,budget,provider.candidate)
                teacher=build(fitted,provider)
                if not complete_hint(teacher,trainer.tokenizer,response_len,hint):
                    raise RuntimeError('Native Teacher did not retain the complete local hint; no update')
                repacked+=1
            for k in ('input_ids','attention_mask','position_ids'):
                teacher_batch.batch[k][i]=teacher.batch[k][0]
        if (not torch.equal(teacher_batch.batch['responses'],batch.batch['responses'])
                or not torch.equal(teacher_batch.batch['input_ids'][:,-response_len:],batch.batch['responses'])
                or not torch.equal(teacher_batch.batch['attention_mask'][:,-response_len:],
                                   batch.batch['attention_mask'][:,-response_len:])):
            raise AssertionError('Teacher must score identical sampled response tokens and mask')
        # Same current Actor; the native worker scores without gradients.
        result=trainer.actor_rollout_wg.compute_log_prob(teacher_batch).batch['old_log_probs']
    report={'step':trainer.global_steps,'decisions':len(hints),
            'actor_marker_positions':sum(x!='not_triggered' for x in reasons),
            'applied':reasons.count('applied'),'fallback_reasons':{x:reasons.count(x) for x in sorted(set(reasons))},
            'teacher_context_repacked':repacked,
            'seconds':time.perf_counter()-begin,'analyzer_is_router':False,'student_sees_hints':False,
            'model':'glm-5.3-flash','protocol':'actor_only_episode_hints_glm_v2_20261015',
            'fixed_initial_teacher':False,'teacher_scoring':'public_pre_update_current_actor'}
    atomic_json(audit_dir/f'update_{trainer.global_steps:06d}.json',report)
    print('[RETHINK hint coverage] '+json.dumps(report),flush=True)
    return result,report
