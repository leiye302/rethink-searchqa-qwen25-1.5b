"""Shared GRPO monitoring/reporting around the untouched native SkillSD trainer."""
import json
import os
from pathlib import Path
import time
import uuid
from common import atomic_json, question_id
from checkpoint_io import NativeCheckpointMixin
from evaluation_contract import summarize_questions, validation_scope
from verl.trainer.ppo.skillsd_ray_trainer import SkillSDRayTrainer


class SearchTrainer(NativeCheckpointMixin, SkillSDRayTrainer):
    def fit(self):
        from verl.utils.tracking import Tracking
        original = Tracking.log
        destination = Path(os.environ['GRPO_RUN_DIR']) / 'metrics.jsonl'

        def capture(tracker, data, step, backend=None):
            original(tracker, data=data, step=step, backend=backend)
            with destination.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'step': int(step), 'metrics': data},
                    default=lambda x: x.item(), allow_nan=False) + '\n')
                stream.flush()

        Tracking.log = capture
        try:
            return super().fit()
        finally:
            Tracking.log = original

    def _evaluation_loader(self, scope):
        if not hasattr(self, '_searchqa_evaluation_loaders'):
            initial = os.environ['QA_INITIAL_EVAL_SCOPE']
            self._searchqa_evaluation_loaders = {initial: (self.val_dataset, self.val_dataloader)}
        if scope not in self._searchqa_evaluation_loaders:
            # Construct ONLY a validation dataset/loader. Never call the native
            # _create_dataloader again: that would recreate the training sampler.
            from verl.trainer.main_ppo import create_rl_dataset
            data_dir = Path(os.environ['QA_DATA_DIRECTORY'])
            path = data_dir / ('test.parquet' if scope == 'full' else 'validation_subset.parquet')
            dataset = create_rl_dataset(str(path), self.config.data, self.tokenizer, self.processor)
            loader = type(self.val_dataloader)(
                dataset=dataset, batch_size=self.config.data.val_batch_size,
                num_workers=self.config.data.get('dataloader_num_workers', 8),
                shuffle=False, drop_last=False, collate_fn=self.val_dataloader.collate_fn)
            if len(loader) < 1:
                raise RuntimeError('Selected evaluation loader is empty')
            self._searchqa_evaluation_loaders[scope] = dataset, loader
        return self._searchqa_evaluation_loaders[scope]

    def _validate(self):
        scope = validation_scope(int(self.global_steps), os.environ.get('QA_EVAL_MODE', 'auto'))
        if (scope == 'full' and os.environ.get('QA_EVAL_MODE', 'auto') == 'auto'
                and not self.config.trainer.val_only
                and int(self.global_steps) == self.config.trainer.total_training_steps):
            # Native fit validates before its final save. Publish and SHA-verify
            # the completed update first; a failed full test must not lose it.
            self._save_checkpoint()
        started = time.monotonic()
        original_dataset, original_loader = self.val_dataset, self.val_dataloader
        self.val_dataset, self.val_dataloader = self._evaluation_loader(scope)
        try:
            return self._validate_selected(scope, started)
        finally:
            self.val_dataset, self.val_dataloader = original_dataset, original_loader

    def _validate_selected(self, scope, started):
        run = Path(os.environ['GRPO_RUN_DIR'])
        if scope == 'full':
            receipt = json.loads(Path(os.environ['QA_DATA_RECEIPT']).read_text(encoding='utf-8'))
            expected = receipt['files']['test']['rows']
            source_test_indices = list(range(expected['count']))
        else:
            receipt = json.loads(Path(os.environ['QA_SUBSET_RECEIPT']).read_text(encoding='utf-8'))
            expected = receipt['rows']
            source_test_indices = receipt['source_test_indices']
        destination = run / 'evaluation' / f'update_{self.global_steps:03d}' / ('attempt-' + uuid.uuid4().hex[:8])
        destination.mkdir(parents=True, exist_ok=False)
        original = self.traj_collector.multi_turn_loop
        questions, seen_ids = {}, []
        batches = 0

        def capture(*args, **kwargs):
            nonlocal batches
            incoming = kwargs['gen_batch']
            identities = [question_id(row) for row in incoming.non_tensor_batch['env_kwargs']]
            offset = len(seen_ids)
            if identities != expected['question_ids'][offset:offset + len(identities)]:
                raise RuntimeError('Actual validation batch differs from the fixed ' + scope + ' order')
            # Native _validate.pop does NOT pass dataset "index" to the
            # collector. Derive scope-local positions only after verifying the
            # actual question/target/source identities against the receipt.
            # The collector's returned "index" is local to THIS input batch.
            input_indices = list(range(offset, offset + len(identities)))
            input_ids = dict(zip(input_indices, identities))
            batch = original(*args, **kwargs)
            nt = batch.non_tensor_batch
            records = []
            for i in range(len(batch.batch['responses'])):
                local_index = int(nt['index'][i])
                if not 0 <= local_index < len(input_indices):
                    raise RuntimeError('Output trajectory lost its input-question identity')
                index = input_indices[local_index]
                reward = float(nt['episode_rewards'][i])
                episode = {'index': index, 'question_id': input_ids[index],
                    'data_source': str(nt['data_source'][i]), 'episode_reward': reward,
                    'trajectory_id': str(nt['traj_uid'][i])}
                if index in questions and questions[index] != episode:
                    raise RuntimeError('Inconsistent episode reward, source or trajectory identity')
                questions[index] = episode
                ids = batch.batch['responses'][i]
                mask = batch.batch['attention_mask'][i, -ids.shape[-1]:].bool()
                tokens = ids[mask].cpu().tolist()
                records.append(dict(episode, original_test_index=source_test_indices[index],
                    step=int(nt['turn_step'][i]), token_ids=tokens,
                    text=self.tokenizer.decode(tokens, skip_special_tokens=False),
                    action_valid=bool(nt['is_action_valid'][i])))
            atomic_json(destination / f'batch_{batches:03d}.json', records)
            seen_ids.extend(identities)
            batches += 1
            return batch

        self.traj_collector.multi_turn_loop = capture
        try:
            native_metrics = super()._validate()
        finally:
            self.traj_collector.multi_turn_loop = original
        if seen_ids != expected['question_ids']:
            raise RuntimeError('Native validation did not traverse every selected ' + scope + ' question')
        summary = summarize_questions(list(questions.values()), expected['question_ids'], expected['source_counts'])
        elapsed = time.monotonic() - started
        summary.update({'status': 'completed', 'scope': scope, 'update': int(self.global_steps), 'batches': batches,
            'wall_seconds': elapsed, 'questions_per_second': len(questions) / elapsed if elapsed > 0 else None,
            'monitor_target_seconds': 600 if scope == 'subset' else None,
            'monitor_target_is_not_a_timeout': True,
            'native_metrics_unchanged': native_metrics,
            'note': 'Subset monitoring and final full-test results are distinct scopes. Question EM does not redefine native decision-row logs.'})
        atomic_json(destination / 'summary.json', summary)
        atomic_json(run / 'evaluation/latest.json', {'path': str(destination.relative_to(run)),
                    'update': int(self.global_steps), 'scope': scope})
        # Preserve each native metric; add separately named question-level curves
        # so a 448-question monitor point cannot be mistaken for a full-test point.
        reported = dict(native_metrics)
        prefix = 'eval/' + scope + '/'
        reported.update({prefix + 'question_micro_em': summary['question_micro_em'],
            prefix + 'dataset_macro_em': summary['dataset_macro_em'], prefix + 'questions': len(questions),
            prefix + 'wall_seconds': elapsed})
        reported.update({prefix + name + '_question_em': record['em']
                         for name, record in summary['by_dataset'].items()})
        return reported
