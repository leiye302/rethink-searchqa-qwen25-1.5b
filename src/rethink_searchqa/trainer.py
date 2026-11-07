"""Add local Teacher context through the native pre-update Teacher hook."""
import os
from pathlib import Path
import uuid
from common import atomic_json
from grpo_eval import SearchTrainer
from . import hints
from .retriever_guard import RetrievalFailure
from .shell import assert_token, wire_environment


class RethinkSearchTrainer(SearchTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert_token(self.tokenizer)
        wire_environment(self.envs, capture_hints=not self.config.trainer.val_only)
        wire_environment(self.val_envs, capture_hints=False)

    def _validate_selected(self, scope, started):
        try:
            return super()._validate_selected(scope, started)
        except RetrievalFailure as error:
            destination = (Path(os.environ['GRPO_RUN_DIR']) / 'evaluation'
                           / f'update_{self.global_steps:03d}' / ('invalid-' + uuid.uuid4().hex + '.json'))
            atomic_json(destination, {**error.receipt(), 'scope': scope, 'update': int(self.global_steps)})
            raise

    def _get_training_teacher_log_probs(self, batch):
        result, report = hints.score_teacher(
            self, batch, Path(os.environ['GRPO_RUN_DIR']) / 'hints',
            native_teacher=super()._get_training_teacher_log_probs,
        )
        return result, {
            'rethink/actor_marker_decisions': report['actor_marker_positions'],
            'rethink/local_hint_applied': report['applied'],
            'rethink/local_hint_fallback': report['actor_marker_positions'] - report['applied'],
            'rethink/teacher_context_repacked': report['teacher_context_repacked'],
        }
