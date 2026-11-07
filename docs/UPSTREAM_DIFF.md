# Search-QA: exact differences from the previously released GRPO+OPSD control

Use the existing Search-QA control, **not** a fresh invocation of the SDAR
example script. Its public SDAR revision is
`29dcb12a55d8fc17ba6ae03fd0a69048f43051f3`. Both sizes retain the identical
data, retrieval, evaluation manifests, native GRPO/SDAR loss, optimizer, masks,
reference KL 0.001, entropy 0.001, invalid-action penalty 0.01 and policy coefficient 1.
The original GRPO+OPSD example uses beta 0; the comparison contract already
uses **beta 5** and OPSD coefficient **0.01** in all applicable Search-QA arms.

1. **Initialization.** A same-size wrapped SFT800 Actor replaces the original
   base. No Stage-1 optimizer state is restored into Stage 2.
2. **One prompt/decoder adapter.** Preserve native history and observations.
   Use the identical Stage-1 optional marker instruction and the common action
   shell. Undo only the shell before native projection. Legacy native tags
   continue to be accepted by the original projector. Reasoning format is a
   diagnostic, not a new reward, execution gate or forced prefix.
3. **Teacher information.** Raw generated IDs at active positions select targets.
   One GLM-5.3-Flash request merges targets within a trajectory, using the final
   ALFWorld transport/JSON contract with the existing Search-QA hint vocabulary.
   It sees Actor-visible H/R, executed search/answer and observed feedback;
   no verifier references. The API neither judges nor changes route selections.
   The native Teacher hook appends validated hints only to those rows' native
   static skills. Teacher scores the exact same response IDs and mask using the
   current Actor. No separate Teacher model, action replacement or new loss.
4. **Failure boundaries.** Invalid returned hint content falls back to the
   native static-skill Teacher and is counted. Transient API failures retain the
   batch and retry before an update. Permanent configuration, auth and cache
   failures stop visibly. As in ALFWorld/WebShop, an accepted hint must survive
   intact. Only when the native left truncation would cut that hint, preserve
   the chosen static skill and complete hint by trimming the oldest **Teacher-only**
   history within the same 4096-token budget, retaining the visible question,
   current turn instructions and most recent available evidence. Student input,
   sampled responses, masks, native skill selection and loss are unchanged.
   The number of such rows is reported as `rethink/teacher_context_repacked`. If the complete
   prefix still cannot fit, stop before an update. Successful requests are cached
   per run. Evaluation never captures or requests hints.
5. **Retrieval validity.** Native search retries, tool execution and EM stay
   unchanged. After retries are exhausted, the shared Search-QA adapter rejects
   explicit native retriever error observations before they become training
   data or scores. Valid empty search results still count normally. A failed
   evaluation writes an invalid-attempt receipt and publishes neither a score
   nor a new latest pointer. The final checkpoint is already saved; after
   fixing retrieval, resume only the final evaluation, not update 151.

The upstream patch changes the Trainer import and adds rollout metadata capture.
The inherited training/evaluation code still calls the pinned native teacher
builder. The patch does not edit native reward, actor update, skills, search
environment, prompts, retriever or the algorithm source. The exact shared
448/51,713 schedules, ordered question IDs and verified native checkpoint
retention remain unchanged. A baseline runs from its separate public repository,
with its original native actions.
