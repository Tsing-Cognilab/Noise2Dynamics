# Fixed-budget adapter training replication

Reference: `outputs/task_restore_method_v1/decision_pilot_v1_20261003`, seed3072.

Only change: adapter training seeds **3073 and 3074**. Can and Square; image-only
and image+continuous-decision-supervision; 2000 updates each; batch16; same
frozen policies, 267195-parameter causal two-frame shared-camera restorer,
AdamW learning rate0.0003/weight decay0.0001, decision weight0.1, synthetic
exposure0.25/shot gain0.004/read0.008, identity probability0.2, and last
fixed-budget checkpoint. Original method source files must remain byte-identical.

Reuse the exact original replayed RGB/teacher dataset (24 train,4 validation
whole demonstrations, stride4), so no rerendering, no data change and no
new test-driven selection. The launcher verifies training specification
against3072 after every training job, allowing only the seed field to differ.

Each new seed: two tasks x two learned methods x two conditions(clean/lowlight)
x50 full400-step episodes =400 new episodes. Both seeds total800 new episodes.
Evaluation seed20000..20049, same XML/state banks and policy/noise RNG protocol.
Original-policy(clean/lowlight) and known-exposure-oracle(lowlight) controls
have no adapter-training seed: read existing validated3072 controls without
rerunning, copying or modifying them. This is not a fresh held-out-state test.

One host queue owns GPU locks shared with previous launchers. GPU4 is currently
busy with an unrelated job and automatic allocation excludes it. One training
worker/GPU, four evaluation workers/GPU. Docker receives one device ID per
container. On interrupt only owned `n2d-repeat-*` containers are stopped;
no tmux exit/kill command. Rerun the same full command to resume retained
training checkpoints and completed episode shards. Do not run two copies.

Final report includes all three training seeds individually, paired gains/losses
and exact per-seed paired tests, clean-input behavior, and mean/sample SD over
training seeds. No pooled150-episode independence assumption, no best seed.
Ever-success and final-success remain separate. No claim of real-camera or
physical-robot validation, or of generalized superiority from three seeds.

User-run command on server:

```bash
cd ~/n2d_dreamerv3
python3 -u task_restore_repeat_v1/run_repeat.py --stage full --gpus auto --seeds 3073,3074 --tag decision_repeat_v1_20261003
```

Summary only, after completion:

```bash
python3 -u task_restore_repeat_v1/run_repeat.py --stage summary --tag decision_repeat_v1_20261003
```
