# Frozen-model new-state confirmation

Primary question: on full MuJoCo / robosuite operation, does continuous decision
denoising-response supervision add utility beyond both image L1 and frozen
observation-encoder feature consistency?

Keep BOTH tasks (Can, Square) and ALL training seeds (3072,3073,3074). Retain
the twelve completed image/decision adapters and original policies byte for byte.
No new best-checkpoint selection, seed filtering, policy retraining, noise scan,
or assumed statistical significance. Prior datasets and original source files
remain unchanged. No additional dependency installation is needed.

New work: six feature-control models, same 267195-parameter restorer, training
data and 2000-update budget, batch16, AdamW lr0.0003/wd0.0001, clip1,
20% identity + 80% exposure0.25/Poisson gain0.004/read0.008. Same sampling and
sensor RNG streams as the corresponding image/decision training seed.

Feature weight is chosen ONCE per task using four fixed TRAIN-only batches,
at the same seed3072 initial restorer: median ratio of gradients matching the
existing weighted decision auxiliary loss (0.1 x MSE). No simulator results
are used for calibration. Actual per-component gradients are logged at updates
1,500,1000,1500,2000. This is a mechanism-level supervision control, not a full
reimplementation of SR4IR, EDTR, DBC or any external SOTA method.

Before rollouts, lock all eighteen adapters, original policies and new state
banks with hashes. Lowlight: 100 new states, IDs30000..30099. Clean: the first
50 of the SAME 100-entry bank, selected before inspecting outcomes. Shared
controls original-clean50/original-lowlight100/exposure-oracle-lowlight100 per
task are evaluated once (500 episodes). Learned arms image/feature/decision,
three training seeds, two tasks, clean50 + lowlight100 (2700 episodes).
Total: **3200 complete 400-step episodes**, 640 five-episode shards.

The physics/action/camera contract is unchanged: Panda, robosuite1.2.0 /
MuJoCo2.1, two84x84 RGB views, two-frame history, 9D proprioception, absolute
pose actions and 100-step DDPM. Noise is applied to each physical image once;
no global synthetic structural-noise grid. Synthetic RGB sensor-inspired
noise is not calibrated real-camera RAW data or hardware validation.

Success comes from EnvRobosuite.is_success()['task'] -> robosuite._check_success()
on MuJoCo object and gripper positions. Ever-success, final400-step success,
first-success step, wall/inference timings are kept separate. Square uses
the benchmark's geometry tolerances, not precision-insertion certification.
All physical initialization fingerprints must match across methods and seeds.

The report includes every group, every seed, paired gains/losses, exact
McNemar p and Holm adjustment across12 lowlight comparisons per metric.
Ever-success is primary; final-success is a separate secondary family.
95% state-cluster bootstrap intervals preserve state reuse across the three
training seeds and are explicitly conditional on these three fixed models.
Do not pool repeated states as independent training runs or extend until p<.05.

Automatic idle-GPU selection excludes occupied cards; one trainer/GPU,
up to four evaluators/GPU, shared advisory GPU locks. Docker takes a single
device ID per container. Keyboard interruption stops only this suite's owned
n2d-confirm-* containers, NOT other jobs or tmux. Resume with the same command;
completed checkpoints and shard episodes are validated and reused.

```bash
cd ~/n2d_dreamerv3
python3 -u task_restore_confirm_v1/run_confirm.py --stage full --gpus auto --workers-per-gpu 4 --tag decision_confirm_v1_20261004
```

Preflight instead of training: `--stage preflight`. After completion, re-summarize
with `--stage summary --tag decision_confirm_v1_20261004`.

Outputs: `outputs/task_restore_confirm_v1/decision_confirm_v1_20261004/`.
The local single experiment ledger is maintained separately; no experimental
results prose is added to the paper during this batch.
