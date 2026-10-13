# Task-preserving visual restoration: first closed-loop baseline

Status: deployment and bounded smoke only. Proposed restoration method is not yet trained. This stage measures whether sensor-inspired corruption damages competent manipulation policies.

## Fixed resources

- Directory: `/data/lizhenhao/lizhenhao/n2d_dreamerv3/task_restore_v1`.
- Isolated image: `noise2dynamics:taskrestore-v1`; robosuite pinned to official Diffusion Policy commit `277ab9588ad7a4f4b55cf75508b44aa67ec171f0` (version 1.2.0), robomimic 0.2.0, free-mujoco-py/MuJoCo 2.1, PyTorch 2.5.1. Legacy CPU OSMesa rendering; GPU policy inference.
- Policies: publicly released Diffusion Policy Transformer, PH datasets, published `train_0`, EMA weights. Highest officially published clean-score checkpoint, tie earliest epoch, selected before our noise evaluation. Source URLs and SHA256 are in `assets/manifest.json`. No checkpoint selection using our corrupted test results.
- Tasks: Lift, PickPlaceCan, NutAssemblySquare; official original demo HDF5 supplies environment metadata only.
- Policy inputs: two 84x84 RGB views, EEF position/quaternion and gripper joints. No object-state input. Two-frame observation history, 8 executed actions per prediction, original 100-step DDPM inference and 400 simulator control steps per episode.
- Absolute 10D policy outputs are converted to 7D simulator commands using the official first-two-rows rotation_6d convention. The OSC controller runs with `control_delta=False`.

## Corruption protocol

RGB is approximately linearized, then receives signal-dependent Poisson and additive Gaussian read noise, clipping, sRGB transfer and 8-bit quantization. This is synthetic, not RAW-camera calibration. Each physical frame is corrupted once and retained in observation history. Separate NumPy generators isolate sensor randomness from robot resets and policy sampling.

|Condition|Exposure|Poisson gain|Read sigma|Local correlation|
|---|---:|---:|---:|---|
|clean|1|0|0|none|
|shot_read_mild|1|.002|.004|none|
|shot_read_medium|1|.008|.012|none|
|shot_read_strong|1|.02|.025|none|
|lowlight|.25|.004|.008|none|
|isp_local|1|.008|.012|variance-normalized 3x3 residual filtering|

The lowlight condition actually reduces signal exposure; it does not only increase noise. ISP-local is a residual correlation approximation, not a complete camera ISP.

## Evaluation and integrity

First problem-evidence run: three tasks x clean / shot_read_medium / lowlight x 50 independent reset seeds = 450 episodes. Mild, strong and ISP conditions are fixed in advance for the later full curve/transfer evaluation; do not tune them to proposed-method test results.

- Seeds 10000..10049, common initial states, sensor streams and policy sampling seeds across conditions. The launcher first prepares a shared bank of full MuJoCo model XML plus float64 simulator states, then each evaluator calls official `reset_to`. Reset seeding alone is not sufficient: legacy Lift samples cube size at environment construction.
- Geometry is fixed once per task with a separate construction seed; the 50 robot/object initial states are independent. This experiment does not establish cross-geometry generalization. The bank is immutable, hashed, and never selected based on noisy-policy success.
- Report official task success ever achieved and final task success separately; no reward-to-success substitution.
- Bank-to-restored state is checked within eight float64 eps: legacy MuJoCo forward normalization can alternate the lowest quaternion bits (measured max 2.22e-16 in Can). The error is recorded. BETWEEN CONDITIONS the actual episode state, restored model XML and shared-bank entry SHA256 must remain bitwise identical; the summarizer rejects mismatches without hash rounding. This tolerance cannot hide the earlier 6.74e-5 Lift geometry discrepancy.
- Save every episode, completion time, inference time, first success time and two sample videos per condition. Videos display clean simulator render on the left and actor's corrupted input on the right. The clean reference panel is never passed to the actor.
- Wilson intervals refer to independent evaluation episodes, not multiple policy training seeds. Underlying policy `train_0` is fixed; this stage does not establish seed-robust method improvement.
- Resume uses exactly the same protocol, evaluator hash and episode IDs. Existing incompatible tags are rejected. Incomplete JSONL is rejected rather than silently discarded.
- GPU locks prevent duplicate launchers in this directory; preflight rejects cards already using >2 GiB. Only owned containers can be stopped on Ctrl+C. No tmux exit/kill commands.

## Server commands

All code, policies and dependencies are deployed; long baseline evaluation is user-launched. The launcher automatically prepares 50 shared states per task before GPU evaluations, with no separate setup command required.

```bash
cd ~/n2d_dreamerv3
python3 task_restore_v1/run_parallel.py --stage preflight --gpus auto --workers-per-gpu 4
python3 -u task_restore_v1/run_parallel.py --stage baseline \
  --gpus auto --workers-per-gpu 4 --shard-size 5 \
  --conditions clean,shot_read_medium,lowlight \
  --episodes 50 --tag sensor_baseline_parallel_v1_20261003
```

To resume, run the exact same baseline command. To generate only the table:

```bash
python3 task_restore_v1/run_parallel.py --stage summary --tag sensor_baseline_parallel_v1_20261003
```

Results: `outputs/task_restore_v1/sensor_baseline_parallel_v1_20261003/RESULTS.md`, `summary.json`, `suite_protocol.json`, `dispatch.json`, and `throughput.json`. Each task/condition contains merged `episodes.jsonl`, `protocol.json`, `result.json`, and `shards/<offset>_<count>/` with isolated logs, records and sample MP4s.

Finish marker: `TASK_RESTORE_PARALLEL_FINISHED`. This replaces the earlier whole-batch six-GPU command; do not run both versions at once. The first evaluation establishes problem magnitude. The next method comparison requires matched-history image restoration, noise-augmentation fine-tuning, nearest restoration work and at least three independent adaptation seeds; none of those are established by this deployment smoke.

## Completed deployment checks (2026-10-03)

- Five unit tests passed, including geometry conversion, sensor noise, bank tampering and strict pairing rejection. Bank-to-restored state permits only measured float64 normalization, not physical-state mismatches.
- Two independent environment constructions per task (different construction RNG) restore identical model, actual physical state and RGB hashes for all three tasks: `TASK_RESTORE_REAL_STATE_TEST_FINISHED`.
- `deploy_paired_smoke_20261003`: one 400-step episode per task per clean/medium condition, 6/6 ever task successes; strict state/model/bank-entry pairing passed. Six-GPU episode times: 151.5–156.4 seconds. This is not a population success estimate or a restoration-method benefit.
- Busy GPU 4 rejection checked read-only; no unrelated processes were modified.
- Full 450-episode baseline is NOT started by the agent. The old 4–5 hour estimate is replaced by the parallel deployment below.

## Dynamic sharding update (2026-10-03)

Seven idle A800 80GB cards were found (0,1,2,3,5,6,7); GPU 4 was busy and is excluded. `--gpus auto` rechecks availability at launch. Up to four evaluator processes per idle GPU receive five-episode shards dynamically: 90 shards, 450 total episodes, up to 28 processes. Per-process Torch/OMP/MKL/OpenBLAS/Mesa threads are capped to prevent oversubscription. The verified core evaluator, camera corruption, policy checkpoints, 100-step DDPM and 400-step horizon remain unchanged.

Six parallel integrity tests passed. The 28-process capacity smoke ran 36 **80-step** episodes in 93.24 worker-pool seconds (26.02–28.10 seconds per episode), with strict paired model/state/bank-entry hashes and merge checks passed. A four-shard, two-episodes-per-shard check also passed (eight short episodes, 66.5 seconds). These short runs validate capacity and repeated resets only; their success rates are NOT full-horizon research results.

Estimated full run: roughly 1–1.5 hours on seven idle cards, extrapolating to 400 steps and allowing for shard dispatch, checkpoint loading, videos and CPU contention. Confirm using actual formal-run progress. Complete shards are checked and skipped on resume; partial shards continue from persisted unique episode IDs. Missing, duplicate, mismatched or unpaired results are rejected before the final table.
