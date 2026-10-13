# E05: external restoration and unseen sensor-inspired corruptions

Pre-fixed before formal E05 outcomes. Code and computation reside on the server.

## Question and evidence boundary

Does decision-consistency restoration offer operational utility beyond a real external low-light restoration baseline? Do the retained frozen models generalize to two pre-fixed unseen corruption profiles?

This is not a significance chase. E04 did not establish a reproducible decision-vs-feature advantage. Retain all seeds3072/3073/3074 and both Can/Square, including unfavorable outcomes. No automatic paper-readiness claim. No best-checkpoint or best-corruption selection.

## Fixed methods

- Retain all18 E04 image/feature/decision adapters and their frozen published policies.
- External baseline: official RetinexFormer (ICCV2023) architecture, MIT license, commit `1e9a0efce4b306b6701b824768370ff26066c32a`; official LOL-v1 public checkpoint selected before robotic outcomes. SHA256 `9b4810dc51322fab70be8658ee0ad1df64a54aa814497b7849767baa2ff32cd1`.
- Distinct zero-shot and adapted baselines. Zero-shot is ONE shared checkpoint, not3 independent training seeds.
- Six adapted models:2tasks x3seeds. Same existing train24/valid4 demonstrations and window stride4,2000 updates,batch16,AdamW3e-4,weight decay1e-4,gradient clip1,20%identity/80%existing lowlight corruption, pixelL1 only. Last checkpoint, no closed-loop selection. Validation is diagnostic.
- Adaptation matches optimization steps/data, NOT initialization/pretraining/capacity or temporal architecture. Official external pretraining uses additional LOL data. Official network processes each frame independently; our causal adapter reads2frames. Do not attribute such a comparison solely to continuous supervision.
- Native84x84 RGB,2cameras, no resize, no `GT_mean`, no self-ensemble. Inference output clips[0,1]; training uses unclipped official network output.
- Frozen policy:2observations,100DDPM inference steps,8-action chunk,10D absolute action converted to7D robot control,400 physical control steps.

## Sensor-inspired profiles fixed in advance

Linear-light RGB approximation, not calibrated RAW measurements. Poisson shot noise, Gaussian read noise, clipping, sRGB and8-bit rounding. No FFT/global structural corruption.

|Profile|Exposure|Poisson gain|Read std|Persistent pixel bias std|
|---|---|---|---|---|
|clean|1|0|0|0|
|lowlight (training/reference)|0.25|0.004|0.008|0|
|exposure_shift (unseen)|0.125|0.004|0.008|0|
|sensor_fpn (unseen)|0.25|0.004|0.008|0.004|

The FPN map is independently drawn for each camera and physical test episode, then held constant across that episode. It is NOT calibrated permanent hardware-specific bias. Noise streams for the map and temporal shot/read are separate. One corruption per physical observation, history frames are retained without re-corruption; initial repeated observations share one corrupted sample. Shared random seeds ensure common randomness on identical observations, not identical trajectories after actions diverge.

## States and budget

Reuse the actual E04 state banks30000..30099 and fingerprint every simulator reset. E05 profiles are unseen; E05 physical states are NOT unseen. Existing E04 states were previously evaluated; no test-state filtering.

- External fine-tuned:6models x(clean50+lowlight100)=900new rollouts.
- External zero-shot:2tasks x(clean50+lowlight100)=300new rollouts.
- Each unseen profile:2tasks x50states x(9retained models+3external adapted models+original1+zero-shot1)=1400new rollouts.
- Total:4000new full400-step rollouts,72groups,800shards of5.
- Reuse3200E04 records without rerunning or recounting as new. Two compatibility rollouts are explicitly NON-formal and not method selection.

## Outcomes and statistics

Primary: `env.is_success()['task']` ever task success. Secondary: final success, first-success control step and inference time over the complete rollout. Report all task x trainingseed x method x condition cells and videos, not offline image quality in place of operation.

Pre-fixed exact paired two-sided McNemar tests:6decision-vs-external-adapted lowlight tests (2tasks x3seeds), Holm family. Unseen secondary family:36tests (2profiles x2tasks x3seeds x3controls=image/feature/external-adapted), Holm. Ever/final have separate primary/secondary families; do not promote final significance as the pre-fixed primary. Clean/shared controls descriptive. State-cluster bootstrap conditions on3fixed trained models and does not quantify uncertainty over new training seeds.

Record source/assets/data/policy/model/state-bank hashes. Reject short episodes, duplicate states, mixed protocols, partial/malformed results. Resume uses identical locked settings; fail rather than overwrite on revision mismatch. GPU-auto skips busy cards;1training worker/GPU and up to4evaluation workers/GPU. Interrupt stops only owned `n2d-external-*` containers; never exits/kills tmux.

## Server command

```bash
cd ~/n2d_dreamerv3
python3 task_restore_external_v1/run_external.py --stage full --gpus auto --workers-per-gpu 4 --tag external_sensor_v1_20261004
```

Preflight: replace `--stage full` with `--stage preflight`. Summary only after all results exist: `--stage summary`. Same full command safely resumes fixed-budget training/valid shards after interruption. Do not run two launchers concurrently.

## Decision after completion

If external restoration matches or exceeds the proposed method and task-supervision has no repeated operational benefit, narrow the claim or stop this idea; do not buy more arbitrary scans. If utility is repeated and clean performance remains acceptable, organize results and limitations for the paper; mechanism claims still need matched supervision controls (already included), not PSNR gains or selected seeds. This round is a fixed decision gate, not an indefinite new experiment chain.
