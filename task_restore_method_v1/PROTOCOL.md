# Decision-preserving restoration: fixed-budget first method test

This is the first **proposed method** experiment, not another corruption scan.
The completed native baseline is preserved unchanged in task_restore_v1.

## Fixed pilot

- Can and Square, frozen official Diffusion Policy Transformer PH train_0 EMA.
- Official training split's first 24 complete demonstrations; first 4 official validation demonstrations. Numerical episode ordering, no success/cherry-picking selection. Render paired two-frame histories from simulator states every 4 control frames; full demonstration XML is restored.
- Pair clean/noisy RGB during adapter training. Sensor-inspired synthetic lowlight: linear-light exposure 0.25, Poisson gain 0.004, Gaussian read std 0.008, 8-bit sRGB quantization. 20% identity examples. Not calibrated RAW or real hardware evidence.
- Same causal two-frame, shared-camera residual U-Net, width 24, same initialization, batches, sensor random stream, AdamW lr 0.0003, batch 16, 2,000 updates. No future image. No test clean reference/proprio difference/object state input.
- Image arm: RGB L1 only. Decision arm: RGB L1 + 0.1 diffusion decision MSE.
- Cache clean frozen-policy action trajectories and observation conditions. Forward-diffuse the teacher trajectory, then compare teacher and student predictions using **the same action noise and diffusion timestep**, on the eight action slots actually executed. Frozen actor weights and deterministic eval-mode encoders; student gradient passes through the frozen actor to restoration pixels. This is a continuous decision-consistency surrogate, not proof of action/closed-loop improvement.
- Last fixed-budget checkpoint, no test-driven checkpoint selection. One adapter training seed 3072: a pilot, not independent training replication. If useful, repeat seeds 3073/3074 before main paper claims.
- Fresh evaluation seeds 20000..20049; no reuse of diagnostic seeds 10000..10049. All seven groups share simulator XML/state bank, sensor RNG and actor sampling seeds. Different paths after actions naturally produce different images; equal RNG does not mean forcing identical observations after divergence.
- Full 400-step trajectories, original 100-step DDPM and eight-action execution unchanged. Official task success and final success both recorded.
- Seven groups: clean original; lowlight original; lowlight **oracle exposure**; image-restoration clean/lowlight; decision-restoration clean/lowlight. Known exposure control is labeled oracle and cannot be claimed as blind restoration. Even this generous simple control is needed to expose a brightness-only result.
- Total 700 full episodes for two tasks. At most four isolated evaluators per idle GPU; skips occupied cards. Data preparation and four adapter jobs precede evaluation automatically.

## Commands (server, user-run long experiment)

```bash
cd ~/n2d_dreamerv3
python3 -u task_restore_method_v1/run_method.py --stage full --gpus auto \
  --tasks can,square --tag decision_pilot_v1_20261003
```

Completion: `METHOD_PILOT_FINISHED`.
Results: `outputs/task_restore_method_v1/decision_pilot_v1_20261003/RESULTS.md` and `summary.json`.
Training JSONL and final checkpoints; whole-demo train/validation manifest; code/asset/data/adapter hashes; all episode logs; clean/noisy/restored videos are retained.
Same command resumes matching checkpoints and completed shards; refuses different protocols under one tag. Ctrl-C stops only this launcher's containers; no tmux exit/kill and no interference with unrelated GPU jobs.

## Decision, not an endless search

This table must answer: does a learned restoration recover operation, and does decision supervision add benefit beyond image-only restoration? Report both tasks and clean retention, not one best cell. If exposure/image-only matches the proposal, don't credit decision supervision. If both learned arms fail after the fixed budget, first assess whether the image objective itself trained; don't randomly switch noise/task/seed until something looks positive. Further training, if justified by ordinary optimization evidence, must be a declared equal-budget continuation for both arms, not selectively helping one arm.
An encouraging pilot only authorizes the scientific case for three seeds, Lift, closest prior-work and noise-augmentation controls, held-out noise, and key ablations. It does not imply novelty or acceptance. No new long run is launched by Codex without the user's request.
