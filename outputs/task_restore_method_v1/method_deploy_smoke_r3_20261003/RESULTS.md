# Matched restoration method pilot

One adapter training seed; fixed 2000-update budget by default. Not training-seed replication or proof of publication readiness.
Exposure oracle uses known synthetic exposure 0.25. It is a diagnostic upper-information control, not blind restoration.
Primary: real environment task success, full 400-step episodes. Fresh test states, paired across all arms.

|Task|Arm|Condition|Ever successes|Final successes|
|---|---|---|---|---|
|can|original|clean|0/1 (0.0%)|0/1 (0.0%)|
|can|original|lowlight|0/1 (0.0%)|0/1 (0.0%)|
|can|exposure_oracle|lowlight|0/1 (0.0%)|0/1 (0.0%)|
|can|image|clean|0/1 (0.0%)|0/1 (0.0%)|
|can|image|lowlight|0/1 (0.0%)|0/1 (0.0%)|
|can|decision|clean|0/1 (0.0%)|0/1 (0.0%)|
|can|decision|lowlight|0/1 (0.0%)|0/1 (0.0%)|
|square|original|clean|0/1 (0.0%)|0/1 (0.0%)|
|square|original|lowlight|0/1 (0.0%)|0/1 (0.0%)|
|square|exposure_oracle|lowlight|0/1 (0.0%)|0/1 (0.0%)|
|square|image|clean|0/1 (0.0%)|0/1 (0.0%)|
|square|image|lowlight|0/1 (0.0%)|0/1 (0.0%)|
|square|decision|clean|0/1 (0.0%)|0/1 (0.0%)|
|square|decision|lowlight|0/1 (0.0%)|0/1 (0.0%)|

## Paired decision vs image-only (lowlight)

- can: gained 0, lost 0, delta +0.0 pp; exact paired p=1.
- square: gained 0, lost 0, delta +0.0 pp; exact paired p=1.

Do not choose best checkpoint/seed/test state. Offline image/decision losses are diagnostics, not closed-loop benefit.
If ordinary recovery or oracle exposure matches decision recovery, added decision supervision is not supported by this pilot.
