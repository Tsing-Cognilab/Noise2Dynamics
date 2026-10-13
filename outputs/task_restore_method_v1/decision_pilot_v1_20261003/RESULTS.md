# Matched restoration method pilot

One adapter training seed; fixed 2000-update budget. Not training-seed replication or proof of publication readiness.
Exposure oracle uses known synthetic exposure 0.25. It is a diagnostic upper-information control, not blind restoration.
Primary: real environment task success, full 400-step episodes. Fresh test states, paired across all arms.

|Task|Arm|Condition|Ever successes|Final successes|
|---|---|---|---|---|
|can|original|clean|50/50 (100.0%)|50/50 (100.0%)|
|can|original|lowlight|0/50 (0.0%)|0/50 (0.0%)|
|can|exposure_oracle|lowlight|46/50 (92.0%)|46/50 (92.0%)|
|can|image|clean|50/50 (100.0%)|50/50 (100.0%)|
|can|image|lowlight|49/50 (98.0%)|49/50 (98.0%)|
|can|decision|clean|50/50 (100.0%)|50/50 (100.0%)|
|can|decision|lowlight|49/50 (98.0%)|49/50 (98.0%)|
|square|original|clean|40/50 (80.0%)|40/50 (80.0%)|
|square|original|lowlight|0/50 (0.0%)|0/50 (0.0%)|
|square|exposure_oracle|lowlight|43/50 (86.0%)|43/50 (86.0%)|
|square|image|clean|40/50 (80.0%)|40/50 (80.0%)|
|square|image|lowlight|43/50 (86.0%)|43/50 (86.0%)|
|square|decision|clean|44/50 (88.0%)|43/50 (86.0%)|
|square|decision|lowlight|48/50 (96.0%)|48/50 (96.0%)|

## Paired decision vs image-only (lowlight)

- can: gained 0, lost 0, delta +0.0 pp; exact paired p=1.
- square: gained 5, lost 0, delta +10.0 pp; exact paired p=0.0625.

Do not choose best checkpoint/seed/test state. Offline image/decision losses are diagnostics, not closed-loop benefit.
If ordinary recovery or oracle exposure matches decision recovery, added decision supervision is not supported by this pilot.
