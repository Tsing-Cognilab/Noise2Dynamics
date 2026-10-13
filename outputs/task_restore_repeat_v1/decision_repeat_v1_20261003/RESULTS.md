# Fixed-budget training-seed replication

Only restoration-adapter training seed changes. The pretrained policy, paired data, noise, loss weights, optimizer, 2000 updates, final-checkpoint rule, and 50 full 400-step evaluation states stay fixed.
Original-policy and exposure-oracle controls are shared from seed3072: they have no adapter-training seed. They are not rerun or counted as new independent evidence.
Evaluation states are reused to test training repeatability, not a new held-out generalization test. Report all seeds, not the best seed.

|Training seed|Task|Arm|Condition|Ever success|Final success|
|---|---|---|---|---|---|
|shared|can|original|clean|50/50|50/50|
|shared|can|original|lowlight|0/50|0/50|
|shared|can|exposure_oracle|lowlight|46/50|46/50|
|3072|can|image|clean|50/50|50/50|
|3072|can|image|lowlight|49/50|49/50|
|3072|can|decision|clean|50/50|50/50|
|3072|can|decision|lowlight|49/50|49/50|
|3073|can|image|clean|50/50|50/50|
|3073|can|image|lowlight|48/50|48/50|
|3073|can|decision|clean|50/50|50/50|
|3073|can|decision|lowlight|48/50|48/50|
|3074|can|image|clean|50/50|50/50|
|3074|can|image|lowlight|49/50|49/50|
|3074|can|decision|clean|50/50|50/50|
|3074|can|decision|lowlight|49/50|49/50|
|shared|square|original|clean|40/50|40/50|
|shared|square|original|lowlight|0/50|0/50|
|shared|square|exposure_oracle|lowlight|43/50|43/50|
|3072|square|image|clean|40/50|40/50|
|3072|square|image|lowlight|43/50|43/50|
|3072|square|decision|clean|44/50|43/50|
|3072|square|decision|lowlight|48/50|48/50|
|3073|square|image|clean|43/50|43/50|
|3073|square|image|lowlight|41/50|41/50|
|3073|square|decision|clean|40/50|40/50|
|3073|square|decision|lowlight|44/50|44/50|
|3074|square|image|clean|39/50|39/50|
|3074|square|image|lowlight|43/50|43/50|
|3074|square|decision|clean|42/50|42/50|
|3074|square|decision|lowlight|47/50|47/50|

## Decision vs image-only: paired lowlight differences

|Training seed|Task|Gains|Losses|Ever delta (pp)|Exact paired p|Final delta (pp)|
|---|---|---|---|---|---|---|
|3072|can|0|0|+0.0|1|+0.0|
|3073|can|1|1|+0.0|1|+0.0|
|3074|can|0|0|+0.0|1|+0.0|
|3072|square|5|0|+10.0|0.0625|+10.0|
|3073|square|6|3|+6.0|0.5078|+6.0|
|3074|square|5|1|+8.0|0.2188|+8.0|

## Across-training-seed summary

Mean and sample SD use the three adapter training seeds, not 150 pooled independent episodes. Three seeds do not establish general superiority across policies, tasks, or real cameras.

|Task|Condition|Metric|Image mean +/- SD (%)|Decision mean +/- SD (%)|Paired delta mean +/- SD (pp)|
|---|---|---|---|---|---|
|can|clean|success_rate|100.00 +/- 0.00|100.00 +/- 0.00|+0.00 +/- 0.00|
|can|clean|final_success_rate|100.00 +/- 0.00|100.00 +/- 0.00|+0.00 +/- 0.00|
|can|lowlight|success_rate|97.33 +/- 1.15|97.33 +/- 1.15|+0.00 +/- 0.00|
|can|lowlight|final_success_rate|97.33 +/- 1.15|97.33 +/- 1.15|+0.00 +/- 0.00|
|square|clean|success_rate|81.33 +/- 4.16|84.00 +/- 4.00|+2.67 +/- 7.57|
|square|clean|final_success_rate|81.33 +/- 4.16|83.33 +/- 3.06|+2.00 +/- 6.93|
|square|lowlight|success_rate|84.67 +/- 2.31|92.67 +/- 4.16|+8.00 +/- 2.00|
|square|lowlight|final_success_rate|84.67 +/- 2.31|92.67 +/- 4.16|+8.00 +/- 2.00|
