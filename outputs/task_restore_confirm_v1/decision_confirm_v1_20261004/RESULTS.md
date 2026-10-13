# New-state confirmation and supervision-form control

Real MuJoCo / robosuite closed-loop task success; all episodes last 400 control steps. No seed or state exclusion.
100 lowlight states (30000..30099), first 50 of the same bank for clean. Six new feature models; twelve prior image/decision models frozen.
Encoder-feature loss was scaled using training-only initial gradients, never evaluation success. This is a supervision ablation, not an official SR4IR/EDTR replication.

|Training seed|Task|Arm|Condition|Ever success|Final success|
|---|---|---|---|---|---|
|shared|can|original|clean|50/50|50/50|
|shared|can|original|lowlight|0/100|0/100|
|shared|can|exposure_oracle|lowlight|93/100|93/100|
|shared|square|original|clean|41/50|41/50|
|shared|square|original|lowlight|0/100|0/100|
|shared|square|exposure_oracle|lowlight|89/100|88/100|
|3072|can|image|clean|50/50|50/50|
|3072|can|image|lowlight|97/100|97/100|
|3072|can|feature|clean|50/50|50/50|
|3072|can|feature|lowlight|96/100|96/100|
|3072|can|decision|clean|50/50|50/50|
|3072|can|decision|lowlight|98/100|98/100|
|3072|square|image|clean|40/50|40/50|
|3072|square|image|lowlight|85/100|85/100|
|3072|square|feature|clean|43/50|43/50|
|3072|square|feature|lowlight|85/100|85/100|
|3072|square|decision|clean|41/50|41/50|
|3072|square|decision|lowlight|91/100|91/100|
|3073|can|image|clean|50/50|50/50|
|3073|can|image|lowlight|93/100|93/100|
|3073|can|feature|clean|50/50|50/50|
|3073|can|feature|lowlight|97/100|97/100|
|3073|can|decision|clean|50/50|50/50|
|3073|can|decision|lowlight|96/100|96/100|
|3073|square|image|clean|40/50|40/50|
|3073|square|image|lowlight|82/100|82/100|
|3073|square|feature|clean|39/50|39/50|
|3073|square|feature|lowlight|86/100|86/100|
|3073|square|decision|clean|42/50|42/50|
|3073|square|decision|lowlight|86/100|86/100|
|3074|can|image|clean|50/50|50/50|
|3074|can|image|lowlight|96/100|96/100|
|3074|can|feature|clean|50/50|50/50|
|3074|can|feature|lowlight|97/100|97/100|
|3074|can|decision|clean|50/50|50/50|
|3074|can|decision|lowlight|96/100|96/100|
|3074|square|image|clean|38/50|38/50|
|3074|square|image|lowlight|81/100|81/100|
|3074|square|feature|clean|39/50|39/50|
|3074|square|feature|lowlight|87/100|87/100|
|3074|square|decision|clean|39/50|39/50|
|3074|square|decision|lowlight|87/100|87/100|

## Lowlight paired comparisons

Primary ever-success: Holm correction across 12 per-seed lowlight tests. Final-success is a separate secondary family. Clean comparisons are descriptive.
|Seed|Task|Decision versus|Gains|Losses|Delta pp|Exact paired p|Holm p|
|---|---|---|---|---|---|---|---|
|3072|can|image|1|0|+1.0|1|1|
|3072|can|feature|2|0|+2.0|0.5|1|
|3072|square|image|10|4|+6.0|0.179565|1|
|3072|square|feature|11|5|+6.0|0.210114|1|
|3073|can|image|3|0|+3.0|0.25|1|
|3073|can|feature|0|1|-1.0|1|1|
|3073|square|image|12|8|+4.0|0.503445|1|
|3073|square|feature|8|8|+0.0|1|1|
|3074|can|image|1|1|+0.0|1|1|
|3074|can|feature|1|2|-1.0|1|1|
|3074|square|image|14|8|+6.0|0.286279|1|
|3074|square|feature|11|11|+0.0|1|1|

## Across-training-seed rates

|Task|Condition|Metric|Image mean +/- SD %|Feature mean +/- SD %|Decision mean +/- SD %|
|---|---|---|---|---|---|
|can|clean|success_ever|100.00 +/- 0.00|100.00 +/- 0.00|100.00 +/- 0.00|
|can|clean|success_final|100.00 +/- 0.00|100.00 +/- 0.00|100.00 +/- 0.00|
|can|lowlight|success_ever|95.33 +/- 2.08|96.67 +/- 0.58|96.67 +/- 1.15|
|can|lowlight|success_final|95.33 +/- 2.08|96.67 +/- 0.58|96.67 +/- 1.15|
|square|clean|success_ever|78.67 +/- 2.31|80.67 +/- 4.62|81.33 +/- 3.06|
|square|clean|success_final|78.67 +/- 2.31|80.67 +/- 4.62|81.33 +/- 3.06|
|square|lowlight|success_ever|82.67 +/- 2.08|86.00 +/- 1.00|88.00 +/- 2.65|
|square|lowlight|success_final|82.67 +/- 2.08|86.00 +/- 1.00|88.00 +/- 2.65|

The JSON also contains state-cluster percentile intervals, conditional on the three frozen training seeds. Shared controls are counted once; 300 repeated lowlight rollouts are NOT 300 independent training runs.
First-success time is conditional on success and must be interpreted with success rate, not alone. No assumption of significant superiority, real-camera calibration, precision-assembly quality, or hardware transfer.
