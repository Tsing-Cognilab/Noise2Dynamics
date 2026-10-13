# Published policy sensor-noise baseline

This measures fixed policy robustness, not a proposed method improvement. Single underlying policy training run per task; episode intervals do not imply training-seed replication.

|Task|Condition|Successes|Ever SR|Final SR|95% Wilson interval|Drop from clean|
|---|---|---|---|---|---|---|
|can|clean|50/50|100.0%|100.0%|[92.9, 100.0]%|0.0 pp|
|can|lowlight|0/50|0.0%|0.0%|[0.0, 7.1]%|100.0 pp|
|can|shot_read_medium|49/50|98.0%|98.0%|[89.5, 99.6]%|2.0 pp|
|lift|clean|50/50|100.0%|56.0%|[92.9, 100.0]%|0.0 pp|
|lift|lowlight|37/50|74.0%|40.0%|[60.4, 84.1]%|26.0 pp|
|lift|shot_read_medium|50/50|100.0%|66.0%|[92.9, 100.0]%|0.0 pp|
|square|clean|46/50|92.0%|92.0%|[81.2, 96.8]%|0.0 pp|
|square|lowlight|0/50|0.0%|0.0%|[0.0, 7.1]%|92.0 pp|
|square|shot_read_medium|44/50|88.0%|88.0%|[76.2, 94.4]%|4.0 pp|
