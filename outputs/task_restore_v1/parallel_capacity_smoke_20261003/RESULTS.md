# Published policy sensor-noise baseline

This measures fixed policy robustness, not a proposed method improvement. Single underlying policy training run per task; episode intervals do not imply training-seed replication.

|Task|Condition|Successes|Ever SR|Final SR|95% Wilson interval|Drop from clean|
|---|---|---|---|---|---|---|
|can|clean|0/4|0.0%|0.0%|[0.0, 49.0]%|0.0 pp|
|can|lowlight|0/4|0.0%|0.0%|[0.0, 49.0]%|0.0 pp|
|can|shot_read_medium|0/4|0.0%|0.0%|[0.0, 49.0]%|0.0 pp|
|lift|clean|4/4|100.0%|100.0%|[51.0, 100.0]%|0.0 pp|
|lift|lowlight|2/4|50.0%|50.0%|[15.0, 85.0]%|50.0 pp|
|lift|shot_read_medium|4/4|100.0%|100.0%|[51.0, 100.0]%|0.0 pp|
|square|clean|0/4|0.0%|0.0%|[0.0, 49.0]%|0.0 pp|
|square|lowlight|0/4|0.0%|0.0%|[0.0, 49.0]%|0.0 pp|
|square|shot_read_medium|0/4|0.0%|0.0%|[0.0, 49.0]%|0.0 pp|
