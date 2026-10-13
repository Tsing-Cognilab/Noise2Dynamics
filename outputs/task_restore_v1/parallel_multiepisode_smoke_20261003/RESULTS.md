# Published policy sensor-noise baseline

This measures fixed policy robustness, not a proposed method improvement. Single underlying policy training run per task; episode intervals do not imply training-seed replication.

|Task|Condition|Successes|Ever SR|Final SR|95% Wilson interval|Drop from clean|
|---|---|---|---|---|---|---|
|lift|clean|4/4|100.0%|100.0%|[51.0, 100.0]%|0.0 pp|
|lift|shot_read_medium|4/4|100.0%|100.0%|[51.0, 100.0]%|0.0 pp|
