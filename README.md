# Noise2Dynamics — CVM research code

This repository contains the **CVM visual-restoration project**. The repository name is retained for continuity. It does **not** contain the retired N2D latent-dynamics or Dreamer research lines, their datasets, or their checkpoints.

The two authorized destinations are `Audience2Li/Noise2Dynamics` (personal, private backup) and `Tsing-Cognilab/Noise2Dynamics` (organization, public delivery). They receive the same CVM snapshot and binary assets. Release access follows the visibility of its repository. Upload progress and completion are recorded independently for each destination.

## Overview

We train a visual restorer for a frozen diffusion-based manipulation policy. The restorer receives causal image histories and improves the observations supplied to the policy. Its training objective combines image reconstruction with continuous decision-response supervision.

The clean-reference and restored-image branches use the **same noisy action trajectory and diffusion timestep**. Their denoising responses are matched on the eight action slots executed by the policy. Gradients pass through the frozen policy to the restorer; the policy itself is not updated. Clean reference images are used during training, not supplied to the deployed policy.

The main manipulation tasks are **Can placement** and **Square-nut assembly** in MuJoCo/robosuite. Lift assets and early sensor-baseline records are retained as CVM preparation evidence, not as additional core-paper tasks. The benchmark records task success from the simulator, rather than substituting image-quality scores for successful manipulation.

## Repository layout

```text
task_restore_v1/             Simulator contract, sensor corruption, state banks,
                            frozen-policy assets, Dockerfile and Diffusion Policy code
task_restore_method_v1/      Causal restorer, training data preparation, Image/Decision study
task_restore_repeat_v1/      Matched-budget training replication, seeds 3073 and 3074
task_restore_confirm_v1/     Feature control and new-state confirmation, all three seeds
task_restore_external_v1/    Retinexformer controls and sensor-profile evaluation
task_restore_video_hd_v1/    High-resolution simulator views and policy-input export
outputs/                    CVM-only episode records, protocols, summaries and videos
manifests/                  File hashes, binary-asset inventory and exclusion list
docs/                       Repository scope, environment provenance and asset instructions
tools/                      Asset packing and verified restoration helpers
README.md
```

The original folder names are retained so imports and provenance records remain traceable. Source files and completed experimental records are copied without rewriting their contents. All three training seeds **3072, 3073 and 3074** are retained. No model or episode is removed because of its performance.

## Data and model assets

Regular Git history contains source code, protocols, configuration files, result records and compact media. Model weights, HDF5 demonstrations, NPZ training caches and compressed source/media bundles are staged separately as Release assets. **The assets are not yet on GitHub merely because this directory has been prepared.** Upload completion is tracked independently for both destinations in the deployment instructions.

The binary inventory is in `manifests/binary_assets.jsonl`. It identifies each file by its repository-relative destination, size and SHA256. The prepared packages are:

|Package|Contents|
|---|---|
|`cvm-datasets.tar.zst`|Demonstration HDF5 files, paired training/validation caches and simulator-state arrays|
|`cvm-models.tar.zst`|Frozen Diffusion Policy checkpoints, learned CVM adapters and Retinexformer weights|
|`cvm-resources.tar.zst`|Compressed upstream source bundles and CVM media-resource archives|

Packages larger than 128 MiB are split into numbered files for upload. This limits the amount that must be retried if a connection fails. Checksums accompany every uploaded file and the unsplit package. The restoration helper reassembles verified parts, checks package hashes, rejects unsafe paths and refuses to overwrite differing destination files. Existing identical assets are verified and retained.

Assets include the actual paired dataset used by the method experiments: the first **24 training demonstrations** and first **4 validation demonstrations** from the official split, with two-frame windows sampled every four control frames. The data manifest and recorded hashes define the split. They are not regenerated from test outcomes.

Upstream policy and restoration assets retain their source URLs and checksums in `task_restore_v1/assets/manifest.json` and `task_restore_external_v1/assets/manifest.json`. Original third-party license files are retained. This repository does not assign a new license to third-party datasets or weights; users remain responsible for observing the original providers' terms. A source-code MIT license should not be read as a blanket commercial license for every dataset or model.

## Experimental configuration

|Item|Configuration|
|---|---|
|Core tasks|Can placement and Square-nut assembly|
|Observations|Two 84 × 84 RGB views, two-frame history and unchanged proprioception|
|Policy|Frozen, publicly released Diffusion Policy Transformer, PH `train_0`, EMA weights|
|Policy inference|100 DDPM steps; execute eight actions per prediction|
|Simulator horizon|400 control steps per full episode|
|Restorer|Causal two-frame, shared-camera residual U-Net, width 24; 267,195 parameters|
|Training seeds|3072, 3073, 3074|
|Training budget|2,000 updates, batch size 16, AdamW learning rate 3 × 10⁻⁴|
|Decision weight|0.1, added to RGB L1 reconstruction|
|Training corruption|Exposure 0.25; Poisson gain 0.004; Gaussian read-noise std 0.008|
|Identity examples|20% of training samples|
|Model selection|Final fixed-budget checkpoint|

Noise is a sensor-inspired approximation applied in linearized RGB, followed by clipping, sRGB transfer and quantization. It is not a calibrated RAW-camera model. Each physical frame is corrupted once and retained in the observation history.

Methods share simulator XML/state banks, reset identifiers and defined random-number streams. Original-policy controls are reused without being counted as independently trained models. Ever-success and final-step success are recorded separately. Complete result files, pairing checks and per-seed records are retained under `outputs/`; a selected video is not a replacement for the complete evaluation.

Historical CVM evaluation conditions remain in the private records even when they are not included in the paper. The paper excludes Square-nut assembly under the additional exposure-halving and fixed-pattern-noise conditions. This publication choice does not delete the underlying records.

## Runtime environment

The recorded CVM runtime is `noise2dynamics:taskrestore-v1`, with:

- PyTorch 2.5.1+cu124;
- robosuite 1.2.0, pinned to `277ab9588ad7a4f4b55cf75508b44aa67ec171f0`;
- robomimic 0.2.0;
- MuJoCo 2.1 / `free-mujoco-py==2.1.6`;
- CPU OSMesa rendering and GPU policy inference;
- container Python entry point `/opt/taskrestore/bin/python`.

The preserved `task_restore_v1/Dockerfile` uses the pre-existing `noise2dynamics:franka-v1` base image. **It is not a standalone Docker recipe that can be built from this repository alone.** The private full-server archive preserves the existing images separately; those old-line images and the 154 GB combined archive are not uploaded into this CVM-only Git repository. `docs/ENVIRONMENT.json` records the CVM image identity and dependencies.

Windows can store and download the repository, but the original simulation runtime requires Linux or a compatible WSL2/Linux-container environment, NVIDIA drivers and Docker GPU support. Do not replace it with the newer Dreamer environment.

## Read existing results without launching training

On the original server, use the original project root:

```bash
cd /data/lizhenhao/lizhenhao/n2d_dreamerv3
python3 task_restore_confirm_v1/run_confirm.py \
  --stage summary --tag decision_confirm_v1_20261004
python3 task_restore_external_v1/run_external.py \
  --stage summary --tag external_sensor_v1_20261004
```

The per-stage `PROTOCOL.md` files document the corresponding training and evaluation commands. `--stage full` actually launches experiments; it is not an installation or read-only check. Existing tags enforce locked budgets, sources, assets and simulator state banks. Use new output tags for new experiments rather than overwriting the paper records.

## Restoring on another machine

1. Clone either destination repository and obtain its corresponding Release packages. The personal backup requires authentication.
2. Restore the binary assets with the provided helper. It verifies uploaded-part and package checksums before safe extraction.
3. Restore the compatible CVM runtime separately.
4. Inspect the original protocol files before launching anything.

Some result locks and video-job specifications preserve the original absolute project paths. For exact replay of the existing locked experiment, a **fresh** Linux environment should reproduce `/data/lizhenhao/lizhenhao/n2d_dreamerv3`; do not install over an existing project. Arbitrary relocation requires an explicit path-migration plan and should not silently rewrite the original evidence.

## Scope and provenance

Included: the six CVM code directories and their six matching output directories, public-policy sources needed by CVM, CVM demonstration/cache data, trained adapters, initial-state banks, simulator result records and available server videos.

Excluded: all old N2D/latent/Dreamer code and outputs outside these directories, their models/datasets, Linux virtual environments, unrelated project assets, account credentials, Git history of the mixed original project, runtime locks, generated Python caches, incomplete downloads and duplicated early planning drafts. Selected roots are listed in `manifests/PREPARATION.json`; skipped entries within those roots are recorded in `manifests/excluded.jsonl`.

The manuscript, editable figures and final edited submission video under the local `E:\CASIA\12 CVM\CVM2027` directory are maintained separately and are not silently added by the server-only preparation tool.

The personal `Audience2Li/Noise2Dynamics` repository must remain **private**. The organization `Tsing-Cognilab/Noise2Dynamics` is **public** at the owner's explicit request. The uploader verifies both settings and never changes visibility automatically. Public delivery exposes the project identity and records; consider anonymous-review obligations before sharing it with reviewers.
