# InternVLA public observation compatibility audit

**Disposition (2026-09-25): no new E2E or task-quality qualification.** The public [griffinlabs InternData-A1 LeRobot v3 derivative](https://huggingface.co/datasets/griffinlabs/InternData-A1-LeRobot-v3.0-by-embodiment) is downloadable, but its `genie1/pick_and_place_tasks/pick_and_place_part1/omniobject3d-pen` recording is not a valid reference set for the pinned Place_Markpen checkpoint. The derivative is CC BY-NC-SA 4.0. Its data remains in the local Hugging Face cache and is not copied into this repository.

The repository revision is `c621c2e12e1f4aaa634f3d89e3573ca969087235`. The local `meta/info.json` (SHA-256 `06dbe1acab5933a229e98472e1b7477235c362a4056fae65589a0cf4532ade8c`) reports robot type `Genie-1`, 25 episodes, 8,489 frames and 30 fps. `meta/tasks.parquet` (SHA-256 `9f436d671d18d6ac5f7af3a479647f6e9f57d4d033b733251c31d4f2bf9181fa`) contains the instruction **“Pick up the pen with the left arm and place it in the basket.”** The first data chunk (SHA-256 `d0a11704507b1812adb52c5d79c5de34a1d1044a245b1dfe4cc5f215f3d165fa`) contains `states.left_joint.position`, `states.right_joint.position` and left/right gripper positions. Three camera streams are available. The downloaded head, left-hand and right-hand MP4 SHA-256 values are `c034d9d9e5da8f4c7e838982970b3ac924a5c223856417841d21199623692302`, `bf6284e5eba4527882194e63e50f49c7ff499e8c9d40976d662acf6efad531c9` and `8b5100f12752221ca75b0959a18685c059396388d035c3b35fdf5d2e859c1b3d`.

The local checkpoint's `train_config.json` (SHA-256 `e50ec42e40ba1925983e0e8db36558d3738bea64e1698045ff7cc08a130e1053`) names `a2d_real/a2d_pen_holder`, and `stats.json` (SHA-256 `354710650bb91fc0e84a461987b3554b30bef9d5de47567480624484cdfbf3a2`) supplies `a2d` normalization for `observation.states.joint.position` (14 values) plus `observation.states.effector.position` (2 values). The maintained [vLLM-Omni A2D open-loop loader](../../../../../../examples/offline_inference/internvla_a1/internvla_a1_common.py) concatenates exactly those fields, uses three A2D camera keys, a `[-15, 0]` frame history, and checkpoint mean/std normalization. The [upstream InternVLA schema](https://github.com/InternRobotics/InternVLA-A-series/blob/e6fc904f9edbfb14532e97095fc2372202517f76/src/lerobot/dataset_schemas/presets.py) also distinguishes the older `a2d` keys from new-format `Genie-1` left/right joint and gripper keys. These are different state semantics despite both totaling 16 physical values. Repacking Genie-1 grippers into the A2D effector fields would create a misleading input.

The [new fixture preparer](../../../../experiments/prepare_internvla_a2d_fixture.py) refuses this derivative before video decoding or model execution:

```text
ValueError: dataset is not the checkpoint's A2D state schema: missing ['observation.states.effector.position', 'observation.states.joint.position']
```

Reproduce that refusal after downloading the listed `meta/info.json`, `meta/tasks.parquet` and first Parquet chunk at the pinned derivative revision. The three MP4 files are not needed for the early schema check:

```bash
python benchmarks/edge_harness/experiments/prepare_internvla_a2d_fixture.py \
  --model-dir /path/to/InternVLA-A1-3B-FT-Place_Markpen \
  --dataset-dir /path/to/omniobject3d-pen \
  --source griffinlabs/InternData-A1-LeRobot-v3.0-by-embodiment@c621c2e12e1f4aaa634f3d89e3573ca969087235 \
  --output-fixture /tmp/internvla-observation.npz \
  --output-manifest /tmp/internvla-observation.json
```

The [Omni policy profiler](../../../../experiments/probe_omni_internvla_policy.py) now accepts `--input-fixture` plus `--fixture-manifest` on every existing PC placement. It checks the A2D state/camera/preprocessing declaration, exact tensor shapes and dtypes, finite values, task, fixture SHA-256 and checkpoint model/stats hashes before worker launch; output reports retain the fixture manifest and hashes. Its existing synthetic fixture remains the default. This is a harness capability, not a new model pass.

Two one-request WSL CPU regression runs used the same **synthetic** patterned observation, zero state and noise. The [default path](synthetic_regression.json) and [external-fixture path](synthetic_external_fixture.json) both passed, returned identical action SHA-256 `007f3087423df63b578193ec3f95e353613219914f145b85ca54c403e9088a25`, and ended with zero reserved/quarantined host RAM. The default action hash also matches the earlier 20-request WSL CPU profile. The fixture path's single complete-request wall time was 3.246 s after 40.549 s startup; a single sample is not a latency distribution. Both worker logs are retained. The first fixture attempt [failed before worker launch](synthetic_external_fixture_manifest_rejection.json) because the driver incorrectly added fixture hashes to the backend's exact artifact manifest. Keeping fixture hashes only in the report repaired that contract; the failed run remains recorded.

For an authorized matching A2D dataset, run the same preparer with that dataset path and its immutable source revision, then append `--input-fixture /path/to/observation.npz --fixture-manifest /path/to/observation.json` to the existing PC placement profile command. The preparer reuses the offline example's frame alignment and checkpoint state normalization; it records the sample, task, source and content hashes. Compare candidate placements against a CPU run on the same fixture and noise, and evaluate task quality separately against reference physical actions.

The matching [official Place_Markpen archive](https://huggingface.co/datasets/InternRobotics/InternData-A1/tree/main/real_lerobotv30/genie1) remains gated under the current account as recorded in the [earlier access audit](../../../e2e_expansion_20260924/evidence/internvla_real_observation_access/README.md). The next valid experiment is to obtain an authorized matching A2D observation/action set, prepare a pinned fixture, run paired CPU and candidate accelerator placements on the **same** observation/noise, and compare physical reference actions with declared units, order, step time and task tolerance. Until then, the existing 20-request synthetic whole-policy profiles remain synthetic.
