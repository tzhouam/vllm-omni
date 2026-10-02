# InternVLA Place_Markpen task-reference access, 2026-10-02

The local `InternVLA-A1-3B-FT-Place_Markpen` checkpoint's `train_config.json`
names `a2d_real/a2d_pen_holder` and delta actions. The vLLM-Omni
[official Place_Markpen example](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/offline_inference/internvla_a1/)
points to the `real_lerobotv30/genie1/Genie1-Place_Markpen.tar.gz` archive in
[InternData-A1](https://huggingface.co/datasets/InternRobotics/InternData-A1/tree/main/real_lerobotv30/genie1).
The current authenticated account cannot download that single archive:
the [raw dry-run record](access_probe.json) returned exit code 1 with
`Access denied. This repository requires approval.` No matching content was
downloaded. The local `a2d-task374` data lacks the checkpoint's required
joint/effector state fields and is not a substitute.

After the account is approved under the dataset's access terms, download
the named archive at a pinned revision, record its SHA-256, verify that its
LeRobot observations include head/left-hand/right-hand images, 14 joint
and two effector state values, and paired 16-dimensional actions. Then use
the existing `prepare_internvla_a2d_fixture.py` and offline evaluator to
check two history frames, the 50-step action chunk, delta decoding,
physical units/order/timing and task quality. Until that evidence exists,
synthetic policy runs remain policy-only and cannot qualify InternVLA cells.

Reproduce the access check:

```bash
hf download InternRobotics/InternData-A1 \
  real_lerobotv30/genie1/Genie1-Place_Markpen.tar.gz \
  --repo-type dataset --dry-run
```
