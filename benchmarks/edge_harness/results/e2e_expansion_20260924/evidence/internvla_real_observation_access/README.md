# InternVLA real-observation data access check

The pinned Place_Markpen checkpoint's `train_config.json` names Hugging Face dataset `a2d_real/a2d_pen_holder`. The [access record](access_report.json) pins the training-config hash and shows that `huggingface_hub` 1.30.0 received HTTP 404 for this dataset under the current authenticated account on 2026-09-24 UTC. A 404 does not distinguish an absent repository from a private repository inaccessible to this account.

No representative real Place_Markpen observation/reference-action pairs have been obtained from that route. Existing synthetic policy outputs and hosted Cosmos component probes therefore cannot establish real-observation action accuracy, physical action units/order/step time, or task quality. The next gate is access to the named dataset or another authorized representative observation/action set, followed by a fixed reference-action comparison on each complete policy route.
