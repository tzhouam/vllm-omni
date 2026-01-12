### Sampling Params Validator（多阶段）

用于 Omni **多阶段**流水线在 orchestrator 侧（`Omni` / `AsyncOmni`）对 `sampling_params_list` 做检查。\n
### 如何启用

在对应的 stage config YAML 顶层增加：\n
```yaml
model_params_validator: "pkg.module:func"
```

其中 `func` 需要返回 `list[str]`（warning 文本），不要 raise。Orchestrator 会将其以 **warning_once** 输出，不会阻断请求。\n
### 函数签名（推荐）

```python
def validate_stage_params_list(
    sampling_params_list: list[dict],
    stage_list: list[Any] | None = None,
    engine_args_list: list[dict[str, Any]] | None = None,
    *,
    model: str | None = None,
    config_path: str | None = None,
) -> list[str]:
    ...
```
