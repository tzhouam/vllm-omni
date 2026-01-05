# 从 stage 初始化超时中剔除模型下载时间的设计

## 背景
- `stage_init_timeout` 目前主要用于：
  - 设备锁等待（`omni_stage._stage_worker`/`_stage_worker_async` 内的 `stage_init_timeout`）。
  - 上层文档描述为 “每个 stage 的初始化 watchdog”，但等待逻辑在 `_wait_for_stages_ready` 里并未真正扣除模型下载/权重加载的耗时。
- 模型权重下载与加载发生在：
  - LLM：`DefaultModelLoader._prepare_weights()` -> `download_weights_from_hf` / `maybe_download_from_modelscope` 等。
  - Diffusion：`DiffusersPipelineLoader._prepare_weights()` -> 同样的下载逻辑。
  - 这些耗时目前会计入整体初始化时间，易导致 stage “超时”或看起来卡住。

## 目标
- 不改变现有默认超时时间配置的前提下，将“模型权重下载 + 权重加载”阶段从 `stage_init_timeout` 的计时中剔除。
- 兼容 multiprocess 与 Ray 两种 backend。
- 不显著增加初始化路径上的复杂度或性能开销。

## 方案概述
引入“可暂停的 stage 初始化 watchdog”。在 stage worker 发送“下载阶段开始/结束”事件时，主进程的 watchdog 计时暂停/恢复，并按真实下载耗时顺延 deadline。

核心思路：
1. **事件上报**：在权重下载/加载前后发送队列消息：
   - `{"type": "stage_init_status", "stage_id": X, "phase": "download_start"}`
   - `{"type": "stage_init_status", "stage_id": X, "phase": "download_done", "duration_ms": ...}`
2. **可暂停计时**：主进程 `_wait_for_stages_ready` 为每个 stage 维护：
   - `deadline = start_ts + stage_init_timeout`
   - `pause_start` 与 `paused_dur` 累积时长
   - 收到 `download_start` -> 记录 `pause_start`
   - 收到 `download_done` -> `paused_dur += now - pause_start`，并把 `deadline += now - pause_start`
   - 轮询时使用调整后的 `deadline` 判定是否超时。
3. **边界**：下载事件缺失或异常时，回退到原始行为（不暂停）。

## 关键改动点
### 1) Loader 层提供下载阶段回调（可选）
- 在 `LoadConfig.model_loader_extra_config` 增加可选回调：
  - `download_hook: Callable[[str, str], ContextManager]` 或简单的 `(event: Literal["start","done"], info: dict) -> None`
- `DefaultModelLoader._prepare_weights` 与 `DiffusersPipelineLoader._prepare_weights` 在真正触发远端下载/筛选文件前后调用该回调：
  ```python
  hook = extra_config.get("download_hook")
  token = hook and hook("start", {"model": model_name_or_path})
  try:
      ... download_weights_from_hf(...)
  finally:
      hook and hook("done", {"model": model_name_or_path, "duration_s": ...})
  ```
- 默认不传则零开销。

### 2) Stage worker 注入 hook，向主进程上报
- 在 `_stage_worker` / `_stage_worker_async` 构造引擎前创建一个小工具：
  ```python
  def make_download_hook(stage_id, out_q):
      def _hook(event, info):
          if event == "start":
              out_q.put({"type": "stage_init_status", "stage_id": stage_id, "phase": "download_start"})
              info["_t0"] = time.time()
          elif event == "done":
              dur = time.time() - info.pop("_t0", time.time())
              out_q.put({"type": "stage_init_status", "stage_id": stage_id,
                         "phase": "download_done", "duration_ms": dur * 1000})
      return _hook
  ```
- 将该 hook 塞入 `LoadConfig.model_loader_extra_config`（LLM + Diffusion），不影响其他调用方。

### 3) 主进程 watchdog 支持暂停
- 在 `_wait_for_stages_ready` 中为每个 stage 维护状态：
  ```python
  state = {
    "deadline": start_ts + stage_init_timeout,
    "pause_start": None,
    "paused": 0.0,
  }
  ```
- 轮询队列时，如果收到 `stage_init_status`：
  - `download_start`: `pause_start = now`
  - `download_done`: `paused += now - pause_start`; `deadline += now - pause_start`; `pause_start = None`
- 超时判定使用 `now > deadline`（含顺延后的时间）。

### 4) 兼容性
- Ray backend：同样通过 `out_q` 传递事件，不依赖特定 IPC。
- 旧行为：未触发 hook 时，计时与当前一致；事件丢失时不会崩溃。
- 统计：可复用 `duration_ms` 计入日志/指标。

## 考虑的替代方案（放弃原因）
- **在 stage worker 内简单延长 `stage_init_timeout`**：无法区分“下载慢”与“真的卡住”，且需要预估下载时长。
- **预下载所有模型再启动 stage**：对多模型/多节点部署场景侵入大、增加总初始化路径。
- **用心跳保活代替暂停**：仍会触发超时，只能降低严格性而非剔除下载时间。

## 验证计划
1. 本地/CI：使用小模型 + 人为限速的 HuggingFace 源，验证超时不触发且 `duration_ms` 合理。
2. Ray 模式：多 stage 并行下载，确认事件能正确顺延各自 deadline。
3. 设备锁竞争场景：保持原有 `stage_init_timeout` 在锁等待处的限制，确保不会放宽设备争用窗口。
4. 回归：未配置 hook 的路径（纯 vLLM）应保持现状。

## 里程碑
- M1：实现 download hook 与事件上报，主进程支持暂停计时；单元/集成测试覆盖。
- M2：增加指标/日志（可选），展示下载时长。
- M3：文档与示例更新，指导用户在长下载场景调优 `stage_init_timeout` 与 `init_timeout`。
