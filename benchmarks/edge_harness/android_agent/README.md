# Android agent toolkit for Spark-X2.5

Lets a text-only model operate a phone: observe the accessibility tree, call
one tool, execute it on the device, observe again.

    ui_tree.py    uiautomator XML -> indexed element list -> back to a tap point
    device.py     AdbDevice (a real phone) | SimulatedDevice (scripted, in-process)
    executor.py   one tool call -> one device action, errors as sentences
    tool_parse.py recover a call from whatever shape the model emitted
    agent.py      the loop: budget, no-op detection, undo detection, fallbacks
    run_episode.py CLI

## Running it

Serve the model (Spark emits GLM-4.5 tool format, so the stock parsers apply):

    vllm serve <model> --port 8077 --enable-auto-tool-choice \
        --tool-call-parser glm45 --reasoning-parser glm45

Then:

    python run_episode.py --task "Turn on Wi-Fi." --device sim   # no phone needed
    python run_episode.py --task "Turn on Wi-Fi." --device adb   # a real phone

`pytest test_android_agent.py` runs 21 tests with no phone and no server.

## What the loop does that a plain ReAct loop does not

Each one is a response to a measured failure of *this* model, not generic
scaffolding:

- **Retry without thinking on an empty reply.** The model can reason past its
  token budget without closing `<think>`; vLLM's reasoning parser only emits
  content once the block closes, so the reply arrives empty with no tool call.
  Raising the cap buys a longer silence. A non-thinking retry costs ~0.6 s.
- **Report actions that changed nothing.** "the screen did not change" is
  information the model cannot otherwise have.
- **Report and stop on self-undo.** Tapping a switch that is already on undoes
  the task. Each tap really does change the screen, so a no-change check
  cannot see it; the loop watches for the *same* action returning the screen to
  its previous state. Deliberately narrow -- opening a submenu and pressing
  back revisits a screen and must not trip it.
- **Recover malformed tool calls.** See below.
- **A step budget**, so a confused episode ends instead of running forever.

The loop never decides a task is finished. It reports what happened and lets
the model call `task_complete`; a harness that completes tasks on the model's
behalf measures itself.

## Measured: which build can actually finish a task

Simulated phone, one run per cell, `max_steps` 10, Xeon 8480C 24 threads.

| build | decode | "Turn on Wi-Fi." | "Set an alarm for 7:30 AM." |
|---|---|---|---|
| bf16 | 41.0 tok/s | done in 4 steps | wrong app (Settings, not Clock), toggles Airplane mode, stuck |
| **W4A16 g128** | 80.0 tok/s | done in 4 steps | right app, full navigation (Clock -> Alarm -> Add -> Save), then no tool call |
| W4A8 g128 | 86.6 tok/s | turns Wi-Fi on, then toggles it off again (2 runs, same failure) | taps "Phone" twice, reaches Alarm, then loops |

Two things fall out, and both matter more than the throughput numbers:

**Navigation survives quantization; completion does not.** Every build walks
home -> Settings -> Network -> Wi-Fi correctly. Only some of them notice that
the switch now reads `checked=true` and stop. W4A8 -- the fastest build, and
the one the speed-first selector recommends -- turns Wi-Fi on and then taps it
off again, ending with the task undone.

**The fidelity number had a task-level meaning all along.** W4A8 reproduced
1/12 greedy prompts against its own bf16 and W4A16 2/12
(analysis/spark_edge_support.md 2c). That looked like a small difference on a
proxy metric. It is the difference between finishing and not.

So for agentic use the recommendation is **W4A16**, not W4A8: 8% slower, and
it is the only 4-bit build here that completed anything. `weight_path.py`'s
`priority="fidelity"` selects it.

## Honest limits

- **No phone was involved.** This host has no `adb`, no emulator and no
  `/dev/kvm`. The simulator emits real uiautomator XML, so the parser, the
  index-to-coordinate mapping and the stopping rules are exercised on the path
  hardware drives -- but whether a real Settings app is laid out the way these
  fixtures assume is untested.
- **Two tasks, one run each.** Enough to show a failure mode reproduces, not
  enough to rank builds by success rate.
- **Stock `glm45` does not always parse Spark.** On 4 of 5 screens it does. On
  the Wi-Fi toggle screen the model emitted
  `[tap]\n{"tool": "tap", "parameters": {"index": 1}` -- unterminated, and
  `tool_calls` came back empty. `tool_parse.py` recovers that and reports every
  recovery, because an agent that silently repairs its model's output hides
  regressions in that model. The earlier claim that the stock parsers work
  "as-is" is true most of the time, which is not the same thing.
