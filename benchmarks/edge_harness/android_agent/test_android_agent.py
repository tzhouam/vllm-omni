"""Tests for the Android agent toolkit.

No phone, emulator or model server is needed: the device is the in-process
simulator and the model is a scripted stand-in. That is deliberate -- this
host has no `adb`, no emulator and no `/dev/kvm`, so anything that could only
be checked against real hardware would otherwise ship unverified.
"""

import json

import pytest

import agent as agent_mod
from agent import AgentConfig, first_tool_call, run_episode
from device import demo_phone
from executor import execute
from ui_tree import parse_ui_dump, render_tree, screen_signature

# --------------------------------------------------------------------- ui_tree

REAL_DUMP = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
 <node index="0" text="" resource-id="" class="android.widget.FrameLayout" package="com.android.settings" content-desc="" checkable="false" checked="false" clickable="false" enabled="true" focusable="false" focused="false" scrollable="false" long-clickable="false" password="false" selected="false" bounds="[0,0][1080,2400]">
  <node index="1" text="Network &amp; internet" resource-id="android:id/title" class="android.widget.TextView" package="com.android.settings" content-desc="" checkable="false" checked="false" clickable="true" enabled="true" focusable="true" focused="false" scrollable="false" long-clickable="false" password="false" selected="false" bounds="[0,300][1080,460]" />
  <node index="2" text="" resource-id="" class="android.widget.LinearLayout" package="com.android.settings" content-desc="" checkable="false" checked="false" clickable="false" enabled="true" focusable="false" focused="false" scrollable="false" long-clickable="false" password="false" selected="false" bounds="[0,460][1080,620]" />
  <node index="3" text="" resource-id="com.android.settings:id/switch_widget" class="android.widget.Switch" package="com.android.settings" content-desc="Wi-Fi" checkable="true" checked="true" clickable="true" enabled="true" focusable="true" focused="false" scrollable="false" long-clickable="false" password="false" selected="false" bounds="[800,460][1000,620]" />
 </node>
</hierarchy>"""


def test_parses_a_real_dump_and_keeps_only_what_is_actionable():
    els = parse_ui_dump(REAL_DUMP)
    labels = [e.text for e in els]
    assert "Network & internet" in labels          # entity decoded
    assert "Wi-Fi" in labels                        # fell back to content-desc
    assert "" not in labels                         # bare layout node dropped
    assert [e.index for e in els] == list(range(len(els)))


def test_switch_state_is_visible_to_the_model():
    """The one clearly wrong action measured earlier was tapping a switch that
    was already on; that is only avoidable if the state is on the screen."""
    rendered = render_tree(parse_ui_dump(REAL_DUMP))
    assert "checked=true" in rendered
    assert 'Switch "Wi-Fi"' in rendered


def test_index_maps_to_the_centre_of_the_element():
    els = parse_ui_dump(REAL_DUMP)
    wifi = next(e for e in els if e.text == "Wi-Fi")
    assert wifi.center == (900, 540)


def test_signature_tracks_state_not_pixels():
    a = parse_ui_dump(REAL_DUMP)
    shifted = REAL_DUMP.replace("[0,300][1080,460]", "[0,310][1080,470]")
    assert screen_signature(parse_ui_dump(shifted)) == screen_signature(a)
    toggled = REAL_DUMP.replace('checkable="true" checked="true"',
                                'checkable="true" checked="false"')
    assert screen_signature(parse_ui_dump(toggled)) != screen_signature(a)


def test_element_cap_bounds_the_context():
    big = '<?xml version="1.0"?><hierarchy>' + "".join(
        f'<node text="row{i}" class="android.widget.TextView" clickable="true" '
        f'bounds="[0,{i},100,{i + 10}]" />' for i in range(200)
    ) + "</hierarchy>"
    assert len(parse_ui_dump(big, max_elements=25)) == 25


# -------------------------------------------------------------------- executor

def test_tap_lands_on_the_right_row_and_reports_prior_state():
    device = demo_phone()
    device.current = "network"
    els = parse_ui_dump(device.dump())
    wifi = next(e for e in els if e.text == "Wi-Fi")
    result = execute("tap", {"index": wifi.index}, els, device)
    assert result.ok and device.state["wifi"] is True
    assert "was off" in result.message


def test_out_of_range_index_is_a_message_not_an_exception():
    device = demo_phone()
    els = parse_ui_dump(device.dump())
    result = execute("tap", {"index": 99}, els, device)
    assert not result.ok
    assert "out of range" in result.message and f"0-{len(els) - 1}" in result.message


def test_bad_arguments_explain_themselves():
    device = demo_phone()
    els = parse_ui_dump(device.dump())
    assert "missing required argument" in execute("tap", {}, els, device).message
    assert "whole number" in execute("tap", {"index": "banana"}, els, device).message
    assert "unknown tool" in execute("frobnicate", {}, els, device).message
    assert "direction must be" in execute("swipe", {"direction": "sideways"}, els, device).message


def test_type_text_focuses_the_field_first():
    device = demo_phone()
    device.current = "new_alarm"
    els = parse_ui_dump(device.dump())
    field = next(e for e in els if e.text == "Hour")
    execute("type_text", {"index": field.index, "text": "07"}, els, device)
    kinds = [a[0] for a in device.actions]
    assert kinds == ["tap", "text"], kinds
    assert device.typed == ["07"]


def test_swipe_direction_is_content_movement():
    device = demo_phone()
    els = parse_ui_dump(device.dump())
    execute("swipe", {"direction": "down"}, els, device)
    _, x1, y1, x2, y2 = device.actions[-1]
    assert int(y1) > int(y2), "scrolling down must move the finger up"


def test_task_complete_carries_its_summary():
    device = demo_phone()
    result = execute("task_complete", {"summary": "Wi-Fi is on"}, [], device)
    assert result.done and result.summary == "Wi-Fi is on"


# ----------------------------------------------------------------------- agent

def _scripted(calls):
    """A stand-in model that plays a fixed list of tool calls."""
    seq = list(calls)

    def fake_chat(cfg, messages, *, think):
        call = seq.pop(0) if seq else None
        if call is None:
            return {"choices": [{"message": {"content": "", "tool_calls": []}}]}, 0.01
        name, args = call
        return {"choices": [{"message": {"tool_calls": [
            {"function": {"name": name, "arguments": json.dumps(args)}}
        ]}}]}, 0.01

    return fake_chat


def test_first_tool_call_handles_both_argument_encodings():
    as_str = {"choices": [{"message": {"tool_calls": [
        {"function": {"name": "tap", "arguments": '{"index": 2}'}}]}}]}
    as_dict = {"choices": [{"message": {"tool_calls": [
        {"function": {"name": "tap", "arguments": {"index": 2}}}]}}]}
    assert first_tool_call(as_str) == ("tap", {"index": 2})
    assert first_tool_call(as_dict) == ("tap", {"index": 2})
    assert first_tool_call({"choices": [{"message": {"content": "hi"}}]}) == (None, {})
    # Malformed JSON must not kill the episode.
    broken = {"choices": [{"message": {"tool_calls": [
        {"function": {"name": "tap", "arguments": "{not json"}}]}}]}
    assert first_tool_call(broken) == ("tap", {})


def test_completes_a_multi_screen_task(monkeypatch):
    device = demo_phone()
    monkeypatch.setattr(agent_mod, "chat", _scripted([
        ("tap", {"index": 3}),                       # home -> Settings
        ("tap", {"index": 1}),                       # -> Network & internet
        ("tap", {"index": 1}),                       # Wi-Fi switch
        ("task_complete", {"summary": "Wi-Fi enabled"}),
    ]))
    ep = run_episode(device, "Turn on Wi-Fi.", AgentConfig(max_steps=8))
    assert ep.done and device.state["wifi"] is True
    assert ep.summary == "Wi-Fi enabled"
    assert ep.stop_reason == "task_complete"
    assert ep.n_steps == 4


def test_says_so_when_an_action_changes_nothing(monkeypatch):
    device = demo_phone()
    monkeypatch.setattr(agent_mod, "chat", _scripted([
        ("tap", {"index": 0}),  # "Phone" on home: goes nowhere
        ("task_complete", {"summary": "done"}),
    ]))
    ep = run_episode(device, "Do something.", AgentConfig(max_steps=4))
    assert "did not change" in ep.steps[0].result.message


def test_stops_when_it_repeats_a_no_op(monkeypatch):
    device = demo_phone()
    monkeypatch.setattr(agent_mod, "chat", _scripted([("tap", {"index": 0})] * 6))
    ep = run_episode(device, "Do something.", AgentConfig(max_steps=6))
    assert not ep.done
    assert "stuck" in ep.stop_reason
    assert ep.n_steps < 6, "should give up before the budget"


def test_empty_reply_retries_without_thinking(monkeypatch):
    """The measured failure: reasoning never closes `<think>`, so the parser
    returns nothing. A non-thinking retry is worth ~0.6 s."""
    seen = []

    def fake_chat(cfg, messages, *, think):
        seen.append(think)
        if think:
            return {"choices": [{"message": {"content": "", "tool_calls": []}}]}, 8.9
        return {"choices": [{"message": {"tool_calls": [
            {"function": {"name": "task_complete",
                          "arguments": '{"summary": "fallback"}'}}]}}]}, 0.61

    monkeypatch.setattr(agent_mod, "chat", fake_chat)
    ep = run_episode(device := demo_phone(), "Task.", AgentConfig(max_steps=3))
    assert seen == [True, False], seen
    assert ep.done and ep.summary == "fallback"
    assert ep.steps[0].thinking is False
    assert device is not None


def test_gives_up_rather_than_looping_when_no_tool_call_ever_comes(monkeypatch):
    monkeypatch.setattr(agent_mod, "chat", _scripted([]))
    ep = run_episode(demo_phone(), "Task.", AgentConfig(max_steps=5))
    assert not ep.done
    assert "no tool call" in ep.stop_reason
    assert ep.n_steps == 1


def test_step_budget_is_enforced(monkeypatch):
    device = demo_phone()
    # Alternate two real navigations so nothing is a repeat or a no-op.
    monkeypatch.setattr(agent_mod, "chat", _scripted(
        [("tap", {"index": 3}), ("press_key", {"key": "back"})] * 5))
    ep = run_episode(device, "Wander.", AgentConfig(max_steps=4))
    assert not ep.done
    assert ep.n_steps == 4
    assert "budget exhausted" in ep.stop_reason


def test_transcript_is_readable(monkeypatch):
    monkeypatch.setattr(agent_mod, "chat", _scripted([
        ("tap", {"index": 3}), ("task_complete", {"summary": "ok"})]))
    ep = run_episode(demo_phone(), "Open settings.", AgentConfig(max_steps=4))
    text = ep.transcript()
    assert "task: Open settings." in text
    assert "tap(" in text and "task_complete" in text
    assert "stopped: task_complete" in text


def test_detects_undoing_its_own_action(monkeypatch):
    """The measured failure: tapping a switch that is already on, turning it
    back off, forever. Each tap really does change the screen, so the
    no-change check cannot see it."""
    device = demo_phone()
    device.current = "network"
    monkeypatch.setattr(agent_mod, "chat", _scripted([("tap", {"index": 1})] * 6))
    ep = run_episode(device, "Turn on Wi-Fi.", AgentConfig(max_steps=6))
    assert not ep.done
    assert "undoing its own action" in ep.stop_reason
    assert any("undid your previous action" in s.result.message
               for s in ep.steps if s.result)


def test_normal_navigation_is_not_mistaken_for_oscillation(monkeypatch):
    """Opening a submenu and pressing back revisits a screen; that is how
    phones work and must not trip the guard."""
    device = demo_phone()
    monkeypatch.setattr(agent_mod, "chat", _scripted([
        ("tap", {"index": 3}),          # home -> settings
        ("press_key", {"key": "back"}),  # -> home
        ("tap", {"index": 2}),          # home -> clock
        ("press_key", {"key": "back"}),  # -> home
        ("task_complete", {"summary": "looked around"}),
    ]))
    ep = run_episode(device, "Look around.", AgentConfig(max_steps=8))
    assert ep.done, ep.stop_reason
    assert ep.n_steps == 5


# ------------------------------------------------------------------ task suite

def test_suite_scores_on_device_state_not_on_what_the_agent_said():
    """The failure this model exhibits is claiming completion it did not
    achieve, so the score must not read the transcript."""
    from tasks import SUITE_BY_ID

    task = SUITE_BY_ID["wifi_on"]
    device = task.build()
    assert task.score(device, declared_done=True)["success"] is False
    device.state["wifi"] = True
    assert task.score(device, declared_done=True)["success"] is True
    # Reaching the goal without stopping is not success either.
    assert task.score(device, declared_done=False)["success"] is False


def test_collateral_damage_fails_the_task():
    from tasks import SUITE_BY_ID

    task = SUITE_BY_ID["wifi_on"]
    device = task.build()
    device.state.update(wifi=True, airplane=True)
    score = task.score(device, declared_done=True)
    assert score["reached_goal"] and score["collateral_damage"]
    assert score["success"] is False


def test_already_satisfied_task_starts_satisfied():
    """Its only correct action is task_complete; tapping the switch fails."""
    from phone_state import read_state
    from tasks import SUITE_BY_ID

    task = SUITE_BY_ID["wifi_on_already"]
    device = task.build()
    assert task.reaches(read_state(device))
    assert task.score(device, declared_done=True)["success"] is True


def test_already_satisfied_task_fails_on_a_switch_round_trip():
    """W4A8 turned Wi-Fi on and then tapped the switch off again. Final state
    cannot see that, so this task scored it a pass until the flips were
    recorded."""
    from tasks import SUITE_BY_ID

    task = SUITE_BY_ID["wifi_on_already"]
    device = task.build()
    device.launch("Settings")
    device.tap(540, 360)          # Network & internet
    device.tap(540, 360)          # Wi-Fi off
    device.tap(540, 360)          # ... and on again
    assert device.state["wifi"] is True          # final state looks correct
    assert task.score(device, declared_done=True)["success"] is False


def test_goal_is_read_from_the_setting_not_from_the_final_screen():
    """An agent that turns Wi-Fi on and then goes Home has still done it."""
    from tasks import SUITE_BY_ID

    task = SUITE_BY_ID["wifi_on"]
    device = task.build()
    device.launch("Settings")
    device.tap(540, 360)
    device.tap(540, 360)
    device.keyevent("home")
    assert task.score(device, declared_done=True)["success"] is True


def test_never_finding_the_screen_is_not_scored_as_off():
    """`switch()` returns None when neither the setting nor the screen knows
    the control; a predicate must not read that as 'off' and pass a task."""
    from phone_state import PhoneState
    from tasks import SUITE_BY_ID

    empty = PhoneState(source="device")
    assert empty.switch("Wi-Fi") is None
    assert SUITE_BY_ID["wifi_off"].score(empty, declared_done=True)["success"] is False


def test_suite_includes_a_mirror_task_for_switch_tapping():
    """Guards against a model that has simply learned 'tap the switch'."""
    from tasks import SUITE_BY_ID

    from phone_state import read_state

    off = SUITE_BY_ID["wifi_off"]
    device = off.build()
    assert device.state["wifi"] is True
    assert not off.reaches(read_state(device))
    device.state["wifi"] = False
    assert off.reaches(read_state(device))


def test_chat_retries_transient_failures(monkeypatch):
    """A phone task is a dozen round trips; one dropped connection should cost
    a retry, not the episode."""
    import agent as mod
    from urllib import error as urlerror

    calls = {"n": 0}

    def flaky(req, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urlerror.URLError("connection reset")
        class _R:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b'{"choices":[{"message":{"tool_calls":[]}}]}'
        return _R()

    monkeypatch.setattr(mod.urlrequest, "urlopen", flaky)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    cfg = AgentConfig(retries=3)
    body, _ = mod.chat(cfg, [{"role": "user", "content": "x"}], think=False)
    assert calls["n"] == 3 and "choices" in body


def test_chat_gives_up_with_a_clear_error(monkeypatch):
    import agent as mod
    from urllib import error as urlerror

    monkeypatch.setattr(mod.urlrequest, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(urlerror.URLError("down")))
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    with pytest.raises(mod.ChatError, match="after 2 attempts"):
        mod.chat(AgentConfig(retries=2), [{"role": "user", "content": "x"}], think=False)


# --------------------------------------------------- oracles that were wrong

def test_alarm_task_needs_a_saved_alarm_not_a_typed_digit():
    """This oracle used to pass on any typed string containing "7", so typing
    `7` and stopping scored a success on "Set an alarm for 7:30 AM"."""
    from tasks import SUITE_BY_ID

    task = SUITE_BY_ID["add_alarm"]

    typed_only = task.build()
    typed_only.launch("Clock")
    typed_only.tap(540, 200)      # Alarm tab
    typed_only.tap(540, 360)      # Add alarm
    typed_only.tap(540, 360)
    typed_only.input_text("7")
    assert task.score(typed_only, declared_done=True)["success"] is False

    both_fields = task.build()
    both_fields.launch("Clock")
    both_fields.tap(540, 200)
    both_fields.tap(540, 360)
    both_fields.tap(540, 360)
    both_fields.input_text("7")
    both_fields.tap(540, 520)
    both_fields.input_text("30")
    assert task.score(both_fields, declared_done=True)["success"] is False, \
        "typing both fields without saving is not a set alarm"

    both_fields.tap(540, 840)     # Save
    assert task.score(both_fields, declared_done=True)["success"] is True


def test_alarm_task_rejects_the_wrong_meridiem():
    from tasks import SUITE_BY_ID

    task = SUITE_BY_ID["add_alarm"]
    d = task.build()
    d.launch("Clock"); d.tap(540, 200); d.tap(540, 360)
    d.tap(540, 360); d.input_text("7")
    d.tap(540, 520); d.input_text("30")
    d.tap(540, 680)               # AM -> PM
    d.tap(540, 840)               # Save
    score = task.score(d, declared_done=True)
    assert score["success"] is False and score["collateral_damage"] is True


def test_a_saved_alarm_is_visible_on_the_screen():
    """The oracle reads the clock app's list, so it works on a real phone."""
    from phone_state import read_state
    from tasks import SUITE_BY_ID

    d = SUITE_BY_ID["add_alarm"].build()
    d.launch("Clock"); d.tap(540, 200); d.tap(540, 360)
    d.tap(540, 360); d.input_text("7")
    d.tap(540, 520); d.input_text("30")
    d.tap(540, 840)
    assert read_state(d).has_text("7:30 AM")


# ------------------------------------------------------------------ variants

def test_every_task_has_variants_so_repeats_are_trials():
    """Five runs of one fixture at temperature 0 measure repeatability. The
    suite has to vary something for a repeat to be a trial."""
    from tasks import SUITE

    for task in SUITE:
        assert len(task.variants) >= 1
        ids = [v.variant_id for v in task.variants]
        assert len(ids) == len(set(ids)), task.task_id
    assert any(len(t.variants) > 1 for t in SUITE)


def test_variants_change_the_phone_but_not_the_goal():
    """A renamed or reordered control is still the same task."""
    from phone_state import read_state
    from tasks import SUITE_BY_ID

    task = SUITE_BY_ID["wifi_on"]
    labels_seen = set()
    for run in range(len(task.variants)):
        d = task.build(run)
        d.launch("Settings")
        d.tap(540, 360)
        labels_seen.add(read_state(d).source)
        # The goal predicate is unchanged; only the route differs.
        assert task.score(d, declared_done=True)["success"] is False
        d.state["wifi"] = True
        assert task.score(d, declared_done=True)["success"] is True
    assert labels_seen == {"sim"}


def test_wlan_variant_renames_the_control_the_agent_must_find():
    from phone_state import read_state
    from tasks import SUITE_BY_ID

    task = SUITE_BY_ID["wifi_on"]
    variant = next(i for i, v in enumerate(task.variants) if v.variant_id == "wlan")
    d = task.build(variant)
    d.launch("Settings"); d.tap(540, 360)
    assert read_state(d).has_text("WLAN")
    assert not read_state(d).has_text("Wi-Fi")


# -------------------------------------------------- real device honesty

def test_history_dependent_task_refuses_to_score_on_a_real_phone():
    """A phone cannot report which switches were flipped, so the task says so
    instead of scoring from final state and calling a round trip a pass."""
    from phone_state import PhoneState
    from tasks import SUITE_BY_ID

    state = PhoneState(source="device", settings={"wifi": True})
    score = SUITE_BY_ID["wifi_on_already"].score(state, declared_done=True)
    assert score["scorable"] is False
    assert "simulator only" in score["reason"]
    assert score["success"] is False


def test_tasks_runnable_on_a_phone_declare_how_to_reset_it():
    """Without a reset the run is scored against an unknown starting state."""
    from tasks import SUITE_BY_ID

    assert SUITE_BY_ID["wifi_on"].reset_adb
    assert SUITE_BY_ID["wifi_off"].reset_adb
    # and the sim-only one is not pretending otherwise
    assert SUITE_BY_ID["wifi_on_already"].needs == ("toggled",)
