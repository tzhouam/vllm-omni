"""Android screens as a text-only agent sees them, with the correct next action.

Each screen is a uiautomator-style accessibility dump flattened to one line per
element -- the same shape droidrun's Portal and AndroidWorld's a11y observation
hand to a text model. `expect` is the action a competent agent must take next;
`accept_names` allows a defensible alternative where one genuinely exists.
"""

SCENARIOS = [
    {
        "id": "launch_clock",
        "task": "Set an alarm for 7:30 AM.",
        "screen": """[0] TextView "Phone" clickable=true
[1] TextView "Messages" clickable=true
[2] TextView "Chrome" clickable=true
[3] TextView "Clock" clickable=true
[4] TextView "Settings" clickable=true
[5] TextView "Camera" clickable=true""",
        "expect": {"name": "tap", "args": {"index": 3}},
        "accept_names": ["tap", "open_app"],
    },
    {
        "id": "clock_alarm_tab",
        "task": "Set an alarm for 7:30 AM.",
        "screen": """[0] Button "Alarm" clickable=true selected=false
[1] Button "Clock" clickable=true selected=true
[2] Button "Timer" clickable=true selected=false
[3] Button "Stopwatch" clickable=true selected=false
[4] TextView "14:32" clickable=false""",
        "expect": {"name": "tap", "args": {"index": 0}},
        "accept_names": ["tap"],
    },
    {
        "id": "add_alarm",
        "task": "Set an alarm for 7:30 AM.",
        "screen": """[0] Button "Alarm" clickable=true selected=true
[1] TextView "No alarms" clickable=false
[2] ImageButton "Add alarm" clickable=true
[3] Button "Timer" clickable=true""",
        "expect": {"name": "tap", "args": {"index": 2}},
        "accept_names": ["tap"],
    },
    {
        "id": "wifi_settings",
        "task": "Turn on Wi-Fi.",
        "screen": """[0] TextView "Settings" clickable=false
[1] TextView "Network & internet" clickable=true
[2] TextView "Connected devices" clickable=true
[3] TextView "Apps" clickable=true
[4] TextView "Battery" clickable=true
[5] TextView "Display" clickable=true""",
        "expect": {"name": "tap", "args": {"index": 1}},
        "accept_names": ["tap"],
    },
    {
        "id": "wifi_toggle",
        "task": "Turn on Wi-Fi.",
        "screen": """[0] TextView "Network & internet" clickable=false
[1] Switch "Wi-Fi" checked=false clickable=true
[2] TextView "Mobile network" clickable=true
[3] Switch "Airplane mode" checked=false clickable=true""",
        "expect": {"name": "tap", "args": {"index": 1}},
        "accept_names": ["tap"],
    },
    {
        "id": "search_contact",
        "task": "Find the contact named Wei Zhang.",
        "screen": """[0] EditText "Search contacts" clickable=true focused=true
[1] TextView "Alice Chen" clickable=true
[2] TextView "Bob Lee" clickable=true
[3] TextView "Carol Wu" clickable=true""",
        "expect": {"name": "type_text", "args": {"index": 0, "text": "Wei Zhang"}},
        "accept_names": ["type_text"],
    },
    {
        "id": "scroll_for_item",
        "task": "Open the Developer options screen in Settings.",
        "screen": """[0] TextView "Settings" clickable=false
[1] TextView "Display" clickable=true
[2] TextView "Sound" clickable=true
[3] TextView "Storage" clickable=true
[4] TextView "Privacy" clickable=true
[5] TextView "Location" clickable=true""",
        "expect": {"name": "swipe", "args": {"direction": "up"}},
        "accept_names": ["swipe"],
    },
    {
        "id": "go_back",
        "task": "Return to the previous screen; you opened the wrong menu.",
        "screen": """[0] TextView "About phone" clickable=false
[1] TextView "Model: Pixel 8" clickable=false
[2] TextView "Android version" clickable=true""",
        "expect": {"name": "press_key", "args": {"key": "back"}},
        "accept_names": ["press_key"],
    },
    {
        "id": "open_app_by_name",
        "task": "Open the Camera app.",
        "screen": """[0] TextView "Swipe up to unlock" clickable=false
[1] TextView "10:04" clickable=false""",
        "expect": {"name": "open_app", "args": {"name": "Camera"}},
        "accept_names": ["open_app", "swipe"],
    },
    {
        "id": "task_done",
        "task": "Turn on Wi-Fi.",
        "screen": """[0] TextView "Network & internet" clickable=false
[1] Switch "Wi-Fi" checked=true clickable=true
[2] TextView "Connected to HOME-5G" clickable=false""",
        "expect": {"name": "task_complete", "args": {}},
        "accept_names": ["task_complete"],
    },
]
