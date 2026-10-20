"""Android control action space for a text-only LLM agent.

The tool surface mirrors what the mainstream ADB-based harnesses expose
(droidrun/mobilerun's Portal tools, ghost-in-the-droid's touch/see tools,
AndroidWorld's `JSONAction`), so an agent written against these schemas can be
pointed at any of them. Every action is expressed over the accessibility tree
rather than pixel coordinates, because Spark-X2.5 is a text-only model: it
never sees a screenshot, it sees the uiautomator dump.

`ADB_TEMPLATES` gives the literal `adb` invocation each tool maps to, which is
what a driver executes against a real device.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "tap",
            "description": (
                "Tap a UI element. Use the element's index exactly as shown in "
                "the accessibility tree."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "Index of the element in the tree",
                    }
                },
                "required": ["index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": (
                "Type text into a focused input field. Tap the field first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "description": "Index of the input field"},
                    "text": {"type": "string", "description": "Text to enter"},
                },
                "required": ["index", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "swipe",
            "description": (
                "Scroll the screen. The direction is where the content should "
                "move: use 'down' to see what is further down the page."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down", "left", "right"],
                    }
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "press_key",
            "description": "Press a hardware or system key.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "enum": ["back", "home", "enter", "delete", "recent"],
                    }
                },
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": "Launch an app by its visible name.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": (
                "Call when the user's request has been fully carried out."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "What was done"}
                },
                "required": ["summary"],
            },
        },
    },
]

# How each tool becomes a real device command. `{bounds}` is filled from the
# element's centre point in the accessibility tree.
ADB_TEMPLATES = {
    "tap": "adb shell input tap {x} {y}",
    "type_text": "adb shell input text {text_escaped}",
    "swipe": "adb shell input swipe {x1} {y1} {x2} {y2} 300",
    "press_key": "adb shell input keyevent {keycode}",
    "open_app": "adb shell monkey -p {package} -c android.intent.category.LAUNCHER 1",
}

KEYCODES = {
    "back": "KEYCODE_BACK",
    "home": "KEYCODE_HOME",
    "enter": "KEYCODE_ENTER",
    "delete": "KEYCODE_DEL",
    "recent": "KEYCODE_APP_SWITCH",
}

SYSTEM_PROMPT = (
    "You control an Android phone. At each step you are given the current "
    "screen as an accessibility tree: one line per element, with its index, "
    "type, text and whether it is clickable. Call exactly one tool per step to "
    "make progress on the user's task. Refer to elements only by the index "
    "shown.\n"
    # Both rules below exist because the model failed them in measurement: it
    # tapped a Wi-Fi switch that already read checked=true, turning it back
    # off, and never called task_complete. A switch's state is on the screen,
    # so the rule can be stated in terms of what it can see.
    "A switch or checkbox shows its state as checked=true or checked=false. "
    "Tapping it flips that state.\n"
    "Before acting, check whether the screen already shows what the task "
    "asked for. If it does, call task_complete immediately and do not tap "
    "anything else -- acting again would undo it."
)
