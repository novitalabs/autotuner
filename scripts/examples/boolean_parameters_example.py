#!/usr/bin/env python3
"""
Example: Using boolean parameters in presets.

This script demonstrates how boolean parameters work in the autotuner:
- true values: parameter flag is added without value
- false values: parameter is omitted from command entirely

This is the correct behavior for CLI flags that act as switches.
"""

import json
import requests

API_BASE = "http://localhost:8000"

def create_boolean_preset_example():
    """Create a preset that uses boolean parameters."""

    preset = {
        "name": "Boolean Parameters Example",
        "description": "Demonstrates boolean parameter handling",
        "category": "example",
        "runtime": "sglang",
        "parameters": {
            # Regular parameters with values
            "tensor-parallel-size": [1, 2],
            "mem-fraction-static": [0.85],
            "schedule-policy": ["fcfs"],

            # Boolean parameters
            # true = flag is present: --enable-mixed-chunk
            "enable-mixed-chunk": [True],

            # false = flag is omitted (not in command at all)
            "disable-cuda-graph": [False],

            # Multiple boolean values for experimentation
            "enable-lora": [True, False],  # Try with and without LoRA

            # Another boolean flag
            "disable-radix-cache": [False],
        }
    }

    print("Creating preset with boolean parameters...")
    print(json.dumps(preset, indent=2))
    print()

    response = requests.post(f"{API_BASE}/api/presets/", json=preset)

    if response.status_code == 201:
        created = response.json()
        print(f"✅ Preset created successfully (ID: {created['id']})")
        print()
        print("Parameter combinations that will be generated:")
        print()

        # Generate combinations manually to show what will happen
        combinations = [
            # tp=1, enable-lora=True
            {"tensor-parallel-size": 1, "mem-fraction-static": 0.85,
             "schedule-policy": "fcfs", "enable-mixed-chunk": True,
             "disable-cuda-graph": False, "enable-lora": True,
             "disable-radix-cache": False},
            # tp=1, enable-lora=False
            {"tensor-parallel-size": 1, "mem-fraction-static": 0.85,
             "schedule-policy": "fcfs", "enable-mixed-chunk": True,
             "disable-cuda-graph": False, "enable-lora": False,
             "disable-radix-cache": False},
            # tp=2, enable-lora=True
            {"tensor-parallel-size": 2, "mem-fraction-static": 0.85,
             "schedule-policy": "fcfs", "enable-mixed-chunk": True,
             "disable-cuda-graph": False, "enable-lora": True,
             "disable-radix-cache": False},
            # tp=2, enable-lora=False
            {"tensor-parallel-size": 2, "mem-fraction-static": 0.85,
             "schedule-policy": "fcfs", "enable-mixed-chunk": True,
             "disable-cuda-graph": False, "enable-lora": False,
             "disable-radix-cache": False},
        ]

        for i, combo in enumerate(combinations, 1):
            print(f"Combination {i}:")
            # Show what the command would look like
            cmd_parts = ["python3 -m sglang.launch_server --model-path /model --port 8000"]
            for param, value in combo.items():
                if isinstance(value, bool):
                    if value:
                        cmd_parts.append(f"--{param}")
                    # False values are skipped
                else:
                    cmd_parts.append(f"--{param} {value}")

            print(f"  Command: {' '.join(cmd_parts)}")
            print()

        print("Key observations:")
        print("  - enable-mixed-chunk: True  → --enable-mixed-chunk (always present)")
        print("  - disable-cuda-graph: False → (always omitted)")
        print("  - enable-lora: True         → --enable-lora (present)")
        print("  - enable-lora: False        → (omitted)")
        print("  - disable-radix-cache: False → (always omitted)")
        print()

    else:
        print(f"❌ Failed to create preset: {response.text}")
        return None

    return created['id']


def ui_usage_guide():
    """Print guide for using boolean parameters in the UI."""
    print("=" * 70)
    print("How to use boolean parameters in the UI")
    print("=" * 70)
    print()
    print("When editing a preset in the web UI:")
    print()
    print("1. Add a parameter (e.g., 'enable-mixed-chunk')")
    print()
    print("2. In the Values field, enter boolean values:")
    print("   - For testing with flag enabled:  true")
    print("   - For testing with flag disabled: false")
    print("   - For testing both:               true, false")
    print()
    print("3. The frontend will parse these as boolean values")
    print()
    print("4. When the experiment runs:")
    print("   - true  → --enable-mixed-chunk (flag is added)")
    print("   - false → (flag is not added to command)")
    print()
    print("Examples:")
    print("  Parameter: enable-lora")
    print("  Values: true, false")
    print("  → Creates 2 experiments: one with --enable-lora, one without")
    print()
    print("  Parameter: disable-cuda-graph")
    print("  Values: false")
    print("  → Flag is never added (CUDA graphs remain enabled)")
    print()
    print("  Parameter: enable-prefix-caching")
    print("  Values: true")
    print("  → Flag is always added (prefix caching enabled)")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("Boolean Parameter Handling Example")
    print("=" * 70)
    print()

    # Show UI usage guide
    ui_usage_guide()

    print()
    print("=" * 70)
    print("Creating example preset via API")
    print("=" * 70)
    print()

    # Create example preset
    preset_id = create_boolean_preset_example()

    if preset_id:
        print()
        print(f"You can view this preset at: http://localhost:5173/presets")
        print(f"Or fetch it via API: curl http://localhost:8000/api/presets/{preset_id}")
