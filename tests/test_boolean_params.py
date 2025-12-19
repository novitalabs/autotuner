#!/usr/bin/env python3
"""
Test script for boolean parameter handling in Docker controller.
"""

def test_boolean_parameter_handling():
    """Test that boolean parameters are handled correctly."""

    # Simulate the parameter building logic
    parameters = {
        "tp-size": 2,
        "mem-fraction-static": 0.85,
        "enable-mixed-chunk": True,
        "disable-cuda-graph": False,
        "enable-lora": True,
        "schedule-policy": "fcfs",
    }

    command_str = "python3 -m sglang.launch_server --model-path /model --port 8000"

    for param_name, param_value in parameters.items():
        # Convert parameter name to CLI format
        if not param_name.startswith("--"):
            cli_param = f"--{param_name}"
        else:
            cli_param = param_name

        # Handle boolean parameters specially
        if isinstance(param_value, bool):
            if param_value:  # Only add flag if True
                command_str += f" {cli_param}"
            # If False, skip this parameter entirely
        else:
            command_str += f" {cli_param} {param_value}"

    print("Generated command:")
    print(command_str)
    print()

    # Verify expectations
    assert "--enable-mixed-chunk" in command_str, "True boolean should be present as flag"
    assert "--disable-cuda-graph" not in command_str, "False boolean should be absent"
    assert "--enable-lora" in command_str, "True boolean should be present as flag"
    assert "--tp-size 2" in command_str, "Integer parameter should have value"
    assert "--mem-fraction-static 0.85" in command_str, "Float parameter should have value"
    assert "--schedule-policy fcfs" in command_str, "String parameter should have value"

    # Check that boolean flags don't have values
    assert "enable-mixed-chunk True" not in command_str, "Boolean flag should not have value"
    assert "enable-lora True" not in command_str, "Boolean flag should not have value"

    print("✅ All tests passed!")
    print()
    print("Expected behavior:")
    print("  - enable-mixed-chunk: True  → --enable-mixed-chunk (flag only)")
    print("  - disable-cuda-graph: False → (omitted)")
    print("  - enable-lora: True         → --enable-lora (flag only)")
    print("  - tp-size: 2                → --tp-size 2")
    print("  - mem-fraction-static: 0.85 → --mem-fraction-static 0.85")
    print("  - schedule-policy: fcfs     → --schedule-policy fcfs")

if __name__ == "__main__":
    test_boolean_parameter_handling()
