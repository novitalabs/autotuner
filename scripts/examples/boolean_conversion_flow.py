#!/usr/bin/env python3
"""
Demonstration: How "true,false" string becomes boolean values

This script traces the complete flow from user input to Docker command.
"""

def demonstrate_conversion_flow():
    """Show step-by-step conversion from string to boolean."""

    print("=" * 70)
    print("FLOW: User Input → Boolean Values → Docker Command")
    print("=" * 70)
    print()

    # Step 1: User input in UI
    print("STEP 1: User Input in Web UI")
    print("-" * 70)
    print("User opens preset editor and adds parameter:")
    print("  Parameter name: enable-lora")
    print("  Values field:   'true, false'  (user types this as a string)")
    print()

    # Step 2: Frontend parsing
    print("STEP 2: Frontend Parsing (parseParameterValue in PresetEditModal.tsx)")
    print("-" * 70)
    print("Code location: frontend/src/components/PresetEditModal.tsx:72-90")
    print()

    # Simulate the frontend parsing logic
    valueStr = "true, false"
    print(f"Input string: '{valueStr}'")
    print()

    # Split and trim
    parts = [s.strip() for s in valueStr.split(',') if s.strip()]
    print(f"After split & trim: {parts}")
    print()

    # Check if all parts are booleans
    allBooleans = all(part == 'true' or part == 'false' for part in parts)
    print(f"All parts are 'true' or 'false'? {allBooleans}")
    print()

    if allBooleans:
        # Convert to actual booleans
        parsed = [part == 'true' for part in parts]
        print(f"Converted to booleans: {parsed}")
        print(f"Type of first element: {type(parsed[0])}")
    print()

    print("Frontend code:")
    print("```typescript")
    print("const parts = valueStr.split(',').map(s => s.trim()).filter(Boolean);")
    print("const allBooleans = parts.every(part => part === 'true' || part === 'false');")
    print("if (allBooleans) {")
    print("  return parts.map(part => part === 'true');  // true → true, false → false")
    print("}")
    print("```")
    print()

    # Step 3: API request
    print("STEP 3: API Request to Backend")
    print("-" * 70)
    print("Frontend sends JSON to backend:")
    print("POST /api/presets/")
    print("{")
    print('  "name": "Test Preset",')
    print('  "parameters": {')
    print('    "enable-lora": [true, false]  ← Actual boolean values, not strings!')
    print("  }")
    print("}")
    print()

    # Step 4: Database storage
    print("STEP 4: Database Storage")
    print("-" * 70)
    print("Backend stores in SQLite as JSON:")
    print('{"enable-lora": [true, false]}')
    print()
    print("Note: SQLite stores true/false as 1/0 or native JSON booleans")
    print()

    # Step 5: Experiment generation
    print("STEP 5: Experiment Generation (generate_parameter_grid)")
    print("-" * 70)
    print("Code location: src/utils/optimizer.py:9-57")
    print()
    print("Parameters loaded from database:")
    print("  {'enable-lora': [true, false]}")
    print()
    print("Grid generator creates combinations:")
    print("  Combination 1: {'enable-lora': true}")
    print("  Combination 2: {'enable-lora': false}")
    print()
    print("Python code:")
    print("```python")
    print("for param_name, spec in parameter_spec.items():")
    print("    if isinstance(spec, list):")
    print("        param_values.append(spec)  # Preserves boolean type!")
    print("```")
    print()

    # Step 6: Docker command building
    print("STEP 6: Docker Command Building")
    print("-" * 70)
    print("Code location: src/controllers/docker_controller.py:147-156")
    print()

    # Simulate command building for both combinations
    combinations = [
        {"tensor-parallel-size": 2, "enable-lora": True},
        {"tensor-parallel-size": 2, "enable-lora": False},
    ]

    for i, params in enumerate(combinations, 1):
        print(f"Combination {i} parameters: {params}")
        print()

        command_str = "python3 -m sglang.launch_server --model-path /model --port 8000"

        for param_name, param_value in params.items():
            cli_param = f"--{param_name}"

            print(f"  Processing: {param_name} = {param_value} (type: {type(param_value).__name__})")

            if isinstance(param_value, bool):
                if param_value:
                    command_str += f" {cli_param}"
                    print(f"    → Boolean True: add '{cli_param}' flag")
                else:
                    print(f"    → Boolean False: skip parameter")
            else:
                command_str += f" {cli_param} {param_value}"
                print(f"    → Non-boolean: add '{cli_param} {param_value}'")

        print()
        print(f"  Final command:")
        print(f"  {command_str}")
        print()

    print("Python code:")
    print("```python")
    print("for param_name, param_value in parameters.items():")
    print("    if isinstance(param_value, bool):  # Type check!")
    print("        if param_value:")
    print("            command_str += f' --{param_name}'  # Flag only")
    print("        # If False, skip entirely")
    print("    else:")
    print("        command_str += f' --{param_name} {param_value}'  # With value")
    print("```")
    print()


def test_type_preservation():
    """Verify that types are preserved through the flow."""
    print("=" * 70)
    print("TYPE PRESERVATION VERIFICATION")
    print("=" * 70)
    print()

    # Simulate the entire flow
    import json

    # Frontend parsing (as if from user input "true, false")
    user_input = "true, false"
    parts = [s.strip() for s in user_input.split(',')]
    parsed_values = [part == 'true' for part in parts]

    print(f"1. User input: '{user_input}' (string)")
    print(f"2. Parsed values: {parsed_values}")
    print(f"   Type: {type(parsed_values)}, Element types: {[type(v).__name__ for v in parsed_values]}")
    print()

    # JSON serialization (API request)
    json_data = {"enable-lora": parsed_values}
    json_str = json.dumps(json_data)

    print(f"3. JSON serialization: {json_str}")
    print(f"   Note: true/false in JSON (not 'true'/'false' strings)")
    print()

    # JSON deserialization (backend receives)
    received_data = json.loads(json_str)

    print(f"4. Backend receives: {received_data}")
    print(f"   Type: {type(received_data['enable-lora'])}")
    print(f"   Element types: {[type(v).__name__ for v in received_data['enable-lora']]}")
    print()

    # Verify isinstance check works
    for value in received_data['enable-lora']:
        is_bool = isinstance(value, bool)
        print(f"5. isinstance({value}, bool) = {is_bool} ✓")
    print()

    print("✅ Type preservation confirmed through entire flow!")
    print()


def show_type_difference():
    """Show the difference between string booleans and actual booleans."""
    print("=" * 70)
    print("IMPORTANT: String vs Boolean Difference")
    print("=" * 70)
    print()

    # String booleans (WRONG)
    string_true = "true"
    string_false = "false"

    # Actual booleans (CORRECT)
    bool_true = True
    bool_false = False

    print("String booleans (what we DON'T want):")
    print(f"  'true' → type: {type(string_true)}, isinstance bool: {isinstance(string_true, bool)}")
    print(f"  'false' → type: {type(string_false)}, isinstance bool: {isinstance(string_false, bool)}")
    print()

    print("Actual booleans (what we DO want):")
    print(f"  True → type: {type(bool_true)}, isinstance bool: {isinstance(bool_true, bool)}")
    print(f"  False → type: {type(bool_false)}, isinstance bool: {isinstance(bool_false, bool)}")
    print()

    print("Why this matters for our code:")
    print()
    print("  if isinstance(param_value, bool):  # Only matches True/False, not 'true'/'false'")
    print("      # This block handles boolean flags")
    print()
    print("If we had string 'true', it would take the else branch and produce:")
    print("  --enable-lora true  ← WRONG! (flag with string value)")
    print()
    print("With actual True boolean, we get:")
    print("  --enable-lora  ← CORRECT! (flag only, no value)")
    print()


if __name__ == "__main__":
    demonstrate_conversion_flow()
    print("\n")
    test_type_preservation()
    print("\n")
    show_type_difference()
