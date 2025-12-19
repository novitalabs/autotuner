#!/usr/bin/env python3
"""
Test script to verify quant_config expansion logic.
"""

import sys
sys.path.insert(0, 'src')

from utils.quantization_integration import expand_quant_config_to_parameter_spec, merge_parameters_with_quant_config
from utils.optimizer import generate_parameter_grid

# Test 1: Simple quant_config with arrays
print("=" * 60)
print("Test 1: Simple quant_config with arrays")
print("=" * 60)

quant_config = {
    "gemm_dtype": ["auto", "fp8"],
    "kvcache_dtype": ["auto", "fp8_e5m2"],
    "attention_dtype": "auto",
    "moe_dtype": "auto"
}

param_spec = expand_quant_config_to_parameter_spec(quant_config)
print(f"Quant config: {quant_config}")
print(f"Parameter spec: {param_spec}")

grid = generate_parameter_grid(param_spec)
print(f"Generated {len(grid)} combinations:")
for i, combo in enumerate(grid, 1):
    print(f"  {i}. {combo}")

# Test 2: Merge with existing parameters
print("\n" + "=" * 60)
print("Test 2: Merge with existing parameters")
print("=" * 60)

base_params = {
    "tp-size": [1, 2]
}

merged = merge_parameters_with_quant_config(base_params, quant_config)
print(f"Base parameters: {base_params}")
print(f"Quant config: {quant_config}")
print(f"Merged parameters: {merged}")

grid = generate_parameter_grid(merged)
print(f"Generated {len(grid)} combinations:")
for i, combo in enumerate(grid, 1):
    print(f"  {i}. {combo}")

# Test 3: Task 9's actual quant_config
print("\n" + "=" * 60)
print("Test 3: Task 9's actual quant_config")
print("=" * 60)

task9_quant = {
    "gemm_dtype": ["auto", "float16", "float32", "bfloat16", "int8"],
    "kvcache_dtype": ["auto", "fp8_e4m3", "fp16", "fp8_e5m2", "bfloat16", "int8", "int4", "fp8"],
    "attention_dtype": ["auto", "float16", "fp8", "bfloat16", "fp8_block", "fp8_e4m3", "fp8_e5m2"],
    "moe_dtype": ["auto", "float16", "w4afp8", "mxfp4", "int8", "bfloat16", "fp8"]
}

param_spec = expand_quant_config_to_parameter_spec(task9_quant)
print(f"Parameter spec: {param_spec}")

grid = generate_parameter_grid(param_spec)
print(f"Generated {len(grid)} combinations (showing first 10):")
for i, combo in enumerate(grid[:10], 1):
    print(f"  {i}. {combo}")
print(f"... and {len(grid) - 10} more combinations")
print(f"Total combinations: {len(grid)}")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
