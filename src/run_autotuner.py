#!/usr/bin/env python3
"""
Inference Autotuner - CLI Entry Point

Command-line interface for running parameter tuning experiments.
"""

import json
import sys
from pathlib import Path

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import AutotunerOrchestrator


def load_task_file(task_file: Path) -> dict:
	"""Load task configuration from JSON or YAML file.

	Args:
	    task_file: Path to task configuration file (JSON or YAML)

	Returns:
	    Task configuration dictionary
	"""
	suffix = task_file.suffix.lower()
	with open(task_file) as f:
		if suffix in ['.yaml', '.yml']:
			return yaml.safe_load(f)
		else:
			return json.load(f)


def main():
	"""Main entry point."""
	import argparse

	parser = argparse.ArgumentParser(
		description="Inference Autotuner - Find optimal parameters for LLM inference",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  # OME (Kubernetes) mode with K8s BenchmarkJob
  python run_autotuner.py examples/simple_task.yaml

  # OME mode with direct genai-bench CLI
  python run_autotuner.py examples/simple_task.yaml --direct

  # Standalone Docker mode (--direct is automatic)
  python run_autotuner.py examples/simple_task.yaml --mode docker

  # Local subprocess mode (no Docker required)
  python run_autotuner.py examples/local_task.yaml --mode local

  # Docker mode with custom model path and verbose output
  python run_autotuner.py examples/docker_task.yaml --mode docker --model-path /data/models --verbose
        """,
	)

	parser.add_argument("task_file", type=Path, help="Path to task configuration file (JSON or YAML)")
	parser.add_argument(
		"--mode",
		choices=["ome", "docker", "local"],
		default="ome",
		help="Deployment mode: ome (Kubernetes), docker (standalone), or local (subprocess)",
	)
	parser.add_argument("--kubeconfig", type=str, default=None, help="Path to kubeconfig file (for OME mode)")
	parser.add_argument(
		"--direct",
		action="store_true",
		help="Use direct genai-bench CLI instead of K8s BenchmarkJob (default for docker mode)",
	)
	parser.add_argument(
		"--model-path",
		type=str,
		default="/mnt/data/models",
		help="Base path for models on host (Docker/Local mode)",
	)
	parser.add_argument(
		"--verbose",
		"-v",
		action="store_true",
		help="Stream genai-bench output in real-time (useful for debugging)",
	)

	args = parser.parse_args()

	# Docker and Local modes always use direct benchmark
	if args.mode in ["docker", "local"]:
		args.direct = True

	if not args.task_file.exists():
		print(f"Error: Task file not found: {args.task_file}")
		sys.exit(1)

	# Load task configuration
	task = load_task_file(args.task_file)

	# Run autotuning
	orchestrator = AutotunerOrchestrator(
		deployment_mode=args.mode,
		kubeconfig_path=args.kubeconfig,
		use_direct_benchmark=args.direct,
		docker_model_path=args.model_path,
		verbose=args.verbose,
	)
	summary = orchestrator.run_task(task)

	# Save results
	results_file = Path("results") / f"{summary['task_name']}_results.json"
	results_file.parent.mkdir(exist_ok=True)
	with open(results_file, "w") as f:
		json.dump(summary, f, indent=2)
	print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
	main()
