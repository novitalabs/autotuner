"""Controllers package for managing OME, Docker, Local and Benchmark resources.

Controllers are imported lazily to avoid dependency issues.
Use the orchestrator to select the appropriate controller based on deployment mode.
"""

# Lazy imports - only import when explicitly requested
__all__ = ['LocalController', 'DockerController', 'OMEController', 'BenchmarkController', 'DirectBenchmarkController']

def __getattr__(name):
    """Lazy import controllers to avoid loading unnecessary dependencies."""
    if name == 'LocalController':
        from controllers.local_controller import LocalController
        return LocalController
    elif name == 'DockerController':
        from controllers.docker_controller import DockerController
        return DockerController
    elif name == 'OMEController':
        from controllers.ome_controller import OMEController
        return OMEController
    elif name == 'BenchmarkController':
        from controllers.benchmark_controller import BenchmarkController
        return BenchmarkController
    elif name == 'DirectBenchmarkController':
        from controllers.direct_benchmark_controller import DirectBenchmarkController
        return DirectBenchmarkController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
