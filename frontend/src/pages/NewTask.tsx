import { useState, useEffect, useRef } from 'react';
import { useMutation, useQueryClient, useQuery } from '@tanstack/react-query';
import { apiClient } from '../services/api';
import toast from 'react-hot-toast';
import { navigateTo } from '../components/Layout';
import { getEditingTaskId } from '../utils/editTaskStore';
import { useEscapeKey } from '@/hooks/useEscapeKey';
import type { Task, QuantizationConfig, ParallelConfig } from '../types/api';
import PresetSelector from '../components/PresetSelector';
import { QuantizationConfigForm } from '../components/QuantizationConfigForm';
import { ParallelConfigForm } from '../components/ParallelConfigForm';
import yaml from 'js-yaml';

interface TaskFormData {
  task_name: string;
  description: string;
  deployment_mode: string;
  base_runtime: string;
  runtime_image_tag?: string;
  model: {
    id_or_path: string;
    namespace: string;
  };
  parameters: Record<string, any[]>;
  quant_config?: QuantizationConfig;
  parallel_config?: ParallelConfig;
  slo?: Record<string, any>;
  optimization: {
    strategy: string;
    objective: string;
    max_iterations: number;
    timeout_per_iteration: number;
  };
  benchmark: {
    task: string;
    model_name: string;
    model_tokenizer: string;
    traffic_scenarios: string[];
    num_concurrency: number[];
    max_time_per_iteration: number;
    max_requests_per_iteration: number;
    additional_params: Record<string, any>;
  };
}

interface ParamField {
  name: string;
  values: string;
}

export default function NewTask() {
  // Handle Escape key to go back to tasks page
  useEscapeKey(() => navigateTo("tasks"));

  const queryClient = useQueryClient();
  // Edit mode support
  const [editingTaskId, setEditingTaskId] = useState<number | null>(null);
  const [originalTask, setOriginalTask] = useState<Task | null>(null);

  // Check for edit mode on mount
  useEffect(() => {
    const taskId = getEditingTaskId();
    if (taskId) {
      setEditingTaskId(taskId);
    }

    // Check for duplicate task configuration from sessionStorage
    const duplicateConfigStr = sessionStorage.getItem('duplicateTaskConfig');
    if (duplicateConfigStr) {
      try {
        const duplicateConfig = JSON.parse(duplicateConfigStr);

        // Load the duplicate configuration into form
        setTaskName(duplicateConfig.task_name || '');
        setDescription(duplicateConfig.description || '');
        setDeploymentMode(duplicateConfig.deployment_mode || 'docker');
        setBaseRuntime(duplicateConfig.base_runtime || 'sglang');
        setRuntimeImageTag(duplicateConfig.runtime_image_tag || '');

        // Model config
        if (duplicateConfig.model) {
          setModelIdOrPath(duplicateConfig.model.id_or_path || '');
          setModelNamespace(duplicateConfig.model.namespace || 'autotuner');
        }

        // Parameters - convert from API format to form format
        if (duplicateConfig.parameters) {
          const params: ParamField[] = [];
          for (const [key, value] of Object.entries(duplicateConfig.parameters)) {
            if (Array.isArray(value)) {
              params.push({ name: key, values: value.join(', ') });
            }
          }
          if (params.length > 0) {
            setParameters(params);
          }
        }

        // Optimization
        if (duplicateConfig.optimization) {
          setStrategy(duplicateConfig.optimization.strategy || 'grid_search');
          setObjective(duplicateConfig.optimization.objective || 'minimize_latency');
          setMaxIterations(duplicateConfig.optimization.max_iterations || 2);
          setTimeoutPerIteration(duplicateConfig.optimization.timeout_per_iteration || 600);
        }

        // Benchmark
        if (duplicateConfig.benchmark) {
          setBenchmarkTask(duplicateConfig.benchmark.task || 'text-to-text');
          setBenchmarkModelName(duplicateConfig.benchmark.model_name || '');
          setModelTokenizer(duplicateConfig.benchmark.model_tokenizer || '');
          setTrafficScenarios(duplicateConfig.benchmark.traffic_scenarios?.join(', ') || 'D(100,100)');
          setNumConcurrency(duplicateConfig.benchmark.num_concurrency?.join(', ') || '1, 4');
          setMaxTimePerIteration(duplicateConfig.benchmark.max_time_per_iteration || 10);
          setMaxRequestsPerIteration(duplicateConfig.benchmark.max_requests_per_iteration || 50);
          setTemperature(duplicateConfig.benchmark.additional_params?.temperature?.toString() || '0.0');
          // Dataset URL
          if (duplicateConfig.benchmark.dataset_url) {
            setDatasetUrl(duplicateConfig.benchmark.dataset_url);
          }
          if (duplicateConfig.benchmark.dataset_deduplicate !== undefined) {
            setDatasetDeduplicate(duplicateConfig.benchmark.dataset_deduplicate);
          }
        }

        // SLO Configuration
        if (duplicateConfig.slo) {
          setEnableSLO(true);

          // Latency metrics
          if (duplicateConfig.slo.latency) {
            const latency = duplicateConfig.slo.latency;

            if (latency.p50) {
              setEnableP50(true);
              setSloP50Threshold(latency.p50.threshold?.toString() || '2.0');
              setSloP50Weight(latency.p50.weight?.toString() || '1.0');
            }

            if (latency.p90) {
              setEnableP90(true);
              setSloP90Threshold(latency.p90.threshold?.toString() || '5.0');
              setSloP90Weight(latency.p90.weight?.toString() || '2.0');
              setSloP90HardFail(latency.p90.hard_fail || false);
              setSloP90FailRatio(latency.p90.fail_ratio?.toString() || '0.2');
            }

            if (latency.p99) {
              setEnableP99(true);
              setSloP99Threshold(latency.p99.threshold?.toString() || '10.0');
              setSloP99Weight(latency.p99.weight?.toString() || '3.0');
              setSloP99HardFail(latency.p99.hard_fail || false);
              setSloP99FailRatio(latency.p99.fail_ratio?.toString() || '0.5');
            }
          }

          // TTFT
          if (duplicateConfig.slo.ttft) {
            setEnableTTFT(true);
            setSloTtftThreshold(duplicateConfig.slo.ttft.threshold?.toString() || '1.0');
            setSloTtftWeight(duplicateConfig.slo.ttft.weight?.toString() || '2.0');
          }

          // TPOT
          if (duplicateConfig.slo.tpot) {
            setEnableTPOT(true);
            setSloTpotThreshold(duplicateConfig.slo.tpot.threshold?.toString() || '0.05');
            setSloTpotWeight(duplicateConfig.slo.tpot.weight?.toString() || '2.0');
          }

          // Steepness
          if (duplicateConfig.slo.steepness !== undefined) {
            setSloSteepness(duplicateConfig.slo.steepness.toString());
          }
        }

        // Quantization Configuration
        if (duplicateConfig.quant_config) {
          setQuantConfig(duplicateConfig.quant_config);
        }

        // Parallel Configuration
        if (duplicateConfig.parallel_config) {
          setParallelConfig(duplicateConfig.parallel_config);
        }

        // Clear the sessionStorage after loading
        sessionStorage.removeItem('duplicateTaskConfig');

        toast.success('Task configuration loaded for duplication');
      } catch (error) {
        console.error('Failed to parse duplicate task config:', error);
        sessionStorage.removeItem('duplicateTaskConfig');
      }
    }
  }, []);

  // Fetch task if editing
  const { data: taskToEdit } = useQuery({
    queryKey: ['task', editingTaskId],
    queryFn: () => editingTaskId ? apiClient.getTask(editingTaskId) : null,
    enabled: editingTaskId !== null,
  });

  // Pre-populate form when task data is loaded
  useEffect(() => {
    if (taskToEdit) {
      setOriginalTask(taskToEdit);
      
      // Basic info
      setTaskName(taskToEdit.task_name);
      setDescription(taskToEdit.description || '');
      setDeploymentMode(taskToEdit.deployment_mode);
      setBaseRuntime(taskToEdit.base_runtime);
      setRuntimeImageTag(taskToEdit.runtime_image_tag || '');

      // Model config
      setModelIdOrPath(taskToEdit.model?.id_or_path || '');
      setModelNamespace(taskToEdit.model?.namespace || 'autotuner');

      // Parameters - convert from API format to form format
      const params: ParamField[] = [];
      if (taskToEdit.parameters) {
        for (const [key, value] of Object.entries(taskToEdit.parameters)) {
          if (Array.isArray(value)) {
            params.push({ name: key, values: value.join(', ') });
          }
        }
      }
      if (params.length > 0) {
        setParameters(params);
      }

      // Optimization
      if (taskToEdit.optimization) {
        setStrategy(taskToEdit.optimization.strategy || 'grid_search');
        setObjective(taskToEdit.optimization.objective || 'minimize_latency');
        setMaxIterations(taskToEdit.optimization.max_iterations || 2);
        setTimeoutPerIteration(taskToEdit.optimization.timeout_per_iteration || 600);
      }

      // Benchmark
      if (taskToEdit.benchmark) {
        setBenchmarkTask(taskToEdit.benchmark.task || 'text-to-text');
        setBenchmarkModelName(taskToEdit.benchmark.model_name || '');
        setModelTokenizer(taskToEdit.benchmark.model_tokenizer || '');
        setTrafficScenarios(taskToEdit.benchmark.traffic_scenarios?.join(', ') || 'D(100,100)');
        setNumConcurrency(taskToEdit.benchmark.num_concurrency?.join(', ') || '1, 4');
        setMaxTimePerIteration(taskToEdit.benchmark.max_time_per_iteration || 10);
        setMaxRequestsPerIteration(taskToEdit.benchmark.max_requests_per_iteration || 50);
        setTemperature(taskToEdit.benchmark.additional_params?.temperature?.toString() || '0.0');
        // Dataset URL
        if (taskToEdit.benchmark.dataset_url) {
          setDatasetUrl(taskToEdit.benchmark.dataset_url);
        }
        if (taskToEdit.benchmark.dataset_deduplicate !== undefined) {
          setDatasetDeduplicate(taskToEdit.benchmark.dataset_deduplicate);
        }
      }

      // Quantization Configuration
      if (taskToEdit.quant_config) {
        setQuantConfig(taskToEdit.quant_config);
      }
      if (taskToEdit.parallel_config) {
        setParallelConfig(taskToEdit.parallel_config);
      }

      // SLO Configuration
      if (taskToEdit.slo) {
        setEnableSLO(true);

        // Latency metrics
        if (taskToEdit.slo.latency) {
          const latency = taskToEdit.slo.latency;

          if (latency.p50) {
            setEnableP50(true);
            setSloP50Threshold(latency.p50.threshold?.toString() || '2.0');
            setSloP50Weight(latency.p50.weight?.toString() || '1.0');
          }

          if (latency.p90) {
            setEnableP90(true);
            setSloP90Threshold(latency.p90.threshold?.toString() || '5.0');
            setSloP90Weight(latency.p90.weight?.toString() || '2.0');
            setSloP90HardFail(latency.p90.hard_fail || false);
            if (latency.p90.fail_ratio !== undefined) {
              setSloP90FailRatio(latency.p90.fail_ratio.toString());
            }
          }

          if (latency.p99) {
            setEnableP99(true);
            setSloP99Threshold(latency.p99.threshold?.toString() || '10.0');
            setSloP99Weight(latency.p99.weight?.toString() || '3.0');
            setSloP99HardFail(latency.p99.hard_fail || false);
            if (latency.p99.fail_ratio !== undefined) {
              setSloP99FailRatio(latency.p99.fail_ratio.toString());
            }
          }
        }

        // TTFT
        if (taskToEdit.slo.ttft) {
          setEnableTTFT(true);
          setSloTtftThreshold(taskToEdit.slo.ttft.threshold?.toString() || '1.0');
          setSloTtftWeight(taskToEdit.slo.ttft.weight?.toString() || '2.0');
        }

        // TPOT
        if (taskToEdit.slo.tpot) {
          setEnableTPOT(true);
          setSloTpotThreshold(taskToEdit.slo.tpot.threshold?.toString() || '0.05');
          setSloTpotWeight(taskToEdit.slo.tpot.weight?.toString() || '2.0');
        }

        // Steepness
        if (taskToEdit.slo.steepness !== undefined) {
          setSloSteepness(taskToEdit.slo.steepness.toString());
        }
      }
    }
  }, [taskToEdit]);

  // Basic info
  const [taskName, setTaskName] = useState('');
  const [description, setDescription] = useState('');
  const [deploymentMode, setDeploymentMode] = useState('docker');
  const [baseRuntime, setBaseRuntime] = useState('sglang');
  const [runtimeImageTag, setRuntimeImageTag] = useState('');

  // YAML Import state
  const [isDragging, setIsDragging] = useState(false);
  const [importedFileName, setImportedFileName] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Model config
  const [modelIdOrPath, setModelIdOrPath] = useState('');
  const [modelNamespace, setModelNamespace] = useState('autotuner');

  // Parameters (dynamic list)
  const [parameters, setParameters] = useState<ParamField[]>([]);
  const [usePresets, setUsePresets] = useState(false);

  // Optimization
  const [strategy, setStrategy] = useState('grid_search');
  const [objective, setObjective] = useState('minimize_latency');
  const [maxIterations, setMaxIterations] = useState(2);
  const [timeoutPerIteration, setTimeoutPerIteration] = useState(600);

  // Benchmark
  const [benchmarkTask, setBenchmarkTask] = useState('text-to-text');
  const [benchmarkModelName, setBenchmarkModelName] = useState('');
  const [modelTokenizer, setModelTokenizer] = useState('');
  const [datasetUrl, setDatasetUrl] = useState('');
  const [datasetDeduplicate, setDatasetDeduplicate] = useState(true);
  const [trafficScenarios, setTrafficScenarios] = useState('D(100,100)');
  const [numConcurrency, setNumConcurrency] = useState('1, 4');
  const [maxTimePerIteration, setMaxTimePerIteration] = useState(10);
  const [maxRequestsPerIteration, setMaxRequestsPerIteration] = useState(50);
  const [temperature, setTemperature] = useState('0.0');

  // SLO Configuration
  const [enableSLO, setEnableSLO] = useState(false);

  // Individual metric enable flags
  const [enableP50, setEnableP50] = useState(false);
  const [enableP90, setEnableP90] = useState(false);
  const [enableP99, setEnableP99] = useState(false);
  const [enableTTFT, setEnableTTFT] = useState(false);
  const [enableTPOT, setEnableTPOT] = useState(false);

  // P50 configuration
  const [sloP50Threshold, setSloP50Threshold] = useState('2.0');
  const [sloP50Weight, setSloP50Weight] = useState('1.0');

  // P90 configuration
  const [sloP90Threshold, setSloP90Threshold] = useState('5.0');
  const [sloP90Weight, setSloP90Weight] = useState('2.0');
  const [sloP90HardFail, setSloP90HardFail] = useState(true);
  const [sloP90FailRatio, setSloP90FailRatio] = useState('0.2');

  // P99 configuration
  const [sloP99Threshold, setSloP99Threshold] = useState('10.0');
  const [sloP99Weight, setSloP99Weight] = useState('3.0');
  const [sloP99HardFail, setSloP99HardFail] = useState(true);
  const [sloP99FailRatio, setSloP99FailRatio] = useState('0.5');

  // TTFT configuration
  const [sloTtftThreshold, setSloTtftThreshold] = useState('1.0');
  const [sloTtftWeight, setSloTtftWeight] = useState('2.0');

  // TPOT configuration
  const [sloTpotThreshold, setSloTpotThreshold] = useState('0.05');
  const [sloTpotWeight, setSloTpotWeight] = useState('2.0');

  // Steepness
  const [sloSteepness, setSloSteepness] = useState('0.1');

  // Quantization Configuration
  const [quantConfig, setQuantConfig] = useState<QuantizationConfig>({});
  const [parallelConfig, setParallelConfig] = useState<ParallelConfig>({});

  // Auto-update benchmarkModelName when modelIdOrPath changes
  useEffect(() => {
    if (modelIdOrPath && !benchmarkModelName) {
      const derivedName = modelIdOrPath.split('/').pop() || modelIdOrPath;
      setBenchmarkModelName(derivedName);
    }
  }, [modelIdOrPath, benchmarkModelName]);

  // Auto-update modelTokenizer when modelIdOrPath changes
  useEffect(() => {
    if (modelIdOrPath && !modelTokenizer) {
      setModelTokenizer(modelIdOrPath);
    }
  }, [modelIdOrPath, modelTokenizer]);

  const createTaskMutation = useMutation({
    mutationFn: async (data: TaskFormData) => {
      if (originalTask) {
        // Update existing task
        return await apiClient.updateTask(originalTask.id, data);
      } else {
        // Create new task
        return await apiClient.createTask(data);
      }
    },
    onSuccess: (response) => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
      toast.success(`Task "${response.task_name}" ${originalTask ? 'updated' : 'created'} successfully`);
      navigateTo('tasks');
    },
    onError: (error: any) => {
      const message = error.response?.data?.detail || `Failed to ${originalTask ? 'update' : 'create'} task`;
      toast.error(message);
    },
  });

  const addParameter = () => {
    setParameters([...parameters, { name: '', values: '' }]);
  };

  const removeParameter = (index: number) => {
    setParameters(parameters.filter((_, i) => i !== index));
  };

  const updateParameter = (index: number, field: 'name' | 'values', value: string) => {
    const newParams = [...parameters];
    newParams[index][field] = value;
    setParameters(newParams);
  };

  const handlePresetParametersChange = (presetParams: Record<string, any[]>) => {
    // Convert preset parameters to ParamField format
    const paramFields: ParamField[] = Object.entries(presetParams).map(([name, values]) => ({
      name,
      values: values.join(', ')
    }));

    setParameters(paramFields);
  };

  const parseNumberArray = (str: string): number[] => {
    return str
      .split(',')
      .map((s) => parseFloat(s.trim()))
      .filter((n) => !isNaN(n));
  };

  const parseParameterArray = (str: string): any[] => {
    // Parse comma-separated values, handling numbers, strings, and booleans
    return str
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean)
      .map((val) => {
        // Try to parse as number first
        const num = parseFloat(val);
        if (!isNaN(num)) {
          return num;
        }
        // Parse boolean
        if (val.toLowerCase() === 'true') {
          return true;
        }
        if (val.toLowerCase() === 'false') {
          return false;
        }
        // Keep as string
        return val;
      });
  };

  // YAML Import handlers
  const handleYAMLImport = async (file: File) => {
    try {
      const text = await file.text();
      const config = yaml.load(text) as any;

      // Load the configuration into form
      if (config.task_name) setTaskName(config.task_name);
      if (config.description) setDescription(config.description);
      if (config.deployment_mode) setDeploymentMode(config.deployment_mode);
      if (config.base_runtime) setBaseRuntime(config.base_runtime);
      if (config.runtime_image_tag) setRuntimeImageTag(config.runtime_image_tag);

      // Model config
      if (config.model) {
        if (config.model.id_or_path) setModelIdOrPath(config.model.id_or_path);
        if (config.model.namespace) setModelNamespace(config.model.namespace);
      }

      // Parameters
      if (config.parameters) {
        const params: ParamField[] = [];
        for (const [key, value] of Object.entries(config.parameters)) {
          if (Array.isArray(value)) {
            params.push({ name: key, values: value.join(', ') });
          }
        }
        if (params.length > 0) setParameters(params);
      }

      // Optimization
      if (config.optimization) {
        if (config.optimization.strategy) setStrategy(config.optimization.strategy);
        if (config.optimization.objective) setObjective(config.optimization.objective);
        if (config.optimization.max_iterations) setMaxIterations(config.optimization.max_iterations);
        if (config.optimization.timeout_per_iteration) setTimeoutPerIteration(config.optimization.timeout_per_iteration);
      }

      // Benchmark
      if (config.benchmark) {
        if (config.benchmark.task) setBenchmarkTask(config.benchmark.task);
        if (config.benchmark.model_name) setBenchmarkModelName(config.benchmark.model_name);
        if (config.benchmark.model_tokenizer) setModelTokenizer(config.benchmark.model_tokenizer);
        if (config.benchmark.traffic_scenarios) setTrafficScenarios(config.benchmark.traffic_scenarios.join(', '));
        if (config.benchmark.num_concurrency) setNumConcurrency(config.benchmark.num_concurrency.join(', '));
        if (config.benchmark.max_time_per_iteration) setMaxTimePerIteration(config.benchmark.max_time_per_iteration);
        if (config.benchmark.max_requests_per_iteration) setMaxRequestsPerIteration(config.benchmark.max_requests_per_iteration);
        if (config.benchmark.additional_params?.temperature !== undefined) {
          setTemperature(config.benchmark.additional_params.temperature.toString());
        }
        // Dataset URL
        if (config.benchmark.dataset_url) {
          setDatasetUrl(config.benchmark.dataset_url);
        }
        if (config.benchmark.dataset_deduplicate !== undefined) {
          setDatasetDeduplicate(config.benchmark.dataset_deduplicate);
        }
      }

      // SLO Configuration
      if (config.slo) {
        setEnableSLO(true);

        // Latency metrics
        if (config.slo.latency) {
          if (config.slo.latency.p50) {
            setEnableP50(true);
            setSloP50Threshold(config.slo.latency.p50.threshold?.toString() || '2.0');
            setSloP50Weight(config.slo.latency.p50.weight?.toString() || '1.0');
          }

          if (config.slo.latency.p90) {
            setEnableP90(true);
            setSloP90Threshold(config.slo.latency.p90.threshold?.toString() || '5.0');
            setSloP90Weight(config.slo.latency.p90.weight?.toString() || '2.0');
            if (config.slo.latency.p90.hard_fail !== undefined) setSloP90HardFail(config.slo.latency.p90.hard_fail);
            if (config.slo.latency.p90.fail_ratio !== undefined) setSloP90FailRatio(config.slo.latency.p90.fail_ratio.toString());
          }

          if (config.slo.latency.p99) {
            setEnableP99(true);
            setSloP99Threshold(config.slo.latency.p99.threshold?.toString() || '10.0');
            setSloP99Weight(config.slo.latency.p99.weight?.toString() || '3.0');
            if (config.slo.latency.p99.hard_fail !== undefined) setSloP99HardFail(config.slo.latency.p99.hard_fail);
            if (config.slo.latency.p99.fail_ratio !== undefined) setSloP99FailRatio(config.slo.latency.p99.fail_ratio.toString());
          }
        }

        // TTFT
        if (config.slo.ttft) {
          setEnableTTFT(true);
          setSloTtftThreshold(config.slo.ttft.threshold?.toString() || '1.0');
          setSloTtftWeight(config.slo.ttft.weight?.toString() || '2.0');
        }

        // TPOT
        if (config.slo.tpot) {
          setEnableTPOT(true);
          setSloTpotThreshold(config.slo.tpot.threshold?.toString() || '0.05');
          setSloTpotWeight(config.slo.tpot.weight?.toString() || '2.0');
        }

        // Steepness
        if (config.slo.steepness !== undefined) {
          setSloSteepness(config.slo.steepness.toString());
        }
      }

      // Quantization and Parallel configs
      if (config.quant_config) setQuantConfig(config.quant_config);
      if (config.parallel_config) setParallelConfig(config.parallel_config);

      // Set imported state
      setImportedFileName(file.name);
      toast.success(`Configuration loaded from ${file.name}`);
    } catch (error) {
      console.error('Error parsing YAML:', error);
      toast.error('Failed to parse YAML file. Please check the file format.');
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleYAMLImport(file);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (file) {
      if (file.name.endsWith('.yaml') || file.name.endsWith('.yml')) {
        handleYAMLImport(file);
      } else {
        toast.error('Please drop a YAML file (.yaml or .yml)');
      }
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Parse parameters
    const parsedParams: Record<string, any[]> = {};
    for (const param of parameters) {
      if (param.name && param.values) {
        parsedParams[param.name] = parseParameterArray(param.values);
      }
    }

    // Parse traffic scenarios - split by comma but respect parentheses
    // D(100,100), D(200,200) should become ["D(100,100)", "D(200,200)"]
    const trafficScenariosList = trafficScenarios
      .split(/,\s*(?![^()]*\))/)  // Split on comma not inside parentheses
      .map((s) => s.trim())
      .filter(Boolean);

    const formData: TaskFormData = {
      task_name: taskName,
      description,
      deployment_mode: deploymentMode,
      base_runtime: baseRuntime,
      ...(runtimeImageTag && { runtime_image_tag: runtimeImageTag }),
      model: {
        id_or_path: modelIdOrPath,
        namespace: modelNamespace,
      },
      parameters: parsedParams,
      // Include quantization config if any field is set
      ...(Object.keys(quantConfig).length > 0 && { quant_config: quantConfig }),
      // Include parallel config if any field is set
      ...(Object.keys(parallelConfig).length > 0 && { parallel_config: parallelConfig }),
      optimization: {
        strategy,
        objective,
        max_iterations: maxIterations,
        timeout_per_iteration: timeoutPerIteration,
      },
      benchmark: {
        task: benchmarkTask,
        model_name: benchmarkModelName || modelIdOrPath.split('/').pop() || modelIdOrPath, // Use editable field or fallback
        model_tokenizer: modelTokenizer || modelIdOrPath, // Auto-fill with full id_or_path if empty
        traffic_scenarios: trafficScenariosList,
        num_concurrency: parseNumberArray(numConcurrency),
        max_time_per_iteration: maxTimePerIteration,
        max_requests_per_iteration: maxRequestsPerIteration,
        additional_params: {
          temperature: parseFloat(temperature),
        },
        // Include dataset URL if provided
        ...(datasetUrl && { dataset_url: datasetUrl }),
        ...(datasetUrl && { dataset_deduplicate: datasetDeduplicate }),
      },
      // Include SLO configuration if enabled
      ...(enableSLO && {
        slo: (() => {
          const slo: any = {};

          // Only include latency section if at least one metric is enabled
          const latency: any = {};

          if (enableP50 && sloP50Threshold) {
            latency.p50 = {
              threshold: parseFloat(sloP50Threshold),
              ...(sloP50Weight && { weight: parseFloat(sloP50Weight) }),
              hard_fail: false,
            };
          }

          if (enableP90 && sloP90Threshold) {
            latency.p90 = {
              threshold: parseFloat(sloP90Threshold),
              ...(sloP90Weight && { weight: parseFloat(sloP90Weight) }),
              hard_fail: sloP90HardFail,
              ...(sloP90HardFail && sloP90FailRatio && { fail_ratio: parseFloat(sloP90FailRatio) }),
            };
          }

          if (enableP99 && sloP99Threshold) {
            latency.p99 = {
              threshold: parseFloat(sloP99Threshold),
              ...(sloP99Weight && { weight: parseFloat(sloP99Weight) }),
              hard_fail: sloP99HardFail,
              ...(sloP99HardFail && sloP99FailRatio && { fail_ratio: parseFloat(sloP99FailRatio) }),
            };
          }

          if (Object.keys(latency).length > 0) {
            slo.latency = latency;
          }

          // Only include TTFT if enabled and threshold is configured
          if (enableTTFT && sloTtftThreshold) {
            slo.ttft = {
              threshold: parseFloat(sloTtftThreshold),
              ...(sloTtftWeight && { weight: parseFloat(sloTtftWeight) }),
              hard_fail: false,
            };
          }

          // Only include TPOT if enabled and threshold is configured
          if (enableTPOT && sloTpotThreshold) {
            slo.tpot = {
              threshold: parseFloat(sloTpotThreshold),
              ...(sloTpotWeight && { weight: parseFloat(sloTpotWeight) }),
              hard_fail: false,
            };
          }

          // Only include steepness if specified
          if (sloSteepness) {
            slo.steepness = parseFloat(sloSteepness);
          }

          return slo;
        })(),
      }),
    };

    createTaskMutation.mutate(formData);
  };

  return (
    <div
      className="max-w-4xl mx-auto relative"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Full-page drag overlay */}
      {isDragging && (
        <div className="fixed inset-0 z-50 bg-blue-500 bg-opacity-10 pointer-events-none flex items-center justify-center">
          <div className="bg-white rounded-lg shadow-2xl p-8 border-4 border-dashed border-blue-500">
            <div className="flex flex-col items-center">
              <svg
                className="w-20 h-20 mb-4 text-blue-500 animate-bounce"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
              <h3 className="text-2xl font-bold text-blue-900 mb-2">
                Drop YAML file anywhere
              </h3>
              <p className="text-blue-700">
                Release to import task configuration
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900">{originalTask ? "Edit Task" : "Create New Task"}</h1>
        <p className="mt-2 text-gray-600">Configure a new autotuning task</p>
      </div>

      {/* YAML Import Section - Always minimal, no resizing */}
      <div className="mb-6">
        <div
          className={`flex items-center gap-3 rounded-lg px-4 py-3 transition-colors cursor-pointer ${
            importedFileName
              ? 'bg-green-50 border-2 border-green-500 hover:bg-green-100'
              : 'bg-gray-50 border border-gray-200 hover:bg-gray-100'
          }`}
          onClick={() => fileInputRef.current?.click()}
        >
          {importedFileName ? (
            // Success state - checkmark icon
            <svg
              className="w-5 h-5 text-green-600"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          ) : (
            // Default state - upload icon
            <svg
              className="w-5 h-5 text-gray-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              />
            </svg>
          )}
          <span className={`text-sm font-medium ${importedFileName ? 'text-green-900' : 'text-gray-700'}`}>
            {importedFileName ? `Imported: ${importedFileName}` : 'Import YAML Configuration'}
          </span>
          {!importedFileName && (
            <span className="text-xs text-gray-500">
              Drag file anywhere or click to browse
            </span>
          )}
          {importedFileName && (
            <span className="text-xs text-green-600 ml-auto">
              Click to import another file
            </span>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept=".yaml,.yml"
            onChange={handleFileSelect}
            className="hidden"
          />
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-8">
        {/* Basic Information */}
        <div className="bg-white shadow-sm rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Basic Information</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Task Name *
              </label>
              <input
                type="text"
                value={taskName}
                onChange={(e) => setTaskName(e.target.value)}
                required
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="docker-simple-tune"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Deployment Mode *
              </label>
              <select
                value={deploymentMode}
                onChange={(e) => setDeploymentMode(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="docker">Docker</option>
                <option value="ome">OME (Kubernetes)</option>
              </select>
            </div>

            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Description
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={2}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Describe the purpose of this autotuning task"
              />
            </div>
          </div>
        </div>

        {/* Runtime Configuration */}
        <div className="bg-white shadow-sm rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Runtime Configuration</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Base Runtime *
              </label>
              <select
                value={baseRuntime}
                onChange={(e) => setBaseRuntime(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="sglang">SGLang</option>
                <option value="vllm">vLLM</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Runtime Image Tag
              </label>
              <input
                type="text"
                value={runtimeImageTag}
                onChange={(e) => setRuntimeImageTag(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="v0.5.2-cu126 (optional)"
              />
              <p className="text-sm text-gray-500 mt-1">
                Docker image tag for the runtime (Docker mode only)
              </p>
            </div>
          </div>
        </div>

        {/* Model Configuration */}
        <div className="bg-white shadow-sm rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Model Configuration</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Model ID or Path *
              </label>
              <input
                type="text"
                value={modelIdOrPath}
                onChange={(e) => setModelIdOrPath(e.target.value)}
                required
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="llama-3-2-1b-instruct or meta-llama/Llama-3.2-1B-Instruct"
              />
              <p className="text-sm text-gray-500 mt-1">
                Local model directory name or HuggingFace model ID
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Model Namespace *
              </label>
              <input
                type="text"
                value={modelNamespace}
                onChange={(e) => setModelNamespace(e.target.value)}
                required
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="autotuner"
              />
            </div>
          </div>
        </div>

        {/* Quantization Configuration */}
        <div className="bg-white shadow-sm rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Quantization Configuration (Optional)</h2>
          <p className="text-sm text-gray-600 mb-4">
            Configure runtime quantization for fine-grained control over model precision.
          </p>
          <QuantizationConfigForm
            value={quantConfig}
            onChange={setQuantConfig}
            baseRuntime={baseRuntime}
          />
        </div>

        {/* Parallel Configuration */}
        <div className="bg-white shadow-sm rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Parallel Execution Configuration (Optional)</h2>
          <p className="text-sm text-gray-600 mb-4">
            Configure multi-GPU parallelism (TP, PP, DP, CP) for distributed inference.
          </p>
          <ParallelConfigForm
            value={parallelConfig}
            onChange={setParallelConfig}
            baseRuntime={baseRuntime}
          />
        </div>

        {/* Parameters */}
        <div className="bg-white shadow-sm rounded-lg p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold">Parameters to Tune</h2>
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={usePresets}
                  onChange={(e) => setUsePresets(e.target.checked)}
                  className="rounded"
                />
                <span className="text-gray-700">Use Parameter Presets</span>
              </label>
              <button
                type="button"
                onClick={addParameter}
                className="px-3 py-1 bg-blue-600 text-white rounded-md hover:bg-blue-700 text-sm"
              >
                Add Parameter
              </button>
            </div>
          </div>

          {/* Preset Selector */}
          {usePresets && (
            <div className="mb-6">
              <PresetSelector
                onParametersChange={handlePresetParametersChange}
                className="mb-4"
              />
            </div>
          )}

          {parameters.length === 0 ? (
            <div className="p-4 bg-gray-50 rounded-md border border-gray-200">
              <p className="text-sm text-gray-600">
                No parameters configured. The model will use default runtime parameters.
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {parameters.map((param, index) => (
              <div key={index} className="flex gap-3 items-start">
                <div className="flex-1">
                  <input
                    type="text"
                    value={param.name}
                    onChange={(e) => updateParameter(index, 'name', e.target.value)}
                    placeholder="Parameter name (e.g., tp-size)"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div className="flex-1">
                  <input
                    type="text"
                    value={param.values}
                    onChange={(e) => updateParameter(index, 'values', e.target.value)}
                    placeholder="Values (e.g., 1, 2, 4)"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <button
                  type="button"
                  onClick={() => removeParameter(index)}
                  className="px-3 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
                >
                  Remove
                </button>
              </div>
            ))}
            </div>
          )}
          <p className="text-sm text-gray-500 mt-2">
            {usePresets
              ? "Parameters from presets are pre-filled below. You can still edit them manually."
              : "Enter parameter values as comma-separated numbers (e.g., 0.7, 0.8, 0.9)"}
          </p>
        </div>

        {/* Optimization Settings */}
        <div className="bg-white shadow-sm rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Optimization Settings</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Strategy
              </label>
              <select
                value={strategy}
                onChange={(e) => setStrategy(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="grid_search">Grid Search</option>
                <option value="random_search">Random Search</option>
                <option value="bayesian">Bayesian Optimization</option>
              </select>
              {strategy === 'bayesian' && (
                <div className="mt-2 p-3 bg-blue-50 border border-blue-200 rounded-md">
                  <p className="text-xs text-blue-900 font-semibold mb-1">Bayesian Optimization (Intelligent Search)</p>
                  <p className="text-xs text-blue-800">
                    Uses machine learning to intelligently explore the parameter space. Typically finds optimal configurations in 20-30 experiments (vs 100+ for grid search).
                    <span className="block mt-1 font-medium">Best for: Large parameter spaces with expensive experiments.</span>
                  </p>
                </div>
              )}
              {strategy === 'grid_search' && (
                <p className="text-xs text-gray-500 mt-1">
                  Exhaustively tests all parameter combinations. Reliable but can be slow for large spaces.
                </p>
              )}
              {strategy === 'random_search' && (
                <p className="text-xs text-gray-500 mt-1">
                  Randomly samples parameter combinations. Good for quick exploration.
                </p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Objective
              </label>
              <select
                value={objective}
                onChange={(e) => setObjective(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="minimize_latency">Minimize Latency</option>
                <option value="maximize_throughput">Maximize Throughput</option>
                <option value="balanced">Balanced</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Max Iterations
              </label>
              <input
                type="number"
                value={maxIterations}
                onChange={(e) => setMaxIterations(parseInt(e.target.value))}
                min="1"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Timeout Per Iteration (seconds)
              </label>
              <input
                type="number"
                value={timeoutPerIteration}
                onChange={(e) => setTimeoutPerIteration(parseInt(e.target.value))}
                min="1"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
        </div>

        {/* Benchmark Configuration */}
        <div className="bg-white shadow-sm rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Benchmark Configuration</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Benchmark Task
              </label>
              <input
                type="text"
                value={benchmarkTask}
                onChange={(e) => setBenchmarkTask(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="text-to-text"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Model Name
              </label>
              <input
                type="text"
                value={benchmarkModelName}
                onChange={(e) => setBenchmarkModelName(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Auto-filled from Model ID/Path"
              />
              <p className="text-sm text-gray-500 mt-1">
                Display name for benchmark results (auto-filled but editable)
              </p>
            </div>

            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Model Tokenizer (HuggingFace ID)
              </label>
              <input
                type="text"
                value={modelTokenizer}
                onChange={(e) => setModelTokenizer(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Leave empty to auto-fill with Model ID/Path"
              />
              <p className="text-sm text-gray-500 mt-1">
                HuggingFace model ID for tokenizer. Linked to Model ID/Path above (auto-fills if empty).
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Traffic Scenarios
              </label>
              <input
                type="text"
                value={trafficScenarios}
                onChange={(e) => setTrafficScenarios(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="D(100,100)"
              />
              <p className="text-sm text-gray-500 mt-1">
                Comma-separated list (e.g., D(100,100), D(200,200))
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Concurrency Levels
              </label>
              <input
                type="text"
                value={numConcurrency}
                onChange={(e) => setNumConcurrency(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="1, 4, 8"
              />
              <p className="text-sm text-gray-500 mt-1">
                Comma-separated numbers
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Max Time Per Iteration (seconds)
              </label>
              <input
                type="number"
                value={maxTimePerIteration}
                onChange={(e) => setMaxTimePerIteration(parseInt(e.target.value))}
                min="1"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Max Requests Per Iteration
              </label>
              <input
                type="number"
                value={maxRequestsPerIteration}
                onChange={(e) => setMaxRequestsPerIteration(parseInt(e.target.value))}
                min="1"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Temperature
              </label>
              <input
                type="text"
                value={temperature}
                onChange={(e) => setTemperature(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="0.0"
              />
            </div>

            {/* Dataset URL Section */}
            <div className="md:col-span-2 border-t pt-4 mt-2">
              <h3 className="text-sm font-medium text-gray-900 mb-3">Custom Dataset (Optional)</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Dataset URL
                  </label>
                  <input
                    type="text"
                    value={datasetUrl}
                    onChange={(e) => setDatasetUrl(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="https://example.com/api_logs.csv.gz"
                  />
                  <p className="text-sm text-gray-500 mt-1">
                    URL to a dataset file (CSV, JSONL, or compressed archive). Supports API logs with request_body column.
                  </p>
                </div>
                <div>
                  <label className="flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={datasetDeduplicate}
                      onChange={(e) => setDatasetDeduplicate(e.target.checked)}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                    <span className="ml-2 text-sm text-gray-700">Deduplicate prompts</span>
                  </label>
                  <p className="text-xs text-gray-500 mt-1 ml-6">
                    Remove duplicate prompts from the dataset
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* SLO Configuration (Optional) */}
        <div className="bg-white shadow-sm rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-xl font-semibold">SLO Configuration (Optional)</h2>
              <p className="text-sm text-gray-500 mt-1">Define Service Level Objectives with exponential penalties</p>
            </div>
            <label className="flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={enableSLO}
                onChange={(e) => setEnableSLO(e.target.checked)}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <span className="ml-2 text-sm font-medium text-gray-700">Enable SLO</span>
            </label>
          </div>

          {enableSLO && (
            <div className="space-y-6 border-t pt-4">
              {/* TTFT */}
              <div className="border-b pb-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-medium text-gray-900">Time to First Token (Soft Penalty)</h3>
                  <label className="flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={enableTTFT}
                      onChange={(e) => setEnableTTFT(e.target.checked)}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                    <span className="ml-2 text-xs text-gray-600">Enable</span>
                  </label>
                </div>
                {enableTTFT && (
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs text-gray-600 mb-1">Threshold (seconds)</label>
                      <input
                        type="text"
                        value={sloTtftThreshold}
                        onChange={(e) => setSloTtftThreshold(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="1.0"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-600 mb-1">Penalty Weight</label>
                      <input
                        type="text"
                        value={sloTtftWeight}
                        onChange={(e) => setSloTtftWeight(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="2.0"
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* TPOT */}
              <div className="border-b pb-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-medium text-gray-900">Time Per Output Token (Soft Penalty)</h3>
                  <label className="flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={enableTPOT}
                      onChange={(e) => setEnableTPOT(e.target.checked)}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                    <span className="ml-2 text-xs text-gray-600">Enable</span>
                  </label>
                </div>
                {enableTPOT && (
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs text-gray-600 mb-1">Threshold (seconds)</label>
                      <input
                        type="text"
                        value={sloTpotThreshold}
                        onChange={(e) => setSloTpotThreshold(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="0.05"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-600 mb-1">Penalty Weight</label>
                      <input
                        type="text"
                        value={sloTpotWeight}
                        onChange={(e) => setSloTpotWeight(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="2.0"
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* P50 Latency */}
              <div className="border-b pb-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-medium text-gray-900">P50 Latency (Soft Penalty)</h3>
                  <label className="flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={enableP50}
                      onChange={(e) => setEnableP50(e.target.checked)}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                    <span className="ml-2 text-xs text-gray-600">Enable</span>
                  </label>
                </div>
                {enableP50 && (
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs text-gray-600 mb-1">Threshold (seconds)</label>
                      <input
                        type="text"
                        value={sloP50Threshold}
                        onChange={(e) => setSloP50Threshold(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="2.0"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-600 mb-1">Penalty Weight</label>
                      <input
                        type="text"
                        value={sloP50Weight}
                        onChange={(e) => setSloP50Weight(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="1.0"
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* P90 Latency */}
              <div className="border-b pb-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-medium text-gray-900">P90 Latency (Tiered Enforcement)</h3>
                  <label className="flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={enableP90}
                      onChange={(e) => setEnableP90(e.target.checked)}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                    <span className="ml-2 text-xs text-gray-600">Enable</span>
                  </label>
                </div>
                {enableP90 && (
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs text-gray-600 mb-1">Threshold (seconds)</label>
                      <input
                        type="text"
                        value={sloP90Threshold}
                        onChange={(e) => setSloP90Threshold(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="5.0"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-600 mb-1">Penalty Weight</label>
                      <input
                        type="text"
                        value={sloP90Weight}
                        onChange={(e) => setSloP90Weight(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="2.0"
                      />
                    </div>
                    <div className="col-span-2">
                      <label className="flex items-center cursor-pointer">
                        <input
                          type="checkbox"
                          checked={sloP90HardFail}
                          onChange={(e) => setSloP90HardFail(e.target.checked)}
                          className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                        />
                        <span className="ml-2 text-xs text-gray-700">Enable hard fail above ratio:</span>
                        <input
                          type="text"
                          value={sloP90FailRatio}
                          onChange={(e) => setSloP90FailRatio(e.target.value)}
                          disabled={!sloP90HardFail}
                          className="ml-2 w-20 px-2 py-1 border border-gray-300 rounded text-xs focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
                          placeholder="0.2"
                        />
                        <span className="ml-1 text-xs text-gray-500">(20% over = fail)</span>
                      </label>
                    </div>
                  </div>
                )}
              </div>

              {/* P99 Latency */}
              <div className="border-b pb-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-medium text-gray-900">P99 Latency (Tiered Enforcement)</h3>
                  <label className="flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={enableP99}
                      onChange={(e) => setEnableP99(e.target.checked)}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                    <span className="ml-2 text-xs text-gray-600">Enable</span>
                  </label>
                </div>
                {enableP99 && (
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs text-gray-600 mb-1">Threshold (seconds)</label>
                      <input
                        type="text"
                        value={sloP99Threshold}
                        onChange={(e) => setSloP99Threshold(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="10.0"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-600 mb-1">Penalty Weight</label>
                      <input
                        type="text"
                        value={sloP99Weight}
                        onChange={(e) => setSloP99Weight(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="3.0"
                      />
                    </div>
                    <div className="col-span-2">
                      <label className="flex items-center cursor-pointer">
                        <input
                          type="checkbox"
                          checked={sloP99HardFail}
                          onChange={(e) => setSloP99HardFail(e.target.checked)}
                          className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                        />
                        <span className="ml-2 text-xs text-gray-700">Enable hard fail above ratio:</span>
                        <input
                          type="text"
                          value={sloP99FailRatio}
                          onChange={(e) => setSloP99FailRatio(e.target.value)}
                          disabled={!sloP99HardFail}
                          className="ml-2 w-20 px-2 py-1 border border-gray-300 rounded text-xs focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
                          placeholder="0.5"
                        />
                        <span className="ml-1 text-xs text-gray-500">(50% over = fail)</span>
                      </label>
                    </div>
                  </div>
                )}
              </div>

              {/* Steepness */}
              <div>
                <h3 className="text-sm font-medium text-gray-900 mb-3">Exponential Curve Steepness</h3>
                <div className="grid grid-cols-1 gap-4">
                  <div>
                    <label className="block text-xs text-gray-600 mb-1">
                      Steepness Parameter (lower = steeper penalty curve)
                    </label>
                    <input
                      type="text"
                      value={sloSteepness}
                      onChange={(e) => setSloSteepness(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                      placeholder="0.1"
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      Default: 0.1 (recommended). Lower values create steeper penalties near SLO boundaries.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex gap-4 justify-end">
          <button
            type="button"
            onClick={() => navigateTo('tasks')}
            className="px-6 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={createTaskMutation.isPending}
            className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-blue-400"
          >
            {createTaskMutation.isPending ? (originalTask ? 'Saving...' : 'Creating...') : (originalTask ? 'Save Changes' : 'Create Task')}
          </button>
        </div>
      </form>
    </div>
  );
}
