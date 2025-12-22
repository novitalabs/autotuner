import React, { useState, useEffect, useMemo } from 'react';
import { ParallelConfig } from '../types/api';

interface ParallelConfigFormProps {
	value: ParallelConfig;
	onChange: (config: ParallelConfig) => void;
	baseRuntime?: string;
}

// Parallel presets per engine
const PARALLEL_PRESETS: Record<string, Array<{ id: string; name: string; description: string; gpus: number }>> = {
	vllm: [
		{ id: 'single-gpu', name: 'Single GPU', description: 'No parallelism (1 GPU)', gpus: 1 },
		{ id: 'high-throughput', name: 'High Throughput', description: 'Data parallel for maximum throughput (8 GPUs)', gpus: 8 },
		{ id: 'large-model-tp', name: 'Large Model TP', description: 'Tensor parallel for large models (8 GPUs)', gpus: 8 },
		{ id: 'large-model-tp-pp', name: 'Large Model TP+PP', description: 'TP + PP for very large models (16 GPUs: 8 TP × 2 PP)', gpus: 16 },
		{ id: 'moe-optimized', name: 'MoE Optimized', description: 'MoE with expert parallelism (16 GPUs: 2 TP × 8 DP)', gpus: 16 },
		{ id: 'long-context', name: 'Long Context', description: 'DCP for long context (16 GPUs: 4 TP × 4 DCP)', gpus: 16 },
		{ id: 'balanced', name: 'Balanced', description: 'Balanced TP and DP (8 GPUs: 2 TP × 4 DP)', gpus: 8 },
	],
	sglang: [
		{ id: 'single-gpu', name: 'Single GPU', description: 'No parallelism (1 GPU)', gpus: 1 },
		{ id: 'high-throughput', name: 'High Throughput', description: 'Data parallel for maximum throughput (8 GPUs)', gpus: 8 },
		{ id: 'large-model-tp', name: 'Large Model TP', description: 'Tensor parallel for large models (8 GPUs)', gpus: 8 },
		{ id: 'large-model-tp-pp', name: 'Large Model TP+PP', description: 'TP + PP for very large models (16 GPUs: 8 TP × 2 PP)', gpus: 16 },
		{ id: 'moe-optimized', name: 'MoE Optimized', description: 'Automatic expert distribution (16 GPUs: 2 TP × 8 DP)', gpus: 16 },
		{ id: 'balanced', name: 'Balanced', description: 'Balanced TP and DP (8 GPUs: 2 TP × 4 DP)', gpus: 8 },
	],
	'tensorrt-llm': [
		{ id: 'single-gpu', name: 'Single GPU', description: 'No parallelism (1 GPU, build-time)', gpus: 1 },
		{ id: 'large-model-tp', name: 'Large Model TP', description: 'Tensor parallel (8 GPUs, build-time)', gpus: 8 },
		{ id: 'large-model-tp-pp', name: 'Large Model TP+PP', description: 'TP + PP (16 GPUs: 8 TP × 2 PP, build-time)', gpus: 16 },
		{ id: 'moe-optimized', name: 'MoE Optimized', description: 'Explicit EP configuration (16 GPUs, build-time)', gpus: 16 },
		{ id: 'long-context', name: 'Long Context', description: 'Context parallel (16 GPUs: 8 TP × 2 CP, build-time)', gpus: 16 },
	],
};

// Allowed values per parameter per engine
const PARAM_VALUES: Record<string, Record<string, number[]>> = {
	vllm: {
		tp: [1, 2, 4, 8, 16],
		pp: [1, 2, 4, 8],
		dp: [1, 2, 4, 8, 16],
		dcp: [1, 2, 4, 8],
	},
	sglang: {
		tp: [1, 2, 4, 8, 16],
		pp: [1, 2, 4],
		dp: [1, 2, 4, 8, 16],
	},
	'tensorrt-llm': {
		tp: [1, 2, 4, 8, 16],
		pp: [1, 2, 4],
		cp: [1, 2, 4, 8],
	},
};

// Preset configurations (matches backend PARALLEL_PRESETS)
const PRESET_CONFIGS: Record<string, Record<string, any>> = {
	vllm: {
		'single-gpu': { tp: 1, pp: 1, dp: 1 },
		'high-throughput': { tp: 1, pp: 1, dp: 8 },
		'large-model-tp': { tp: 8, pp: 1, dp: 1 },
		'large-model-tp-pp': { tp: 8, pp: 2, dp: 1 },
		'moe-optimized': { tp: 2, pp: 1, dp: 8, enable_expert_parallel: true },
		'long-context': { tp: 4, pp: 1, dp: 1, dcp: 4 },
		'balanced': { tp: 2, pp: 1, dp: 4 },
	},
	sglang: {
		'single-gpu': { tp: 1, pp: 1, dp: 1 },
		'high-throughput': { tp: 1, pp: 1, dp: 8 },
		'large-model-tp': { tp: 8, pp: 1, dp: 1 },
		'large-model-tp-pp': { tp: 8, pp: 2, dp: 1 },
		'moe-optimized': { tp: 2, pp: 1, dp: 8, moe_dense_tp: 2 },
		'balanced': { tp: 2, pp: 1, dp: 4 },
	},
	'tensorrt-llm': {
		'single-gpu': { tp: 1, pp: 1 },
		'large-model-tp': { tp: 8, pp: 1 },
		'large-model-tp-pp': { tp: 8, pp: 2 },
		'moe-optimized': { tp: 4, pp: 1, moe_tp: 2, moe_ep: 8 },
		'long-context': { tp: 8, pp: 1, cp: 2 },
	},
};

// Map parallel config to runtime-specific CLI arguments
function mapToRuntimeArgs(runtime: string, config: Record<string, any>): string[] {
	const args: string[] = [];

	if (runtime === 'vllm') {
		if (config.tp && config.tp !== 1) args.push(`--tensor-parallel-size ${config.tp}`);
		if (config.pp && config.pp !== 1) args.push(`--pipeline-parallel-size ${config.pp}`);
		if (config.dp && config.dp !== 1) args.push(`--data-parallel-size ${config.dp}`);
		if (config.dcp && config.dcp !== 1) args.push(`--decode-context-parallel-size ${config.dcp}`);
		if (config.enable_expert_parallel) args.push('--enable-expert-parallel');
	} else if (runtime === 'sglang') {
		if (config.tp && config.tp !== 1) args.push(`--tp-size ${config.tp}`);
		if (config.pp && config.pp !== 1) args.push(`--pp-size ${config.pp}`);
		if (config.dp && config.dp !== 1) args.push(`--dp-size ${config.dp}`);
		if (config.moe_dense_tp) args.push(`--moe-dense-tp-size ${config.moe_dense_tp}`);
	} else if (runtime === 'tensorrt-llm') {
		if (config.tp) args.push(`tp_size=${config.tp}`);
		if (config.pp) args.push(`pp_size=${config.pp}`);
		if (config.cp && config.cp !== 1) args.push(`cp_size=${config.cp}`);
		if (config.moe_tp) args.push(`moe_tp_size=${config.moe_tp}`);
		if (config.moe_ep) args.push(`moe_ep_size=${config.moe_ep}`);
	}

	return args;
}

// Validate parallel config combination for engine constraints
function isValidCombination(runtime: string, config: Record<string, any>): boolean {
	if (runtime === 'sglang') {
		// SGLang constraint: tp % dp == 0
		const tp = config.tp || 1;
		const dp = config.dp || 1;
		if (tp % dp !== 0) {
			return false;
		}
	} else if (runtime === 'tensorrt-llm') {
		// TensorRT-LLM doesn't support DP
		if (config.dp && config.dp !== 1) {
			return false;
		}
	}

	// All other combinations are valid
	return true;
}

// Deduplicate argument combinations based on their content
function deduplicateCombinations(combinations: string[][]): string[][] {
	const seen = new Set<string>();
	const unique: string[][] = [];

	for (const combo of combinations) {
		// Create a canonical string representation by sorting arguments
		// This handles cases where different configs produce the same CLI args
		const sortedArgs = [...combo].sort();
		const signature = JSON.stringify(sortedArgs);

		if (!seen.has(signature)) {
			seen.add(signature);
			unique.push(combo);
		}
	}

	return unique;
}

// Get all runtime argument combinations
function getAllRuntimeArgCombinations(runtime: string, config: ParallelConfig): { combinations: string[][]; total: number; truncated: boolean; invalidCount: number } {
	const combinations: string[][] = [];
	const maxDisplay = 10;
	let invalidCount = 0;

	// Preset mode
	if (config.presets && config.presets.length > 0) {
		const normalizedRuntime = runtime === 'tensorrtllm' ? 'tensorrt-llm' : runtime;
		const presetConfigs = PRESET_CONFIGS[normalizedRuntime] || {};

		for (const presetName of config.presets) {
			const presetConfig = presetConfigs[presetName];
			if (presetConfig) {
				// Validate preset configuration
				if (!isValidCombination(runtime, presetConfig)) {
					invalidCount++;
					continue;
				}

				const args = mapToRuntimeArgs(runtime, presetConfig);
				if (args.length > 0) {
					combinations.push(args);
				} else {
					combinations.push(['(single GPU, no parallel args)']);
				}
			}
		}

		// Deduplicate preset combinations
		const uniqueCombinations = deduplicateCombinations(combinations);

		return {
			combinations: uniqueCombinations.slice(0, maxDisplay),
			total: uniqueCombinations.length,
			truncated: uniqueCombinations.length > maxDisplay,
			invalidCount
		};
	}

	// Custom mode - expand arrays
	const expandedConfigs: Record<string, any>[] = [{}];

	for (const [key, value] of Object.entries(config)) {
		if (value === undefined || value === null) continue;

		const values = Array.isArray(value) ? value : [value];
		const newConfigs: Record<string, any>[] = [];

		for (const config of expandedConfigs) {
			for (const val of values) {
				newConfigs.push({ ...config, [key]: val });
			}
		}

		expandedConfigs.length = 0;
		expandedConfigs.push(...newConfigs);
	}

	for (const expandedConfig of expandedConfigs) {
		// Validate configuration
		if (!isValidCombination(runtime, expandedConfig)) {
			invalidCount++;
			continue;
		}

		const args = mapToRuntimeArgs(runtime, expandedConfig);
		if (args.length > 0) {
			combinations.push(args);
		}
	}

	// Deduplicate custom mode combinations
	const uniqueCombinations = deduplicateCombinations(combinations);

	return {
		combinations: uniqueCombinations.slice(0, maxDisplay),
		total: uniqueCombinations.length,
		truncated: uniqueCombinations.length > maxDisplay,
		invalidCount
	};
}

export const ParallelConfigForm: React.FC<ParallelConfigFormProps> = ({ value, onChange, baseRuntime = 'sglang' }) => {
	const [configMode, setConfigMode] = useState<'none' | 'preset' | 'custom'>(
		value.presets && value.presets.length > 0 ? 'preset' :
		(value.tp || value.pp || value.dp || value.cp || value.dcp) ? 'custom' :
		'none'
	);

	// Update configMode when value prop changes
	useEffect(() => {
		const detectedMode = value.presets && value.presets.length > 0 ? 'preset' :
			(value.tp || value.pp || value.dp || value.cp || value.dcp) ? 'custom' :
			'none';
		setConfigMode(detectedMode);
	}, [value]);

	const runtimeKey = baseRuntime.toLowerCase().replace(/-/g, '').replace(/_/g, '');
	const normalizedRuntime = runtimeKey === 'tensorrtllm' ? 'tensorrt-llm' : baseRuntime.toLowerCase();
	const presets = PARALLEL_PRESETS[normalizedRuntime] || PARALLEL_PRESETS['sglang'];
	const paramValues = PARAM_VALUES[normalizedRuntime] || PARAM_VALUES['sglang'];

	const handleModeChange = (mode: 'none' | 'preset' | 'custom') => {
		setConfigMode(mode);
		if (mode === 'none') {
			onChange({});
		} else if (mode === 'preset') {
			onChange({ presets: ['single-gpu', 'high-throughput'] });
		} else if (mode === 'custom') {
			onChange({
				tp: 1,
				pp: 1,
				dp: 1
			});
		}
	};

	const handlePresetToggle = (preset: string) => {
		const currentPresets = value.presets || [];
		const newPresets = currentPresets.includes(preset)
			? currentPresets.filter(p => p !== preset)
			: [...currentPresets, preset];
		onChange({ presets: newPresets });
	};

	const handleCustomFieldToggle = (field: keyof ParallelConfig, fieldValue: number) => {
		const currentValues = value[field] as number | number[] | undefined;
		let newValues: number[];

		if (Array.isArray(currentValues)) {
			if (currentValues.includes(fieldValue)) {
				// Remove the value
				newValues = currentValues.filter(v => v !== fieldValue);
			} else {
				// Add the value
				newValues = [...currentValues, fieldValue];
			}
		} else {
			// Convert single value to array or start new array
			newValues = currentValues !== undefined ? [currentValues, fieldValue] : [fieldValue];
		}

		onChange({
			...value,
			[field]: newValues.length === 1 ? newValues[0] : newValues
		});
	};

	const isFieldValueSelected = (field: keyof ParallelConfig, fieldValue: number): boolean => {
		const currentValues = value[field] as number | number[] | undefined;
		if (Array.isArray(currentValues)) {
			return currentValues.includes(fieldValue);
		}
		return currentValues === fieldValue;
	};

	const getTotalGPUs = () => {
		const tp = Array.isArray(value.tp) ? Math.max(...value.tp) : (value.tp || 1);
		const pp = Array.isArray(value.pp) ? Math.max(...value.pp) : (value.pp || 1);
		const dp = Array.isArray(value.dp) ? Math.max(...value.dp) : (value.dp || 1);
		const cp = Array.isArray(value.cp) ? Math.max(...value.cp) : (value.cp || 1);
		const dcp = Array.isArray(value.dcp) ? Math.max(...value.dcp) : (value.dcp || 1);

		if (normalizedRuntime === 'tensorrt-llm') {
			return tp * pp * cp;
		}
		return tp * pp * Math.max(dp, dcp);
	};

	// Compute mapped runtime arguments
	const argCombinations = useMemo(() => {
		if (configMode === 'none') {
			return { combinations: [], total: 0, truncated: false, invalidCount: 0 };
		}
		return getAllRuntimeArgCombinations(normalizedRuntime, value);
	}, [normalizedRuntime, value, configMode]);

	const formattedArgs = useMemo(() => {
		if (argCombinations.total === 0) {
			return '(no parallel arguments)';
		}

		const lines = argCombinations.combinations.map((args, idx) => {
			if (argCombinations.total > 1) {
				return `[${idx + 1}] ${args.join(' ')}`;
			}
			return args.join(' ');
		});

		let result = lines.join('\n');
		if (argCombinations.truncated) {
			result += `\n... (${argCombinations.total - argCombinations.combinations.length} more combinations)`;
		}

		return result;
	}, [argCombinations]);

	return (
		<div className="space-y-4">
			{/* Mode Selector */}
			<div>
				<label className="block text-sm font-medium text-gray-700 mb-2">
					Parallel Configuration Mode
				</label>
				<div className="grid grid-cols-3 gap-2">
					<button
						type="button"
						onClick={() => handleModeChange('none')}
						className={`px-4 py-2 border rounded-md text-sm font-medium transition-colors ${
							configMode === 'none'
								? 'bg-emerald-500 text-white border-emerald-500'
								: 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
						}`}
					>
						None
					</button>
					<button
						type="button"
						onClick={() => handleModeChange('preset')}
						className={`px-4 py-2 border rounded-md text-sm font-medium transition-colors ${
							configMode === 'preset'
								? 'bg-emerald-500 text-white border-emerald-500'
								: 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
						}`}
					>
						Preset
					</button>
					<button
						type="button"
						onClick={() => handleModeChange('custom')}
						className={`px-4 py-2 border rounded-md text-sm font-medium transition-colors ${
							configMode === 'custom'
								? 'bg-emerald-500 text-white border-emerald-500'
								: 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
						}`}
					>
						Custom
					</button>
				</div>
			</div>

			{/* None Mode */}
			{configMode === 'none' && (
				<div className="p-4 bg-gray-50 rounded-md border border-gray-200">
					<p className="text-sm text-gray-600">
						No parallel configuration. Single GPU execution will be used.
					</p>
				</div>
			)}

			{/* Preset Mode */}
			{configMode === 'preset' && (
				<div className="space-y-3">
					<label className="block text-sm font-medium text-gray-700">
						Select Presets to Compare
					</label>
					<p className="text-sm text-gray-600">
						The autotuner will create experiments for each selected preset.
					</p>
					{presets.map(preset => (
						<div
							key={preset.id}
							onClick={() => handlePresetToggle(preset.id)}
							className={`p-3 border rounded-md cursor-pointer transition-colors ${
								value.presets?.includes(preset.id)
									? 'border-emerald-500 bg-blue-50'
									: 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
							}`}
						>
							<div className="flex items-center justify-between">
								<div className="flex-1">
									<div className="flex items-center">
										<div className="font-medium text-gray-900">{preset.name}</div>
										<span className="ml-2 px-2 py-0.5 text-xs bg-gray-200 text-gray-700 rounded">
											{preset.gpus} GPU{preset.gpus !== 1 ? 's' : ''}
										</span>
									</div>
									<div className="text-sm text-gray-600 mt-1">{preset.description}</div>
								</div>
								<div className="ml-3">
									<input
										type="checkbox"
										checked={value.presets?.includes(preset.id) || false}
										onChange={() => {}}
										className="w-5 h-5 text-blue-600 rounded border-gray-300 focus:ring-emerald-500"
									/>
								</div>
							</div>
						</div>
					))}
					{normalizedRuntime === 'tensorrt-llm' && (
						<p className="mt-2 text-sm text-amber-600">
							⚠️ TensorRT-LLM requires build-time parallelism configuration. Engines must be pre-built with these settings.
						</p>
					)}
				</div>
			)}

			{/* Custom Mode */}
			{configMode === 'custom' && (
				<div className="space-y-4">
					<p className="text-sm text-gray-600">
						Configure parallel execution parameters. Select multiple values to test different combinations.
					</p>

					{/* Tensor Parallel */}
					{paramValues.tp && (
						<div>
							<label className="block text-sm font-medium text-gray-700 mb-2">
								Tensor Parallelism (TP)
								<span className="text-gray-500 font-normal ml-2">(Splits model layers across GPUs)</span>
							</label>
							<div className="grid grid-cols-3 md:grid-cols-5 gap-2">
								{paramValues.tp.map(val => (
									<div
										key={val}
										onClick={() => handleCustomFieldToggle('tp', val)}
										className={`flex items-center justify-center px-3 py-2 border rounded-md cursor-pointer transition-colors ${
											isFieldValueSelected('tp', val)
												? 'border-emerald-500 bg-blue-50'
												: 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
										}`}
									>
										<input
											type="checkbox"
											checked={isFieldValueSelected('tp', val)}
											onChange={() => {}}
											className="h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-emerald-500 pointer-events-none"
										/>
										<span className="ml-2 text-sm text-gray-900">{val}</span>
									</div>
								))}
							</div>
						</div>
					)}

					{/* Pipeline Parallel */}
					{paramValues.pp && (
						<div>
							<label className="block text-sm font-medium text-gray-700 mb-2">
								Pipeline Parallelism (PP)
								<span className="text-gray-500 font-normal ml-2">(Splits model into stages)</span>
							</label>
							<div className="grid grid-cols-3 md:grid-cols-4 gap-2">
								{paramValues.pp.map(val => (
									<div
										key={val}
										onClick={() => handleCustomFieldToggle('pp', val)}
										className={`flex items-center justify-center px-3 py-2 border rounded-md cursor-pointer transition-colors ${
											isFieldValueSelected('pp', val)
												? 'border-emerald-500 bg-blue-50'
												: 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
										}`}
									>
										<input
											type="checkbox"
											checked={isFieldValueSelected('pp', val)}
											onChange={() => {}}
											className="h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-emerald-500 pointer-events-none"
										/>
										<span className="ml-2 text-sm text-gray-900">{val}</span>
									</div>
								))}
							</div>
						</div>
					)}

					{/* Data Parallel (vLLM/SGLang only) */}
					{paramValues.dp && (
						<div>
							<label className="block text-sm font-medium text-gray-700 mb-2">
								Data Parallelism (DP)
								<span className="text-gray-500 font-normal ml-2">(Replicates model for throughput)</span>
							</label>
							<div className="grid grid-cols-3 md:grid-cols-5 gap-2">
								{paramValues.dp.map(val => (
									<div
										key={val}
										onClick={() => handleCustomFieldToggle('dp', val)}
										className={`flex items-center justify-center px-3 py-2 border rounded-md cursor-pointer transition-colors ${
											isFieldValueSelected('dp', val)
												? 'border-emerald-500 bg-blue-50'
												: 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
										}`}
									>
										<input
											type="checkbox"
											checked={isFieldValueSelected('dp', val)}
											onChange={() => {}}
											className="h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-emerald-500 pointer-events-none"
										/>
										<span className="ml-2 text-sm text-gray-900">{val}</span>
									</div>
								))}
							</div>
							{normalizedRuntime === 'sglang' && (value.tp || value.dp) && (
								<p className="mt-1 text-sm text-amber-600">
									⚠️ SGLang constraint: TP must be divisible by DP (tp % dp == 0)
								</p>
							)}
						</div>
					)}

					{/* Decode Context Parallel (vLLM only) */}
					{paramValues.dcp && (
						<div>
							<label className="block text-sm font-medium text-gray-700 mb-2">
								Decode Context Parallel (DCP)
								<span className="text-gray-500 font-normal ml-2">(For long context, vLLM only)</span>
							</label>
							<div className="grid grid-cols-3 md:grid-cols-4 gap-2">
								{paramValues.dcp.map(val => (
									<div
										key={val}
										onClick={() => handleCustomFieldToggle('dcp', val)}
										className={`flex items-center justify-center px-3 py-2 border rounded-md cursor-pointer transition-colors ${
											isFieldValueSelected('dcp', val)
												? 'border-emerald-500 bg-blue-50'
												: 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
										}`}
									>
										<input
											type="checkbox"
											checked={isFieldValueSelected('dcp', val)}
											onChange={() => {}}
											className="h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-emerald-500 pointer-events-none"
										/>
										<span className="ml-2 text-sm text-gray-900">{val}</span>
									</div>
								))}
							</div>
						</div>
					)}

					{/* Context Parallel (TensorRT-LLM only) */}
					{paramValues.cp && (
						<div>
							<label className="block text-sm font-medium text-gray-700 mb-2">
								Context Parallelism (CP)
								<span className="text-gray-500 font-normal ml-2">(For long context, TensorRT-LLM only)</span>
							</label>
							<div className="grid grid-cols-3 md:grid-cols-4 gap-2">
								{paramValues.cp.map(val => (
									<div
										key={val}
										onClick={() => handleCustomFieldToggle('cp', val)}
										className={`flex items-center justify-center px-3 py-2 border rounded-md cursor-pointer transition-colors ${
											isFieldValueSelected('cp', val)
												? 'border-emerald-500 bg-blue-50'
												: 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
										}`}
									>
										<input
											type="checkbox"
											checked={isFieldValueSelected('cp', val)}
											onChange={() => {}}
											className="h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-emerald-500 pointer-events-none"
										/>
										<span className="ml-2 text-sm text-gray-900">{val}</span>
									</div>
								))}
							</div>
						</div>
					)}

					{/* GPU Count Display */}
					{configMode === 'custom' && (value.tp || value.pp || value.dp || value.cp || value.dcp) && (
						<div className="p-3 bg-blue-50 border border-blue-200 rounded-md">
							<div className="text-sm text-blue-800">
								<strong>Maximum GPU requirement:</strong> {getTotalGPUs()} GPUs
							</div>
						</div>
					)}

					{normalizedRuntime === 'tensorrt-llm' && (
						<p className="mt-2 text-sm text-amber-600">
							⚠️ TensorRT-LLM parallelism is configured at build time. Engines must be pre-built with these settings.
						</p>
					)}
				</div>
			)}

			{/* Runtime Arguments Display */}
			{configMode !== 'none' && (
				<div className="mt-4 p-4 bg-gray-50 border border-gray-300 rounded-md">
					<div className="flex items-start">
						<div className="flex-shrink-0">
							<svg className="h-5 w-5 text-gray-600" fill="currentColor" viewBox="0 0 20 20">
								<path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
							</svg>
						</div>
						<div className="ml-3 flex-1">
							<h3 className="text-sm font-medium text-gray-800">
								Mapped Arguments for {baseRuntime.toUpperCase()}
								{argCombinations.total > 0 && (
									<span className="ml-2 text-xs text-gray-600 font-normal">
										({argCombinations.total} valid combination{argCombinations.total !== 1 ? 's' : ''})
									</span>
								)}
								{argCombinations.invalidCount > 0 && (
									<span className="ml-2 text-xs text-red-600 font-normal">
										({argCombinations.invalidCount} invalid excluded)
									</span>
								)}
							</h3>
							<div className="mt-2">
								<code className="text-xs bg-white px-3 py-2 rounded border border-gray-200 block overflow-x-auto whitespace-pre text-gray-800 font-mono">
									{formattedArgs}
								</code>
							</div>
							{argCombinations.invalidCount > 0 && (
								<p className="mt-2 text-xs text-red-600">
									⚠️ {argCombinations.invalidCount} combination{argCombinations.invalidCount !== 1 ? 's' : ''} excluded due to engine constraints
									{normalizedRuntime === 'sglang' && ' (SGLang requires tp % dp == 0)'}
									{normalizedRuntime === 'tensorrt-llm' && ' (TensorRT-LLM does not support data parallelism)'}
								</p>
							)}
						</div>
					</div>
				</div>
			)}

			{/* Help Text */}
			{configMode !== 'none' && (
				<div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-md">
					<div className="flex">
						<div className="flex-shrink-0">
							<svg className="h-5 w-5 text-blue-400" fill="currentColor" viewBox="0 0 20 20">
								<path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
							</svg>
						</div>
						<div className="ml-3">
							<h3 className="text-sm font-medium text-blue-800">Parallel Execution Info</h3>
							<div className="mt-1 text-sm text-blue-700">
								<p>Parallel parameters will be converted to runtime-specific CLI arguments automatically.</p>
								<p className="mt-1">
									<strong>TP</strong> reduces memory per GPU, <strong>DP</strong> increases throughput, <strong>PP</strong> enables very large models.
								</p>
							</div>
						</div>
					</div>
				</div>
			)}
		</div>
	);
};
