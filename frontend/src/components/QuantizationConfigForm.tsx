import React, { useState, useMemo, useEffect } from 'react';
import { QuantizationConfig } from '../types/api';
import { getAllRuntimeArgCombinations, formatMultipleCombinationsForDisplay } from '../utils/quantizationMapper';

interface QuantizationConfigFormProps {
	value: QuantizationConfig;
	onChange: (config: QuantizationConfig) => void;
	baseRuntime?: string;
}

const QUANTIZATION_PRESETS = [
	{
		id: 'none',
		name: 'None',
		description: 'No quantization configuration'
	},
	{
		id: 'default',
		name: 'Default',
		description: 'No runtime quantization (baseline)'
	},
	{
		id: 'kv-cache-fp8',
		name: 'KV Cache FP8 (Recommended)',
		description: 'FP8 KV cache only - 25-50% memory savings, <0.1% quality loss'
	},
	{
		id: 'dynamic-fp8',
		name: 'Dynamic FP8',
		description: 'Full FP8 (GEMM + KV + Attention) - 50% memory, 1.5-2x throughput (Hopper GPU)'
	},
	{
		id: 'bf16-stable',
		name: 'BF16 Stable',
		description: 'BF16 computation with FP8 KV cache - Better numerical stability'
	},
	{
		id: 'aggressive-moe',
		name: 'Aggressive MoE',
		description: 'Aggressive MoE quantization (W4A8, SGLang only)'
	}
];

const GEMM_DTYPES = ['auto', 'float16', 'bfloat16', 'float32', 'fp8', 'int8'];
const KVCACHE_DTYPES = ['auto', 'fp16', 'bfloat16', 'fp8', 'fp8_e5m2', 'fp8_e4m3', 'int8', 'int4'];
const ATTENTION_DTYPES = ['auto', 'float16', 'bfloat16', 'fp8', 'fp8_e5m2', 'fp8_e4m3', 'fp8_block'];
const MOE_DTYPES = ['auto', 'float16', 'bfloat16', 'fp8', 'w4afp8', 'mxfp4', 'int8'];

export const QuantizationConfigForm: React.FC<QuantizationConfigFormProps> = ({ value, onChange, baseRuntime = 'sglang' }) => {
	const [configMode, setConfigMode] = useState<'none' | 'preset' | 'custom'>(
		value.presets && value.presets.length > 0 ? 'preset' :
		(value.gemm_dtype || value.kvcache_dtype || value.attention_dtype || value.moe_dtype) ? 'custom' :
		'none'
	);

	// Update configMode when value prop changes (e.g., when loading task for editing)
	useEffect(() => {
		const detectedMode = value.presets && value.presets.length > 0 ? 'preset' :
			(value.gemm_dtype || value.kvcache_dtype || value.attention_dtype || value.moe_dtype) ? 'custom' :
			'none';
		setConfigMode(detectedMode);
	}, [value]);

	// Compute mapped runtime arguments
	const argCombinations = useMemo(() => {
		if (configMode === 'none') {
			return { combinations: [], total: 0, truncated: false };
		}
		return getAllRuntimeArgCombinations(baseRuntime, value, 10);
	}, [baseRuntime, value, configMode]);

	const formattedArgs = useMemo(() => {
		return formatMultipleCombinationsForDisplay(
			argCombinations.combinations,
			argCombinations.total,
			argCombinations.truncated
		);
	}, [argCombinations]);

	const handleModeChange = (mode: 'none' | 'preset' | 'custom') => {
		setConfigMode(mode);
		if (mode === 'none') {
			onChange({});
		} else if (mode === 'preset') {
			onChange({ presets: ['default', 'kv-cache-fp8'] });
		} else if (mode === 'custom') {
			onChange({
				gemm_dtype: 'auto',
				kvcache_dtype: 'auto',
				attention_dtype: 'auto',
				moe_dtype: 'auto'
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

	const handleCustomFieldToggle = (field: keyof QuantizationConfig, fieldValue: string) => {
		const currentValues = value[field];
		let newValues: string[];

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
			newValues = currentValues ? [currentValues, fieldValue] : [fieldValue];
		}

		onChange({
			...value,
			[field]: newValues.length === 1 ? newValues[0] : newValues
		});
	};

	const isFieldValueSelected = (field: keyof QuantizationConfig, fieldValue: string): boolean => {
		const currentValues = value[field];
		if (Array.isArray(currentValues)) {
			return currentValues.includes(fieldValue);
		}
		return currentValues === fieldValue;
	};

	return (
		<div className="space-y-4">
			{/* Mode Selector */}
			<div>
				<label className="block text-sm font-medium text-gray-700 mb-2">
					Quantization Mode
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
						No quantization configuration. The model will use default precision.
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
					{QUANTIZATION_PRESETS.filter(p => p.id !== 'none').map(preset => (
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
									<div className="font-medium text-gray-900">{preset.name}</div>
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
				</div>
			)}

			{/* Custom Mode */}
			{configMode === 'custom' && (
				<div className="space-y-4">
					<p className="text-sm text-gray-600">
						Configure each field independently. Select multiple values to test different combinations.
					</p>

					{/* GEMM dtype */}
					<div>
						<label className="block text-sm font-medium text-gray-700 mb-2">
							GEMM dtype
							<span className="text-gray-500 font-normal ml-2">(Linear layers, MLPs)</span>
						</label>
						<div className="grid grid-cols-2 md:grid-cols-3 gap-2">
							{GEMM_DTYPES.map(dtype => (
								<div
									key={dtype}
									onClick={() => handleCustomFieldToggle('gemm_dtype', dtype)}
									className={`flex items-center px-3 py-2 border rounded-md cursor-pointer transition-colors ${
										isFieldValueSelected('gemm_dtype', dtype)
											? 'border-emerald-500 bg-blue-50'
											: 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
									}`}
								>
									<input
										type="checkbox"
										checked={isFieldValueSelected('gemm_dtype', dtype)}
										onChange={() => {}}
										className="h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-emerald-500 pointer-events-none"
									/>
									<span className="ml-2 text-sm text-gray-900">{dtype}</span>
								</div>
							))}
						</div>
						{(isFieldValueSelected('gemm_dtype', 'fp8') || (Array.isArray(value.gemm_dtype) && value.gemm_dtype.includes('fp8'))) && (
							<p className="mt-1 text-sm text-amber-600">
								⚠️ FP8 applies W8A8 dynamic quantization (only for unquantized models)
							</p>
						)}
						{(isFieldValueSelected('gemm_dtype', 'int8') || (Array.isArray(value.gemm_dtype) && value.gemm_dtype.includes('int8'))) && (
							<p className="mt-1 text-sm text-red-600">
								⚠️ WARNING: Plain 'int8' is not supported by SGLang. Use 'fp8' for dynamic quantization or pre-quantized models (AWQ, GPTQ) instead.
							</p>
						)}
					</div>

					{/* KV Cache dtype */}
					<div>
						<label className="block text-sm font-medium text-gray-700 mb-2">
							KV Cache dtype
							<span className="text-gray-500 font-normal ml-2">(Key-value cache storage)</span>
						</label>
						<div className="grid grid-cols-2 md:grid-cols-4 gap-2">
							{KVCACHE_DTYPES.map(dtype => (
								<div
									key={dtype}
									onClick={() => handleCustomFieldToggle('kvcache_dtype', dtype)}
									className={`flex items-center px-3 py-2 border rounded-md cursor-pointer transition-colors ${
										isFieldValueSelected('kvcache_dtype', dtype)
											? 'border-emerald-500 bg-blue-50'
											: 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
									}`}
								>
									<input
										type="checkbox"
										checked={isFieldValueSelected('kvcache_dtype', dtype)}
										onChange={() => {}}
										className="h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-emerald-500 pointer-events-none"
									/>
									<span className="ml-2 text-sm text-gray-900">{dtype}</span>
								</div>
							))}
						</div>
						{(isFieldValueSelected('kvcache_dtype', 'fp8') ||
						  isFieldValueSelected('kvcache_dtype', 'fp8_e5m2') ||
						  isFieldValueSelected('kvcache_dtype', 'fp8_e4m3') ||
						  (Array.isArray(value.kvcache_dtype) && value.kvcache_dtype.some(v => v.includes('fp8')))) && (
							<p className="mt-1 text-sm text-green-600">
								✓ ~50% KV cache memory savings (Recommended: fp8_e5m2 for best quality)
							</p>
						)}
					</div>

					{/* Attention dtype */}
					<div>
						<label className="block text-sm font-medium text-gray-700 mb-2">
							Attention dtype
							<span className="text-gray-500 font-normal ml-2">(Attention mechanism)</span>
						</label>
						<div className="grid grid-cols-2 md:grid-cols-4 gap-2">
							{ATTENTION_DTYPES.map(dtype => (
								<div
									key={dtype}
									onClick={() => handleCustomFieldToggle('attention_dtype', dtype)}
									className={`flex items-center px-3 py-2 border rounded-md cursor-pointer transition-colors ${
										isFieldValueSelected('attention_dtype', dtype)
											? 'border-emerald-500 bg-blue-50'
											: 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
									}`}
								>
									<input
										type="checkbox"
										checked={isFieldValueSelected('attention_dtype', dtype)}
										onChange={() => {}}
										className="h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-emerald-500 pointer-events-none"
									/>
									<span className="ml-2 text-sm text-gray-900">{dtype}</span>
								</div>
							))}
						</div>
						{(isFieldValueSelected('attention_dtype', 'fp8') ||
						  isFieldValueSelected('attention_dtype', 'fp8_e5m2') ||
						  isFieldValueSelected('attention_dtype', 'fp8_e4m3') ||
						  (Array.isArray(value.attention_dtype) && value.attention_dtype.some(v => v.includes('fp8')))) && (
							<p className="mt-1 text-sm text-blue-600">
								ℹ️ FP8 attention supported by TensorRT-LLM and SGLang (vLLM falls back to GEMM dtype)
							</p>
						)}
					</div>

					{/* MoE dtype */}
					<div>
						<label className="block text-sm font-medium text-gray-700 mb-2">
							MoE dtype
							<span className="text-gray-500 font-normal ml-2">(Mixture-of-Experts)</span>
						</label>
						<div className="grid grid-cols-2 md:grid-cols-4 gap-2">
							{MOE_DTYPES.map(dtype => (
								<div
									key={dtype}
									onClick={() => handleCustomFieldToggle('moe_dtype', dtype)}
									className={`flex items-center px-3 py-2 border rounded-md cursor-pointer transition-colors ${
										isFieldValueSelected('moe_dtype', dtype)
											? 'border-emerald-500 bg-blue-50'
											: 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
									}`}
								>
									<input
										type="checkbox"
										checked={isFieldValueSelected('moe_dtype', dtype)}
										onChange={() => {}}
										className="h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-emerald-500 pointer-events-none"
									/>
									<span className="ml-2 text-sm text-gray-900">{dtype}</span>
								</div>
							))}
						</div>
						{(isFieldValueSelected('moe_dtype', 'w4afp8') ||
						  isFieldValueSelected('moe_dtype', 'mxfp4') ||
						  (Array.isArray(value.moe_dtype) && (value.moe_dtype.includes('w4afp8') || value.moe_dtype.includes('mxfp4')))) && (
							<p className="mt-1 text-sm text-purple-600">
								ℹ️ w4afp8/mxfp4 are SGLang-specific (not supported by vLLM or TensorRT-LLM)
							</p>
						)}
						{(isFieldValueSelected('moe_dtype', 'int8') || (Array.isArray(value.moe_dtype) && value.moe_dtype.includes('int8'))) && (
							<p className="mt-1 text-sm text-red-600">
								⚠️ WARNING: Plain 'int8' for MoE is not supported by SGLang. Consider using 'w4afp8' or 'mxfp4' instead.
							</p>
						)}
					</div>
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
								{argCombinations.total > 1 && (
									<span className="ml-2 text-xs text-gray-600 font-normal">
										({argCombinations.total} combination{argCombinations.total !== 1 ? 's' : ''})
									</span>
								)}
							</h3>
							<div className="mt-2">
								<code className="text-xs bg-white px-3 py-2 rounded border border-gray-200 block overflow-x-auto whitespace-pre text-gray-800 font-mono">
									{formattedArgs}
								</code>
							</div>
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
							<h3 className="text-sm font-medium text-blue-800">Parameter Priority</h3>
							<div className="mt-1 text-sm text-blue-700">
								<p>User parameters in the "Parameters" section will override quant_config-derived arguments.</p>
								<p className="mt-1">For offline-quantized models (AWQ, GPTQ, GGUF), GEMM dtype is ignored but KV cache quantization still applies.</p>
							</div>
						</div>
					</div>
				</div>
			)}
		</div>
	);
};
