import { useState, useEffect } from 'react';
import { useMutation, useQueryClient, useQuery } from '@tanstack/react-query';
import { presetService } from '../services/presetService';
import { runtimeParamsService } from '../services/runtimeParamsService';
import toast from 'react-hot-toast';
import { useEscapeKey } from '@/hooks/useEscapeKey';
import type { Preset } from '../types/preset';

interface PresetEditModalProps {
  preset: Preset;
  onClose: () => void;
}

interface ParamField {
  name: string;
  values: string;
}

export default function PresetEditModal({ preset, onClose }: PresetEditModalProps) {
  // Handle Escape key to close modal
  useEscapeKey(onClose);
  const queryClient = useQueryClient();

  // Form state
  const [name, setName] = useState(preset.name);
  const [description, setDescription] = useState(preset.description || '');
  const [category, setCategory] = useState(preset.category || '');
  const [runtime, setRuntime] = useState<'sglang' | 'vllm' | ''>(preset.runtime || '');
  const [showSuggestions, setShowSuggestions] = useState(false);

  // Fetch commonly tuned parameters for the selected runtime
  const { data: commonlyTuned } = useQuery({
    queryKey: ['commonly-tuned', runtime],
    queryFn: () => runtime ? runtimeParamsService.getCommonlyTuned(runtime as 'sglang' | 'vllm') : Promise.resolve(null),
    enabled: !!runtime,
  });
  const [parameters, setParameters] = useState<ParamField[]>([]);

  // Initialize parameters from preset
  useEffect(() => {
    const paramFields: ParamField[] = Object.entries(preset.parameters).map(([key, values]) => ({
      name: key,
      values: Array.isArray(values) ? values.join(', ') : String(values)
    }));
    setParameters(paramFields);
  }, [preset]);

  // Update mutation
  const updateMutation = useMutation({
    mutationFn: (data: any) => presetService.update(preset.id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['presets'] });
      toast.success('Preset updated successfully');
      onClose();
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to update preset');
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

  const parseParameterValue = (valueStr: string): any[] => {
    // Try to parse as numbers first
    const parts = valueStr.split(',').map(s => s.trim()).filter(Boolean);

    // Check if all parts are numbers
    const allNumbers = parts.every(part => !isNaN(parseFloat(part)));
    if (allNumbers) {
      return parts.map(part => parseFloat(part));
    }

    // Check if all parts are booleans
    const allBooleans = parts.every(part => part === 'true' || part === 'false');
    if (allBooleans) {
      return parts.map(part => part === 'true');
    }

    // Otherwise treat as strings
    return parts;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Parse parameters
    const parsedParams: Record<string, any[]> = {};
    for (const param of parameters) {
      if (param.name && param.values) {
        parsedParams[param.name] = parseParameterValue(param.values);
      }
    }

    const updateData = {
      name: name !== preset.name ? name : undefined,
      description: description !== preset.description ? description : undefined,
      category: category !== preset.category ? category : undefined,
      runtime: (runtime || undefined) !== preset.runtime ? (runtime || undefined) : undefined,
      parameters: JSON.stringify(parsedParams) !== JSON.stringify(preset.parameters) ? parsedParams : undefined,
    };

    // Remove undefined fields
    const cleanData = Object.fromEntries(
      Object.entries(updateData).filter(([_, v]) => v !== undefined)
    );

    if (Object.keys(cleanData).length === 0) {
      toast.error('No changes detected');
      return;
    }

    updateMutation.mutate(cleanData);
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-3xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Edit Preset</h2>
            {preset.is_system && (
              <p className="text-sm text-amber-600 mt-1 flex items-center gap-1">
                <span>⚠️</span>
                <span>Warning: You are editing a system preset</span>
              </p>
            )}
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 text-2xl leading-none"
          >
            ×
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          {/* Basic Info */}
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Preset Name *
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500"
                placeholder="e.g., High Performance"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Description
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={2}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500"
                placeholder="Describe this preset..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Category
              </label>
              <input
                type="text"
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500"
                placeholder="e.g., performance, memory, general"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Runtime
              </label>
              <select
                value={runtime}
                onChange={(e) => setRuntime(e.target.value as 'sglang' | 'vllm' | '')}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500"
              >
                <option value="">Universal (all runtimes)</option>
                <option value="sglang">SGLang</option>
                <option value="vllm">vLLM</option>
              </select>
              <p className="text-xs text-gray-500 mt-1">
                Select the target runtime for this preset, or leave as universal
              </p>
            </div>
          </div>

          {/* Parameters */}
          <div>
            <div className="flex justify-between items-center mb-3">
              <label className="block text-sm font-medium text-gray-700">
                Parameters
              </label>
              <div className="flex gap-2">
                {runtime && commonlyTuned && (
                  <button
                    type="button"
                    onClick={() => setShowSuggestions(!showSuggestions)}
                    className="px-3 py-1 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 text-sm"
                  >
                    {showSuggestions ? 'Hide' : 'Show'} Suggestions
                  </button>
                )}
                <button
                  type="button"
                  onClick={addParameter}
                  className="px-3 py-1 bg-emerald-500 text-white rounded-md hover:bg-emerald-600 text-sm"
                >
                  Add Parameter
                </button>
              </div>
            </div>

            {/* Commonly tuned parameters suggestions */}
            {showSuggestions && runtime && commonlyTuned && (
              <div className="mb-3 p-3 bg-blue-50 border border-blue-200 rounded-md">
                <p className="text-sm font-medium text-blue-900 mb-2">
                  Commonly tuned parameters for {runtime.toUpperCase()}:
                </p>
                <div className="flex flex-wrap gap-2">
                  {commonlyTuned.parameters.map((param) => (
                    <button
                      key={param}
                      type="button"
                      onClick={() => {
                        // Add parameter if not already in the list
                        if (!parameters.some(p => p.name === param)) {
                          setParameters([...parameters, { name: param, values: '' }]);
                        }
                      }}
                      className="px-2 py-1 bg-white border border-blue-300 text-blue-700 rounded text-xs hover:bg-blue-100"
                      disabled={parameters.some(p => p.name === param)}
                    >
                      {param}
                    </button>
                  ))}
                </div>
                <p className="text-xs text-blue-600 mt-2">
                  Click a parameter to add it to your preset
                </p>
              </div>
            )}

            <div className="space-y-3">
              {parameters.map((param, index) => (
                <div key={index} className="flex gap-3 items-start">
                  <div className="flex-1">
                    <input
                      type="text"
                      value={param.name}
                      onChange={(e) => updateParameter(index, 'name', e.target.value)}
                      placeholder="Parameter name (e.g., tp-size)"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    />
                  </div>
                  <div className="flex-1">
                    <input
                      type="text"
                      value={param.values}
                      onChange={(e) => updateParameter(index, 'values', e.target.value)}
                      placeholder="Values (e.g., 1, 2, 4)"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    />
                  </div>
                  <button
                    type="button"
                    onClick={() => removeParameter(index)}
                    className="px-3 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
                    disabled={parameters.length === 1}
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>

            <p className="text-sm text-gray-500 mt-2">
              Enter values as comma-separated. Numbers: "1, 2, 4", Booleans: "true, false", Strings: "fcfs, lpm"
            </p>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-4 justify-end pt-4 border-t border-gray-200">
            <button
              type="button"
              onClick={onClose}
              className="px-6 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={updateMutation.isPending}
              className="px-6 py-2 bg-emerald-500 text-white rounded-md hover:bg-emerald-600 disabled:bg-blue-400"
            >
              {updateMutation.isPending ? 'Saving...' : 'Save Changes'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
