import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { presetService } from '../services/presetService';
import toast from 'react-hot-toast';
import type { Preset } from '../types/preset';
import PresetEditModal from '../components/PresetEditModal';

export default function Presets() {
  const queryClient = useQueryClient();
  const [selectedCategory, setSelectedCategory] = useState<string>('');
  const [deleteConfirm, setDeleteConfirm] = useState<number | null>(null);
  const [editingPreset, setEditingPreset] = useState<Preset | null>(null);

  // Fetch presets
  const { data: presets = [], isLoading } = useQuery({
    queryKey: ['presets', selectedCategory],
    queryFn: () => presetService.getAll(selectedCategory || undefined),
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: (id: number) => presetService.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['presets'] });
      toast.success('Preset deleted successfully');
      setDeleteConfirm(null);
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to delete preset');
      setDeleteConfirm(null);
    },
  });

  // Export preset
  const handleExport = async (preset: Preset) => {
    try {
      const blob = await presetService.exportPreset(preset.id);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `preset-${preset.name.toLowerCase().replace(/\s+/g, '-')}.json`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      toast.success('Preset exported successfully');
    } catch (error) {
      toast.error('Failed to export preset');
    }
  };

  // Import preset
  const handleImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      await presetService.importPreset(file);
      queryClient.invalidateQueries({ queryKey: ['presets'] });
      toast.success('Preset imported successfully');
      event.target.value = ''; // Reset input
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to import preset');
      event.target.value = '';
    }
  };

  const categories = Array.from(new Set(presets.map(p => p.category).filter(Boolean)));

  if (isLoading) {
    return (
      <div className="p-8 text-center">
        <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-emerald-500"></div>
        <p className="mt-2 text-gray-600">Loading presets...</p>
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-2xl font-bold">Parameter Presets</h1>
          <div className="flex gap-2">
            <label className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 cursor-pointer">
              Import
              <input
                type="file"
                accept=".json"
                onChange={handleImport}
                className="hidden"
              />
            </label>
          </div>
        </div>

        {/* Category Filter */}
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium text-gray-700">Filter by category:</label>
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="border border-gray-300 rounded px-3 py-1 text-sm"
          >
            <option value="">All</option>
            {categories.map(cat => (
              <option key={cat} value={cat}>{cat}</option>
            ))}
          </select>
          {selectedCategory && (
            <button
              onClick={() => setSelectedCategory('')}
              className="text-sm text-gray-600 hover:text-gray-800 underline"
            >
              Clear filter
            </button>
          )}
        </div>
      </div>

      {/* Presets Grid */}
      {presets.length === 0 ? (
        <div className="text-center py-12 bg-gray-50 rounded-lg">
          <p className="text-gray-600">No presets found</p>
          {selectedCategory && (
            <button
              onClick={() => setSelectedCategory('')}
              className="mt-2 text-blue-600 hover:text-blue-800 text-sm"
            >
              Show all presets
            </button>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4">
          {presets.map((preset) => (
            <div
              key={preset.id}
              className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow"
            >
              {/* Header Row */}
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <h3 className="text-lg font-semibold text-gray-900">{preset.name}</h3>
                    {preset.is_system && (
                      <span className="px-2 py-0.5 text-xs bg-gray-200 text-gray-700 rounded">
                        System
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-gray-600">{preset.description || 'No description'}</p>
                </div>

                {/* Action Buttons */}
                <div className="flex items-center gap-2 ml-4">
                  <button
                    onClick={() => handleExport(preset)}
                    className="px-3 py-1.5 text-sm text-blue-600 hover:bg-blue-50 rounded border border-blue-300"
                    title="Export"
                  >
                    Export
                  </button>
                  <button
                    onClick={() => setEditingPreset(preset)}
                    className="px-3 py-1.5 text-sm text-green-600 hover:bg-green-50 rounded border border-green-300"
                    title="Edit"
                  >
                    Edit
                  </button>
                  {deleteConfirm === preset.id ? (
                    <>
                      <button
                        onClick={() => deleteMutation.mutate(preset.id)}
                        className="px-3 py-1.5 text-sm text-white bg-red-600 hover:bg-red-700 rounded"
                      >
                        Confirm
                      </button>
                      <button
                        onClick={() => setDeleteConfirm(null)}
                        className="px-3 py-1.5 text-sm text-gray-600 hover:bg-gray-100 rounded border border-gray-300"
                      >
                        Cancel
                      </button>
                    </>
                  ) : (
                    <button
                      onClick={() => setDeleteConfirm(preset.id)}
                      className="px-3 py-1.5 text-sm text-red-600 hover:bg-red-50 rounded border border-red-300"
                      title="Delete"
                    >
                      Delete
                    </button>
                  )}
                </div>
              </div>

              {/* Metadata Row */}
              <div className="flex items-center gap-4 mb-3">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-500">Category:</span>
                  <span className="px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded">
                    {preset.category || 'uncategorized'}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-500">Runtime:</span>
                  {preset.runtime ? (
                    <span className={`px-2 py-1 text-xs rounded ${
                      preset.runtime === 'sglang'
                        ? 'bg-green-100 text-green-800'
                        : 'bg-purple-100 text-purple-800'
                    }`}>
                      {preset.runtime}
                    </span>
                  ) : (
                    <span className="text-xs text-gray-400">universal</span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-500">Parameters:</span>
                  <span className="px-2 py-1 text-xs bg-gray-100 text-gray-800 rounded">
                    {Object.keys(preset.parameters).length}
                  </span>
                </div>
              </div>

              {/* Parameters List */}
              <div className="border-t border-gray-200 pt-3">
                <div className="text-xs text-gray-500 mb-2">Parameter Configuration:</div>
                <div className="flex flex-wrap gap-2 max-h-48 overflow-y-auto">
                  {Object.entries(preset.parameters).map(([name, values]) => (
                    <div
                      key={name}
                      className="px-2 py-1 bg-gray-50 border border-gray-200 rounded text-xs"
                      title={`${name}: ${JSON.stringify(values)}`}
                    >
                      <span className="font-medium text-gray-700">{name}</span>
                      <span className="text-gray-500 ml-1">
                        ({Array.isArray(values) ? values.length : 1})
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Summary */}
      <div className="mt-4 text-sm text-gray-600">
        Showing {presets.length} preset{presets.length !== 1 ? 's' : ''}
        {selectedCategory && ` in category "${selectedCategory}"`}
      </div>

      {/* Edit Modal */}
      {editingPreset && (
        <PresetEditModal
          preset={editingPreset}
          onClose={() => setEditingPreset(null)}
        />
      )}
    </div>
  );
}
