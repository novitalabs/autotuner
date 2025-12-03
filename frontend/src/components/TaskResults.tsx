import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/services/api';
import type { Task, Experiment } from '@/types/api';
import toast from 'react-hot-toast';
import { useEscapeKey } from '@/hooks/useEscapeKey';
import { useState, useMemo, useEffect } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
  ScatterChart,
  Scatter,
  ZAxis,
  LineChart,
  Line,
  ReferenceLine,
} from 'recharts';

interface TaskResultsProps {
  task: Task;
  onClose: () => void;
}

export default function TaskResults({ task, onClose }: TaskResultsProps) {
  // Handle Escape key to close modal
  useEscapeKey(onClose);

  // State for scatter plot axis selection (initialized from localStorage or defaults)
  const [scatterXAxis, setScatterXAxis] = useState<string>(() => {
    return localStorage.getItem('taskResults.scatterXAxis') || 'mean_output_throughput_tokens_per_s';
  });
  const [scatterYAxis, setScatterYAxis] = useState<string>(() => {
    return localStorage.getItem('taskResults.scatterYAxis') || 'num_concurrency';
  });
  const [hoveredExperiment, setHoveredExperiment] = useState<number | null>(null);

  // New state for enhanced features
  const [selectedExperiments, setSelectedExperiments] = useState<number[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'id' | 'score' | 'duration'>('id');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [viewMode, setViewMode] = useState<'table' | 'comparison' | 'sensitivity' | 'pareto'>('table');
  const [sensitivityParam, setSensitivityParam] = useState<string>('');

  // Save axis selections to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem('taskResults.scatterXAxis', scatterXAxis);
  }, [scatterXAxis]);

  useEffect(() => {
    localStorage.setItem('taskResults.scatterYAxis', scatterYAxis);
  }, [scatterYAxis]);

  // Fetch experiments for this task
  const {
    data: experiments = [],
    isLoading,
  } = useQuery({
    queryKey: ['experiments', task.id],
    queryFn: () => apiClient.getExperimentsByTask(task.id),
  });

  // Initialize selectedExperiments with best experiment when data loads
  useEffect(() => {
    if (experiments.length > 0 && selectedExperiments.length === 0 && task.best_experiment_id) {
      const bestExp = experiments.find(exp => exp.id === task.best_experiment_id);
      if (bestExp) {
        setSelectedExperiments([bestExp.experiment_id]);
      }
    }
  }, [experiments, task.best_experiment_id, selectedExperiments.length]);

  if (isLoading) {
    return (
      <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-6">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading results...</p>
        </div>
      </div>
    );
  }

  const successfulExperiments = experiments.filter((exp) => exp.status === 'success');
  const bestExperiment = experiments.find((exp) => exp.id === task.best_experiment_id);

  // Helper function to get parameter differences
  const getParameterDiff = (expParams: any, bestParams: any): string[] => {
    if (!expParams || !bestParams) return [];
    const diffs: string[] = [];

    // Get all parameter keys from both experiments
    const allKeys = new Set([...Object.keys(expParams), ...Object.keys(bestParams)]);

    for (const key of allKeys) {
      const expValue = expParams[key];
      const bestValue = bestParams[key];

      if (expValue !== bestValue) {
        diffs.push(`${key}: ${expValue} (best: ${bestValue})`);
      }
    }

    return diffs;
  };

  // Helper function to format metric value for display
  const formatMetricValue = (value: any): string => {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value === 'number') return value.toFixed(2);
    if (typeof value === 'boolean') return value ? 'true' : 'false';
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
  };

  // Filter metrics to only include primitive values for charts
  const getPrimitiveMetrics = (metrics: any): Record<string, number> => {
    if (!metrics) return {};
    const result: Record<string, number> = {};
    for (const [key, value] of Object.entries(metrics)) {
      if (typeof value === 'number') {
        result[key] = value;
      }
    }
    return result;
  };

  // Extract all parameter keys
  const allParameterKeys = useMemo(() => {
    const keys = new Set<string>();
    experiments.forEach(exp => {
      Object.keys(exp.parameters || {}).forEach(key => keys.add(key));
    });
    return Array.from(keys);
  }, [experiments]);

  // Initialize sensitivity param if not set
  useEffect(() => {
    if (!sensitivityParam && allParameterKeys.length > 0) {
      setSensitivityParam(allParameterKeys[0]);
    }
  }, [allParameterKeys, sensitivityParam]);

  // Extract all metric keys
  const allMetricKeys = useMemo(() => {
    const keys = new Set<string>();
    successfulExperiments.forEach(exp => {
      Object.keys(getPrimitiveMetrics(exp.metrics)).forEach(key => keys.add(key));
    });
    return Array.from(keys);
  }, [successfulExperiments]);

  // Filter and sort experiments
  const filteredExperiments = useMemo(() => {
    let filtered = experiments.filter(exp => {
      // Status filter
      if (filterStatus !== 'all' && exp.status !== filterStatus) {
        return false;
      }

      // Search filter
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        const paramMatch = Object.entries(exp.parameters || {}).some(([key, value]) =>
          `${key}=${value}`.toLowerCase().includes(query)
        );
        const idMatch = exp.experiment_id.toString().includes(query);
        return paramMatch || idMatch;
      }

      return true;
    });

    // Sort
    filtered.sort((a, b) => {
      let comparison = 0;
      switch (sortBy) {
        case 'id':
          comparison = a.experiment_id - b.experiment_id;
          break;
        case 'score':
          comparison = (a.objective_score || Infinity) - (b.objective_score || Infinity);
          break;
        case 'duration':
          comparison = (a.elapsed_time || 0) - (b.elapsed_time || 0);
          break;
      }
      return sortOrder === 'asc' ? comparison : -comparison;
    });

    return filtered;
  }, [experiments, filterStatus, searchQuery, sortBy, sortOrder]);

  // Prepare data for charts (only primitive numeric values)
  const chartData = successfulExperiments.map((exp) => ({
    name: `Exp ${exp.experiment_id}`,
    experiment_id: exp.experiment_id,
    objective_score: exp.objective_score || 0,
    parameters: exp.parameters, // Include parameters for comparison
    ...getPrimitiveMetrics(exp.metrics),
  }));

  const formatDuration = (seconds: number | null) => {
    if (!seconds) return 'N/A';
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    if (hours > 0) return `${hours}h ${mins}m`;
    if (mins > 0) return `${mins}m ${secs}s`;
    return `${secs}s`;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success':
        return 'bg-green-100 text-green-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      case 'deploying':
        return 'bg-blue-100 text-blue-800';
      case 'benchmarking':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  // Extract raw_results for scatter plot (sub-rounds of focused experiment)
  const getScatterDataForExperiment = (exp: any) => {
    if (!exp?.metrics?.raw_results || !Array.isArray(exp.metrics.raw_results)) {
      return [];
    }

    return exp.metrics.raw_results.map((rawResult: any, index: number) => {
      // Flatten nested stats into top-level for easier access
      const flatData: any = {
        round_index: index,
        round_name: `Round ${index + 1}`,
        num_concurrency: rawResult.num_concurrency,
        batch_size: rawResult.batch_size,
        scenario: rawResult.scenario,
        mean_output_throughput_tokens_per_s: rawResult.mean_output_throughput_tokens_per_s,
        mean_input_throughput_tokens_per_s: rawResult.mean_input_throughput_tokens_per_s,
        mean_total_tokens_throughput_tokens_per_s: rawResult.mean_total_tokens_throughput_tokens_per_s,
        requests_per_second: rawResult.requests_per_second,
        error_rate: rawResult.error_rate,
        num_requests: rawResult.num_requests,
        num_completed_requests: rawResult.num_completed_requests,
      };

      // Flatten stats.* fields
      if (rawResult.stats) {
        // Extract mean values from each stat category
        if (rawResult.stats.ttft) {
          Object.entries(rawResult.stats.ttft).forEach(([key, value]) => {
            flatData[`ttft_${key}`] = value;
          });
        }
        if (rawResult.stats.tpot) {
          Object.entries(rawResult.stats.tpot).forEach(([key, value]) => {
            flatData[`tpot_${key}`] = value;
          });
        }
        if (rawResult.stats.e2e_latency) {
          Object.entries(rawResult.stats.e2e_latency).forEach(([key, value]) => {
            flatData[`e2e_latency_${key}`] = value;
          });
        }
        if (rawResult.stats.num_input_tokens) {
          Object.entries(rawResult.stats.num_input_tokens).forEach(([key, value]) => {
            flatData[`input_tokens_${key}`] = value;
          });
        }
        if (rawResult.stats.num_output_tokens) {
          Object.entries(rawResult.stats.num_output_tokens).forEach(([key, value]) => {
            flatData[`output_tokens_${key}`] = value;
          });
        }
      }

      return flatData;
    });
  };

  // Helper function to get SLO threshold for a given metric axis
  const getSLOThreshold = (metricName: string): number | null => {
    if (!task.slo) return null;

    // Map metric names to SLO config paths
    const metricMap: Record<string, { path: string; percentile?: string }> = {
      'ttft_mean': { path: 'ttft' },
      'tpot_mean': { path: 'tpot' },
      'e2e_latency_p50': { path: 'latency', percentile: 'p50' },
      'e2e_latency_p90': { path: 'latency', percentile: 'p90' },
      'e2e_latency_p99': { path: 'latency', percentile: 'p99' },
    };

    const mapping = metricMap[metricName];
    if (!mapping) return null;

    try {
      if (mapping.percentile) {
        // For latency percentiles
        const latencyConfig = task.slo.latency?.[mapping.percentile as 'p50' | 'p90' | 'p99'];
        return latencyConfig?.threshold ?? null;
      } else {
        // For ttft/tpot
        const metricConfig = task.slo[mapping.path as 'ttft' | 'tpot'];
        return metricConfig?.threshold ?? null;
      }
    } catch {
      return null;
    }
  };

  // Get scatter data for best experiment (always show in green)
  const bestExperimentData = useMemo(() => {
    return bestExperiment ? getScatterDataForExperiment(bestExperiment) : [];
  }, [bestExperiment]);

  // Get scatter data for hovered experiment (show in blue if different from best)
  const hoveredExperimentData = useMemo(() => {
    if (!hoveredExperiment || hoveredExperiment === bestExperiment?.experiment_id) {
      return [];
    }
    const hoveredExp = experiments.find(exp => exp.experiment_id === hoveredExperiment);
    return hoveredExp ? getScatterDataForExperiment(hoveredExp) : [];
  }, [hoveredExperiment, experiments, bestExperiment]);

  // Get all available numeric fields from scatter data for axis selection
  const scatterAxisOptions = useMemo(() => {
    const data = bestExperimentData.length > 0 ? bestExperimentData : hoveredExperimentData;
    return data.length > 0
      ? Object.keys(data[0]).filter(key =>
          typeof data[0][key] === 'number' && key !== 'round_index'
        )
      : [];
  }, [bestExperimentData, hoveredExperimentData]);

  // Parameter sensitivity analysis data
  const sensitivityData = useMemo(() => {
    if (!sensitivityParam) return [];

    // Group experiments by the selected parameter value
    const grouped = new Map<any, Experiment[]>();
    successfulExperiments.forEach(exp => {
      const paramValue = exp.parameters?.[sensitivityParam];
      if (paramValue !== undefined) {
        if (!grouped.has(paramValue)) {
          grouped.set(paramValue, []);
        }
        grouped.get(paramValue)!.push(exp);
      }
    });

    // Calculate average metrics for each parameter value
    return Array.from(grouped.entries()).map(([paramValue, exps]) => {
      const avgMetrics: any = { [sensitivityParam]: paramValue, count: exps.length };

      // Calculate average objective score
      const scores = exps.map(e => e.objective_score).filter(s => s !== null) as number[];
      if (scores.length > 0) {
        avgMetrics.avg_objective_score = scores.reduce((a, b) => a + b, 0) / scores.length;
        avgMetrics.min_objective_score = Math.min(...scores);
        avgMetrics.max_objective_score = Math.max(...scores);
      }

      // Calculate average for key metrics
      const keyMetrics = ['mean_output_throughput_tokens_per_s', 'ttft_mean', 'tpot_mean', 'e2e_latency_mean'];
      keyMetrics.forEach(metric => {
        const values = exps
          .map(e => getPrimitiveMetrics(e.metrics)[metric])
          .filter(v => v !== undefined) as number[];
        if (values.length > 0) {
          avgMetrics[`avg_${metric}`] = values.reduce((a, b) => a + b, 0) / values.length;
        }
      });

      return avgMetrics;
    }).sort((a, b) => {
      // Sort by parameter value
      const aVal = a[sensitivityParam];
      const bVal = b[sensitivityParam];
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return aVal - bVal;
      }
      return String(aVal).localeCompare(String(bVal));
    });
  }, [sensitivityParam, successfulExperiments]);

  // Pareto frontier data (for multi-objective optimization)
  const paretoData = useMemo(() => {
    // Define two objectives to plot (customize based on task.optimization.objective)
    const objective1Key = 'mean_output_throughput_tokens_per_s';
    const objective2Key = 'e2e_latency_mean';

    return successfulExperiments
      .map(exp => {
        const metrics = getPrimitiveMetrics(exp.metrics);
        return {
          experiment_id: exp.experiment_id,
          name: `Exp ${exp.experiment_id}`,
          throughput: metrics[objective1Key],
          latency: metrics[objective2Key],
          objective_score: exp.objective_score,
          is_best: exp.id === task.best_experiment_id,
          parameters: exp.parameters,
        };
      })
      .filter(d => d.throughput !== undefined && d.latency !== undefined);
  }, [successfulExperiments, task.best_experiment_id]);

  // Calculate Pareto frontier (non-dominated solutions)
  const paretoFrontier = useMemo(() => {
    if (paretoData.length === 0) return [];

    const frontier: typeof paretoData = [];
    paretoData.forEach(candidate => {
      // Check if this point is dominated by any other point
      const isDominated = paretoData.some(other => {
        if (other === candidate) return false;
        // For throughput: higher is better, for latency: lower is better
        return (other.throughput >= candidate.throughput && other.latency <= candidate.latency) &&
               (other.throughput > candidate.throughput || other.latency < candidate.latency);
      });
      if (!isDominated) {
        frontier.push(candidate);
      }
    });

    return frontier.sort((a, b) => a.throughput - b.throughput);
  }, [paretoData]);

  // Export functions
  const exportToCSV = () => {
    const headers = [
      'experiment_id',
      'status',
      'objective_score',
      'duration_seconds',
      ...allParameterKeys,
      ...allMetricKeys,
    ];

    const rows = experiments.map(exp => [
      exp.experiment_id,
      exp.status,
      exp.objective_score ?? '',
      exp.elapsed_time ?? '',
      ...allParameterKeys.map(key => exp.parameters?.[key] ?? ''),
      ...allMetricKeys.map(key => getPrimitiveMetrics(exp.metrics)[key] ?? ''),
    ]);

    const csvContent = [
      headers.join(','),
      ...rows.map(row => row.map(cell => `"${cell}"`).join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${task.task_name}_results.csv`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success('Results exported to CSV');
  };

  const exportToJSON = () => {
    const data = {
      task: {
        id: task.id,
        name: task.task_name,
        description: task.description,
        status: task.status,
        total_experiments: task.total_experiments,
        successful_experiments: task.successful_experiments,
        best_experiment_id: task.best_experiment_id,
        elapsed_time: task.elapsed_time,
      },
      experiments: experiments.map(exp => ({
        experiment_id: exp.experiment_id,
        status: exp.status,
        parameters: exp.parameters,
        metrics: exp.metrics,
        objective_score: exp.objective_score,
        elapsed_time: exp.elapsed_time,
        error_message: exp.error_message,
      })),
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${task.task_name}_results.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success('Results exported to JSON');
  };

  // Toggle experiment selection
  const toggleExperimentSelection = (expId: number) => {
    setSelectedExperiments(prev => {
      if (prev.includes(expId)) {
        return prev.filter(id => id !== expId);
      } else {
        return [...prev, expId];
      }
    });
  };

  // Get selected experiment objects
  const selectedExperimentObjects = useMemo(() => {
    return experiments.filter(exp => selectedExperiments.includes(exp.experiment_id));
  }, [experiments, selectedExperiments]);

  return (
    <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto z-50">
      <div className="min-h-screen px-4 py-8">
        <div className="max-w-7xl mx-auto bg-white rounded-lg shadow-xl">
          {/* Header */}
          <div className="border-b border-gray-200 px-6 py-4 flex justify-between items-center">
            <div>
              <h2 className="text-2xl font-bold text-gray-900">{task.task_name} - Results</h2>
              <p className="text-sm text-gray-500 mt-1">
                {task.successful_experiments} / {task.total_experiments} successful experiments
              </p>
            </div>
            <div className="flex items-center gap-3">
              {/* Export buttons */}
              <div className="flex gap-2">
                <button
                  onClick={exportToCSV}
                  className="px-3 py-1.5 text-sm bg-green-600 text-white rounded hover:bg-green-700 transition-colors flex items-center gap-1"
                  title="Export to CSV"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  CSV
                </button>
                <button
                  onClick={exportToJSON}
                  className="px-3 py-1.5 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors flex items-center gap-1"
                  title="Export to JSON"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  JSON
                </button>
              </div>
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-gray-600 transition-colors"
              >
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>

          <div className="p-6 space-y-6">
            {/* View Mode Tabs */}
            <div className="border-b border-gray-200">
              <nav className="flex gap-4">
                <button
                  onClick={() => setViewMode('table')}
                  className={`pb-3 px-1 border-b-2 font-medium text-sm transition-colors ${
                    viewMode === 'table'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Overview
                </button>
                <button
                  onClick={() => setViewMode('comparison')}
                  className={`pb-3 px-1 border-b-2 font-medium text-sm transition-colors ${
                    viewMode === 'comparison'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Compare ({selectedExperiments.length})
                </button>
                <button
                  onClick={() => setViewMode('sensitivity')}
                  className={`pb-3 px-1 border-b-2 font-medium text-sm transition-colors ${
                    viewMode === 'sensitivity'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Sensitivity Analysis
                </button>
                <button
                  onClick={() => setViewMode('pareto')}
                  className={`pb-3 px-1 border-b-2 font-medium text-sm transition-colors ${
                    viewMode === 'pareto'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Pareto Frontier
                </button>
              </nav>
            </div>

            {/* Overview View */}
            {viewMode === 'table' && (
              <>
                {/* Best Configuration Card */}
                {bestExperiment && (
                  <div className="bg-gradient-to-r from-green-50 to-emerald-50 border-2 border-green-200 rounded-lg p-4">
                    <div className="flex items-center mb-3">
                      <svg className="w-5 h-5 text-green-600 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      <h3 className="text-lg font-bold text-green-900">Best Configuration</h3>
                      <span className="ml-auto text-xs text-green-700">Exp #{bestExperiment.experiment_id}</span>
                    </div>

                    {/* Objective Score on Top */}
                    <div className="mb-3">
                      <h4 className="text-xs font-medium text-green-700 mb-1.5">Objective Score</h4>
                        <div className="text-green-900">
                          {bestExperiment.objective_score?.toFixed(4) || 'N/A'}
                        </div>
                        <p className="text-xs text-green-600 mt-1">
                          {task.optimization?.objective === 'minimize_latency' && 'Lower is better'}
                          {task.optimization?.objective === 'maximize_throughput' && 'Higher is better'}
                          {task.optimization?.objective === 'balanced' && 'Balanced score'}
                        </p>
                    </div>

                    {/* Parameters and Metrics in a Row */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {/* Parameters Section - Table */}
                      <div>
                        <div className="flex items-center justify-between mb-1.5">
                          <h4 className="text-xs font-medium text-green-700">Optimal Parameters</h4>
                          <button
                            onClick={() => {
                              const paramsText = Object.entries(bestExperiment.parameters)
                                .map(([key, value]) => `${key}=${value}`)
                                .join(' ');
                              navigator.clipboard.writeText(paramsText);
                              toast.success('Parameters copied to clipboard!');
                            }}
                            className="text-xs px-2 py-0.5 bg-green-600 text-white rounded hover:bg-green-700 transition-colors"
                            title="Copy parameters to clipboard"
                          >
                            Copy
                          </button>
                        </div>
                        <div className="bg-white rounded-md border border-green-200 overflow-hidden">
                          <table className="min-w-full divide-y divide-green-200">
                            <thead className="bg-green-50">
                              <tr>
                                <th className="px-2 py-1 text-left text-xs font-medium text-green-700">Parameter</th>
                                <th className="px-2 py-1 text-right text-xs font-medium text-green-700">Value</th>
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-green-100">
                              {Object.entries(bestExperiment.parameters).map(([key, value]) => (
                                <tr key={key} className="hover:bg-green-50">
                                  <td className="px-2 py-1 text-xs font-medium text-gray-700">{key}</td>
                                  <td className="px-2 py-1 text-right text-xs font-mono font-bold text-green-800">
                                    {String(value)}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>

                      {/* Metrics Section - Table */}
                      {bestExperiment.metrics && Object.keys(bestExperiment.metrics).length > 0 && (
                        <div>
                          <h4 className="text-xs font-medium text-green-700 mb-1.5">Performance Metrics</h4>
                          <div className="bg-white rounded-md border border-green-200 overflow-hidden">
                            <table className="min-w-full divide-y divide-green-200">
                              <thead className="bg-green-50">
                                <tr>
                                  <th className="px-2 py-1 text-left text-xs font-medium text-green-700">Metric</th>
                                  <th className="px-2 py-1 text-right text-xs font-medium text-green-700">Value</th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-green-100">
                                {Object.entries(bestExperiment.metrics)
                                  .filter(([_, value]) => typeof value === 'number')
                                  .map(([key, value]) => (
                                    <tr key={key} className="hover:bg-green-50">
                                      <td className="px-2 py-1 text-xs text-gray-700">{key}</td>
                                      <td className="px-2 py-1 text-right text-xs font-semibold text-gray-900">
                                        {formatMetricValue(value)}
                                      </td>
                                    </tr>
                                  ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Charts */}
                {successfulExperiments.length > 0 && (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Objective Score Bar Chart */}
                    <div className="bg-white border border-gray-200 rounded-lg p-4">
                      <h3 className="text-lg font-semibold text-gray-900 mb-4">
                        Objective Scores by Experiment
                        <span className="ml-3 text-sm font-normal text-gray-500">
                          <span className="inline-block w-3 h-3 bg-green-500 rounded mr-1"></span>
                          Best
                        </span>
                      </h3>
                      <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="name" />
                          <YAxis />
                          <Tooltip
                            content={({ active, payload }) => {
                              if (active && payload && payload.length) {
                                const data = payload[0].payload;
                                const isBest = data.experiment_id === bestExperiment?.experiment_id;
                                const paramDiffs = !isBest && bestExperiment
                                  ? getParameterDiff(data.parameters, bestExperiment.parameters)
                                  : [];

                                return (
                                  <div className="bg-white border border-gray-200 rounded shadow-lg p-3 max-w-sm">
                                    <p className="text-sm font-semibold text-gray-900">{data.name}</p>
                                    <p className="text-sm text-gray-600">
                                      Score: <span className="font-mono">{(payload[0].value as number).toFixed(4)}</span>
                                    </p>
                                    {isBest && (
                                      <p className="text-xs text-green-600 font-semibold mt-1">⭐ Best Experiment</p>
                                    )}
                                    {paramDiffs.length > 0 && (
                                      <div className="mt-2 pt-2 border-t border-gray-200">
                                        <p className="text-xs font-semibold text-gray-700 mb-1">Parameter Differences vs Best:</p>
                                        <div className="space-y-1">
                                          {paramDiffs.map((diff, idx) => (
                                            <p key={idx} className="text-xs text-gray-600 font-mono">
                                              {diff}
                                            </p>
                                          ))}
                                        </div>
                                      </div>
                                    )}
                                  </div>
                                );
                              }
                              return null;
                            }}
                          />
                          <Bar
                            dataKey="objective_score"
                            name="Objective Score"
                            onMouseEnter={(data: any) => {
                              if (data && data.experiment_id) {
                                setHoveredExperiment(data.experiment_id);
                              }
                            }}
                            onMouseLeave={() => {
                              setHoveredExperiment(null);
                            }}
                            label={(props: any) => {
                            const { x, y, width, index } = props;
                            if (x === undefined || y === undefined || width === undefined || index === undefined) return null;
                            const isBest = chartData[index]?.experiment_id === bestExperiment?.experiment_id;
                            if (isBest) {
                              return (
                                <text x={Number(x) + Number(width) / 2} y={Number(y) - 5} fill="#10b981" textAnchor="middle" fontSize={16}>
                                  ⭐
                                </text>
                              );
                            }
                            return null;
                          }}>
                            {chartData.map((entry, index) => (
                              <Cell
                                key={`cell-${index}`}
                                fill={entry.experiment_id === bestExperiment?.experiment_id ? '#10b981' : '#3b82f6'}
                              />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Metrics Comparison */}
                    {(bestExperimentData.length > 0 || hoveredExperimentData.length > 0) && scatterAxisOptions.length > 0 && (
                      <div className="bg-white border border-gray-200 rounded-lg p-4">
                        {/* Title and Legend */}
                        <div className="mb-3">
                          <h3 className="text-lg font-semibold text-gray-900">
                            Performance Metrics - Sub-Rounds
                          </h3>
                          <div className="flex items-center gap-4 mt-2">
                            {bestExperimentData.length > 0 && (
                              <p className="text-xs text-green-600">
                                <span className="inline-block w-3 h-3 bg-green-500 rounded-full mr-1"></span>
                                Best Exp #{bestExperiment?.experiment_id} ({bestExperimentData.length} rounds)
                              </p>
                            )}
                            {hoveredExperimentData.length > 0 && (
                              <p className="text-xs text-blue-600">
                                <span className="inline-block w-3 h-3 bg-blue-500 rounded-full mr-1"></span>
                                Hovered Exp #{hoveredExperiment} ({hoveredExperimentData.length} rounds)
                              </p>
                            )}
                          </div>
                        </div>

                        {/* Axis Controls */}
                        <div className="flex items-center gap-4 mb-4 pb-3 border-b border-gray-200">
                          <div className="flex items-center gap-2">
                            <label className="text-xs font-medium text-gray-700">X-Axis:</label>
                            <select
                              value={scatterXAxis}
                              onChange={(e) => setScatterXAxis(e.target.value)}
                              className="text-xs border border-gray-300 rounded px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-500 max-w-[200px]"
                            >
                              {scatterAxisOptions.map((key) => (
                                <option key={key} value={key}>
                                  {key}
                                </option>
                              ))}
                            </select>
                          </div>
                          <div className="flex items-center gap-2">
                            <label className="text-xs font-medium text-gray-700">Y-Axis:</label>
                            <select
                              value={scatterYAxis}
                              onChange={(e) => setScatterYAxis(e.target.value)}
                              className="text-xs border border-gray-300 rounded px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-500 max-w-[200px]"
                            >
                              {scatterAxisOptions.map((key) => (
                                <option key={key} value={key}>
                                  {key}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>

                        {/* Chart */}
                        <ResponsiveContainer width="100%" height={300}>
                          <ScatterChart key={`${bestExperiment?.id}-${hoveredExperiment}`}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis
                              type="number"
                              dataKey={scatterXAxis}
                              name={scatterXAxis}
                              label={{ value: scatterXAxis, position: 'insideBottom', offset: -5, fontSize: 11 }}
                            />
                            <YAxis
                              type="number"
                              dataKey={scatterYAxis}
                              name={scatterYAxis}
                              label={{ value: scatterYAxis, angle: -90, position: 'insideLeft', fontSize: 11 }}
                            />
                            <ZAxis range={[100, 400]} />
                            <Tooltip
                              content={({ active, payload }) => {
                                if (active && payload && payload.length) {
                                  const data = payload[0].payload;
                                  const isFromBest = payload[0].name === 'Best Experiment';
                                  return (
                                    <div className="bg-white border border-gray-200 rounded shadow-lg p-3 max-w-sm">
                                      <p className="text-sm font-semibold text-gray-900">
                                        {data.round_name} (Concurrency: {data.num_concurrency})
                                      </p>
                                      {isFromBest && (
                                        <p className="text-xs text-green-600 font-semibold">⭐ Best Experiment #{bestExperiment?.experiment_id}</p>
                                      )}
                                      {!isFromBest && (
                                        <p className="text-xs text-blue-600 font-semibold">Experiment #{hoveredExperiment}</p>
                                      )}
                                      <div className="mt-2 space-y-1">
                                        <p className="text-xs text-gray-600">
                                          {scatterXAxis}: <span className="font-mono font-semibold">{data[scatterXAxis]?.toFixed(2)}</span>
                                        </p>
                                        <p className="text-xs text-gray-600">
                                          {scatterYAxis}: <span className="font-mono font-semibold">{data[scatterYAxis]?.toFixed(2)}</span>
                                        </p>
                                        <p className="text-xs text-gray-600">
                                          Scenario: <span className="font-mono">{data.scenario}</span>
                                        </p>
                                        <p className="text-xs text-gray-600">
                                          Requests: <span className="font-mono">{data.num_completed_requests}/{data.num_requests}</span>
                                        </p>
                                        {data.error_rate > 0 && (
                                          <p className="text-xs text-red-600">
                                            Error Rate: <span className="font-mono">{(data.error_rate * 100).toFixed(2)}%</span>
                                          </p>
                                        )}
                                      </div>
                                    </div>
                                  );
                                }
                                return null;
                              }}
                            />
                            <Legend />

                            {/* SLO Reference Lines */}
                            {(() => {
                              const xSLO = getSLOThreshold(scatterXAxis);
                              const ySLO = getSLOThreshold(scatterYAxis);

                              return (
                                <>
                                  {xSLO !== null && (
                                    <ReferenceLine
                                      x={xSLO}
                                      stroke="#ef4444"
                                      strokeWidth={2}
                                      strokeDasharray="5 5"
                                      label={{
                                        value: `SLO: ${xSLO}`,
                                        position: 'top',
                                        fill: '#ef4444',
                                        fontSize: 11,
                                        fontWeight: 'bold',
                                      }}
                                    />
                                  )}
                                  {ySLO !== null && (
                                    <ReferenceLine
                                      y={ySLO}
                                      stroke="#ef4444"
                                      strokeWidth={2}
                                      strokeDasharray="5 5"
                                      label={{
                                        value: `SLO: ${ySLO}`,
                                        position: 'right',
                                        fill: '#ef4444',
                                        fontSize: 11,
                                        fontWeight: 'bold',
                                      }}
                                    />
                                  )}
                                </>
                              );
                            })()}

                            {/* Best experiment data (green dots) */}
                            {bestExperimentData.length > 0 && (
                              <Scatter
                                name="Best Experiment"
                                data={bestExperimentData}
                                fill="#10b981"
                                shape={(props: any) => {
                                  const { cx, cy } = props;
                                  return (
                                    <circle
                                      cx={cx}
                                      cy={cy}
                                      r={6}
                                      fill="#10b981"
                                      stroke="#059669"
                                      strokeWidth={1.5}
                                      opacity={0.8}
                                    />
                                  );
                                }}
                              />
                            )}

                            {/* Hovered experiment data (blue dots) */}
                            {hoveredExperimentData.length > 0 && (
                              <Scatter
                                name="Hovered Experiment"
                                data={hoveredExperimentData}
                                fill="#3b82f6"
                                shape={(props: any) => {
                                  const { cx, cy } = props;
                                  return (
                                    <circle
                                      cx={cx}
                                      cy={cy}
                                      r={6}
                                      fill="#3b82f6"
                                      stroke="#2563eb"
                                      strokeWidth={1.5}
                                      opacity={0.8}
                                    />
                                  );
                                }}
                              />
                            )}
                          </ScatterChart>
                        </ResponsiveContainer>
                      </div>
                    )}
                  </div>
                )}

                {/* Filters and Search */}
                <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div>
                      <label className="text-xs font-medium text-gray-700 mb-1 block">Search</label>
                      <input
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="Search parameters or ID..."
                        className="w-full text-sm border border-gray-300 rounded px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-500"
                      />
                    </div>
                    <div>
                      <label className="text-xs font-medium text-gray-700 mb-1 block">Status</label>
                      <select
                        value={filterStatus}
                        onChange={(e) => setFilterStatus(e.target.value)}
                        className="w-full text-sm border border-gray-300 rounded px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-500"
                      >
                        <option value="all">All</option>
                        <option value="success">Success</option>
                        <option value="failed">Failed</option>
                        <option value="pending">Pending</option>
                        <option value="deploying">Deploying</option>
                        <option value="benchmarking">Benchmarking</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-xs font-medium text-gray-700 mb-1 block">Sort By</label>
                      <select
                        value={sortBy}
                        onChange={(e) => setSortBy(e.target.value as 'id' | 'score' | 'duration')}
                        className="w-full text-sm border border-gray-300 rounded px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-500"
                      >
                        <option value="id">Experiment ID</option>
                        <option value="score">Objective Score</option>
                        <option value="duration">Duration</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-xs font-medium text-gray-700 mb-1 block">Order</label>
                      <select
                        value={sortOrder}
                        onChange={(e) => setSortOrder(e.target.value as 'asc' | 'desc')}
                        className="w-full text-sm border border-gray-300 rounded px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-500"
                      >
                        <option value="asc">Ascending</option>
                        <option value="desc">Descending</option>
                      </select>
                    </div>
                  </div>
                </div>

                {/* Experiments Table */}
                <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
                  <div className="px-4 py-3 bg-gray-50 border-b border-gray-200 flex justify-between items-center">
                    <h3 className="text-lg font-semibold text-gray-900">All Experiments ({filteredExperiments.length})</h3>
                    {selectedExperiments.length > 0 && (
                      <button
                        onClick={() => setViewMode('comparison')}
                        className="text-sm px-3 py-1.5 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
                      >
                        Compare Selected ({selectedExperiments.length})
                      </button>
                    )}
                  </div>
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-4 py-3 text-left">
                            <input
                              type="checkbox"
                              checked={selectedExperiments.length === successfulExperiments.length}
                              onChange={(e) => {
                                if (e.target.checked) {
                                  setSelectedExperiments(successfulExperiments.map(exp => exp.experiment_id));
                                } else {
                                  setSelectedExperiments([]);
                                }
                              }}
                              className="rounded"
                            />
                          </th>
                          <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">ID</th>
                          <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                          <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Parameters</th>
                          <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Objective Score</th>
                          <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Duration</th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {filteredExperiments.map((exp) => (
                          <tr
                            key={exp.id}
                            className={exp.id === task.best_experiment_id ? 'bg-green-50' : ''}
                          >
                            <td className="px-4 py-3">
                              <input
                                type="checkbox"
                                checked={selectedExperiments.includes(exp.experiment_id)}
                                onChange={() => toggleExperimentSelection(exp.experiment_id)}
                                className="rounded"
                                disabled={exp.status !== 'success'}
                              />
                            </td>
                            <td className="px-4 py-3 whitespace-nowrap">
                              <div className="flex items-center">
                                <span className="text-sm font-medium text-gray-900">#{exp.experiment_id}</span>
                                {exp.id === task.best_experiment_id && (
                                  <span className="ml-2 text-xs bg-green-100 text-green-800 px-2 py-0.5 rounded-full">
                                    Best
                                  </span>
                                )}
                              </div>
                            </td>
                            <td className="px-4 py-3 whitespace-nowrap">
                              <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getStatusColor(exp.status)}`}>
                                {exp.status}
                              </span>
                            </td>
                            <td className="px-4 py-3">
                              <div className="text-sm space-y-0.5">
                                {Object.entries(exp.parameters).map(([key, value]) => (
                                  <div key={key} className="flex gap-2">
                                    <span className="text-gray-500">{key}:</span>
                                    <span className="font-mono text-gray-900">{String(value)}</span>
                                  </div>
                                ))}
                              </div>
                            </td>
                            <td className="px-4 py-3 whitespace-nowrap">
                              <span className="text-sm font-mono text-gray-900">
                                {exp.objective_score !== null ? exp.objective_score.toFixed(4) : 'N/A'}
                              </span>
                            </td>
                            <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                              {formatDuration(exp.elapsed_time)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Task Summary */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <div className="text-sm text-blue-600 font-medium">Total Experiments</div>
                    <div className="text-2xl font-bold text-blue-900 mt-1">{task.total_experiments}</div>
                  </div>
                  <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                    <div className="text-sm text-green-600 font-medium">Successful</div>
                    <div className="text-2xl font-bold text-green-900 mt-1">{task.successful_experiments}</div>
                  </div>
                  <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                    <div className="text-sm text-red-600 font-medium">Failed</div>
                    <div className="text-2xl font-bold text-red-900 mt-1">
                      {task.total_experiments - task.successful_experiments}
                    </div>
                  </div>
                  <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                    <div className="text-sm text-purple-600 font-medium">Total Duration</div>
                    <div className="text-2xl font-bold text-purple-900 mt-1">{formatDuration(task.elapsed_time)}</div>
                  </div>
                </div>
              </>
            )}

            {/* Comparison View */}
            {viewMode === 'comparison' && (
              <div className="space-y-6">
                {selectedExperimentObjects.length === 0 ? (
                  <div className="text-center py-12 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
                    <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    <h3 className="mt-2 text-sm font-medium text-gray-900">No experiments selected</h3>
                    <p className="mt-1 text-sm text-gray-500">
                      Select experiments from the Overview tab to compare them side-by-side
                    </p>
                    <button
                      onClick={() => setViewMode('table')}
                      className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                    >
                      Go to Overview
                    </button>
                  </div>
                ) : (
                  <>
                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                      <p className="text-sm text-blue-800">
                        Comparing {selectedExperimentObjects.length} experiment{selectedExperimentObjects.length !== 1 ? 's' : ''}
                      </p>
                    </div>

                    {/* Side-by-side comparison */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {selectedExperimentObjects.map(exp => (
                        <div
                          key={exp.id}
                          className={`border-2 rounded-lg p-4 ${
                            exp.id === task.best_experiment_id
                              ? 'border-green-500 bg-green-50'
                              : 'border-gray-200 bg-white'
                          }`}
                        >
                          <div className="flex items-center justify-between mb-3">
                            <h4 className="text-lg font-bold text-gray-900">Exp #{exp.experiment_id}</h4>
                            {exp.id === task.best_experiment_id && (
                              <span className="text-xs bg-green-600 text-white px-2 py-0.5 rounded-full">Best</span>
                            )}
                          </div>

                          <div className="space-y-3">
                            {/* Objective Score */}
                            <div className="bg-white rounded-md p-3 border">
                              <div className="text-xs text-gray-500 mb-1">Objective Score</div>
                              <div className="text-2xl font-bold text-gray-900">
                                {exp.objective_score?.toFixed(4) || 'N/A'}
                              </div>
                            </div>

                            {/* Parameters */}
                            <div>
                              <div className="text-xs font-medium text-gray-700 mb-1.5">Parameters</div>
                              <div className="bg-white rounded-md border p-2 space-y-1">
                                {Object.entries(exp.parameters).map(([key, value]) => (
                                  <div key={key} className="flex justify-between text-xs">
                                    <span className="text-gray-600">{key}</span>
                                    <span className="font-mono font-semibold text-gray-900">{String(value)}</span>
                                  </div>
                                ))}
                              </div>
                            </div>

                            {/* Key Metrics */}
                            <div>
                              <div className="text-xs font-medium text-gray-700 mb-1.5">Key Metrics</div>
                              <div className="bg-white rounded-md border p-2 space-y-1">
                                {Object.entries(getPrimitiveMetrics(exp.metrics))
                                  .slice(0, 5)
                                  .map(([key, value]) => (
                                    <div key={key} className="flex justify-between text-xs">
                                      <span className="text-gray-600">{key}</span>
                                      <span className="font-mono font-semibold text-gray-900">{value.toFixed(2)}</span>
                                    </div>
                                  ))}
                              </div>
                            </div>

                            {/* Duration */}
                            <div className="bg-white rounded-md p-2 border text-center">
                              <div className="text-xs text-gray-500">Duration</div>
                              <div className="text-sm font-semibold text-gray-900">{formatDuration(exp.elapsed_time)}</div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* Comparison Charts */}
                    {selectedExperimentObjects.length > 1 && (
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* Objective Score Comparison */}
                        <div className="bg-white border border-gray-200 rounded-lg p-4">
                          <h4 className="text-lg font-semibold text-gray-900 mb-4">Objective Score Comparison</h4>
                          <ResponsiveContainer width="100%" height={250}>
                            <BarChart
                              data={selectedExperimentObjects.map(exp => ({
                                name: `Exp ${exp.experiment_id}`,
                                score: exp.objective_score || 0,
                              }))}
                            >
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="name" />
                              <YAxis />
                              <Tooltip />
                              <Bar dataKey="score" fill="#3b82f6" />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>

                        {/* Key Metric Comparison (pick first available metric) */}
                        {allMetricKeys.length > 0 && (
                          <div className="bg-white border border-gray-200 rounded-lg p-4">
                            <h4 className="text-lg font-semibold text-gray-900 mb-4">
                              {allMetricKeys[0]} Comparison
                            </h4>
                            <ResponsiveContainer width="100%" height={250}>
                              <BarChart
                                data={selectedExperimentObjects.map(exp => ({
                                  name: `Exp ${exp.experiment_id}`,
                                  value: getPrimitiveMetrics(exp.metrics)[allMetricKeys[0]] || 0,
                                }))}
                              >
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="name" />
                                <YAxis />
                                <Tooltip />
                                <Bar dataKey="value" fill="#10b981" />
                              </BarChart>
                            </ResponsiveContainer>
                          </div>
                        )}
                      </div>
                    )}
                  </>
                )}
              </div>
            )}

            {/* Sensitivity Analysis View */}
            {viewMode === 'sensitivity' && (
              <div className="space-y-6">
                {allParameterKeys.length === 0 ? (
                  <div className="text-center py-12 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
                    <p className="text-sm text-gray-500">No parameter data available for sensitivity analysis</p>
                  </div>
                ) : (
                  <>
                    {/* Parameter Selection */}
                    <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                      <label className="text-sm font-medium text-gray-700 mb-2 block">
                        Analyze sensitivity for parameter:
                      </label>
                      <select
                        value={sensitivityParam}
                        onChange={(e) => setSensitivityParam(e.target.value)}
                        className="text-sm border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                      >
                        {allParameterKeys.map(key => (
                          <option key={key} value={key}>{key}</option>
                        ))}
                      </select>
                    </div>

                    {/* Sensitivity Charts */}
                    {sensitivityData.length > 0 && (
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* Objective Score vs Parameter */}
                        <div className="bg-white border border-gray-200 rounded-lg p-4">
                          <h4 className="text-lg font-semibold text-gray-900 mb-4">
                            Objective Score vs {sensitivityParam}
                          </h4>
                          <ResponsiveContainer width="100%" height={300}>
                            <LineChart data={sensitivityData}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey={sensitivityParam} />
                              <YAxis />
                              <Tooltip />
                              <Legend />
                              <Line type="monotone" dataKey="avg_objective_score" stroke="#3b82f6" name="Avg Score" strokeWidth={2} />
                              <Line type="monotone" dataKey="min_objective_score" stroke="#ef4444" name="Min" strokeWidth={1} strokeDasharray="5 5" />
                              <Line type="monotone" dataKey="max_objective_score" stroke="#10b981" name="Max" strokeWidth={1} strokeDasharray="5 5" />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>

                        {/* Throughput vs Parameter */}
                        {sensitivityData[0]?.avg_mean_output_throughput_tokens_per_s && (
                          <div className="bg-white border border-gray-200 rounded-lg p-4">
                            <h4 className="text-lg font-semibold text-gray-900 mb-4">
                              Throughput vs {sensitivityParam}
                            </h4>
                            <ResponsiveContainer width="100%" height={300}>
                              <LineChart data={sensitivityData}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey={sensitivityParam} />
                                <YAxis />
                                <Tooltip />
                                <Legend />
                                <Line
                                  type="monotone"
                                  dataKey="avg_mean_output_throughput_tokens_per_s"
                                  stroke="#10b981"
                                  name="Avg Throughput"
                                  strokeWidth={2}
                                />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                        )}

                        {/* TTFT vs Parameter */}
                        {sensitivityData[0]?.avg_ttft_mean && (
                          <div className="bg-white border border-gray-200 rounded-lg p-4">
                            <h4 className="text-lg font-semibold text-gray-900 mb-4">
                              TTFT vs {sensitivityParam}
                            </h4>
                            <ResponsiveContainer width="100%" height={300}>
                              <LineChart data={sensitivityData}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey={sensitivityParam} />
                                <YAxis />
                                <Tooltip />
                                <Legend />
                                <Line
                                  type="monotone"
                                  dataKey="avg_ttft_mean"
                                  stroke="#f59e0b"
                                  name="Avg TTFT"
                                  strokeWidth={2}
                                />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                        )}

                        {/* Latency vs Parameter */}
                        {sensitivityData[0]?.avg_e2e_latency_mean && (
                          <div className="bg-white border border-gray-200 rounded-lg p-4">
                            <h4 className="text-lg font-semibold text-gray-900 mb-4">
                              Latency vs {sensitivityParam}
                            </h4>
                            <ResponsiveContainer width="100%" height={300}>
                              <LineChart data={sensitivityData}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey={sensitivityParam} />
                                <YAxis />
                                <Tooltip />
                                <Legend />
                                <Line
                                  type="monotone"
                                  dataKey="avg_e2e_latency_mean"
                                  stroke="#ef4444"
                                  name="Avg Latency"
                                  strokeWidth={2}
                                />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Sensitivity Summary Table */}
                    <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
                      <div className="px-4 py-3 bg-gray-50 border-b">
                        <h4 className="text-lg font-semibold text-gray-900">Sensitivity Data</h4>
                      </div>
                      <div className="overflow-x-auto">
                        <table className="min-w-full divide-y divide-gray-200">
                          <thead className="bg-gray-50">
                            <tr>
                              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">{sensitivityParam}</th>
                              <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Count</th>
                              <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Avg Score</th>
                              <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Min Score</th>
                              <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Max Score</th>
                            </tr>
                          </thead>
                          <tbody className="bg-white divide-y divide-gray-200">
                            {sensitivityData.map((row, idx) => (
                              <tr key={idx} className="hover:bg-gray-50">
                                <td className="px-4 py-3 text-sm font-medium text-gray-900">{String(row[sensitivityParam])}</td>
                                <td className="px-4 py-3 text-right text-sm text-gray-900">{row.count}</td>
                                <td className="px-4 py-3 text-right text-sm font-mono text-gray-900">
                                  {row.avg_objective_score?.toFixed(4) || 'N/A'}
                                </td>
                                <td className="px-4 py-3 text-right text-sm font-mono text-gray-500">
                                  {row.min_objective_score?.toFixed(4) || 'N/A'}
                                </td>
                                <td className="px-4 py-3 text-right text-sm font-mono text-gray-500">
                                  {row.max_objective_score?.toFixed(4) || 'N/A'}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </>
                )}
              </div>
            )}

            {/* Pareto Frontier View */}
            {viewMode === 'pareto' && (
              <div className="space-y-6">
                {paretoData.length === 0 ? (
                  <div className="text-center py-12 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
                    <p className="text-sm text-gray-500">
                      Insufficient data for Pareto frontier analysis. Need throughput and latency metrics.
                    </p>
                  </div>
                ) : (
                  <>
                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                      <h4 className="text-sm font-semibold text-blue-900 mb-2">Pareto Frontier Analysis</h4>
                      <p className="text-xs text-blue-700">
                        The Pareto frontier shows non-dominated solutions where no other configuration is strictly better in both objectives.
                        Points on the frontier (orange) represent optimal trade-offs between throughput and latency.
                      </p>
                    </div>

                    {/* Pareto Scatter Plot */}
                    <div className="bg-white border border-gray-200 rounded-lg p-4">
                      <h4 className="text-lg font-semibold text-gray-900 mb-4">
                        Throughput vs Latency Trade-off
                      </h4>
                      <ResponsiveContainer width="100%" height={400}>
                        <ScatterChart>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis
                            type="number"
                            dataKey="throughput"
                            name="Throughput"
                            label={{ value: 'Throughput (tokens/s)', position: 'insideBottom', offset: -5 }}
                          />
                          <YAxis
                            type="number"
                            dataKey="latency"
                            name="Latency"
                            label={{ value: 'Latency (s)', angle: -90, position: 'insideLeft' }}
                          />
                          <ZAxis range={[100, 400]} />
                          <Tooltip
                            content={({ active, payload }) => {
                              if (active && payload && payload.length) {
                                const data = payload[0].payload;
                                const onFrontier = paretoFrontier.some(p => p.experiment_id === data.experiment_id);
                                return (
                                  <div className="bg-white border border-gray-200 rounded shadow-lg p-3 max-w-sm">
                                    <p className="text-sm font-semibold text-gray-900">{data.name}</p>
                                    {data.is_best && (
                                      <p className="text-xs text-green-600 font-semibold">⭐ Best Experiment</p>
                                    )}
                                    {onFrontier && (
                                      <p className="text-xs text-orange-600 font-semibold">🎯 Pareto Frontier</p>
                                    )}
                                    <div className="mt-2 space-y-1">
                                      <p className="text-xs text-gray-600">
                                        Throughput: <span className="font-mono font-semibold">{data.throughput.toFixed(2)}</span>
                                      </p>
                                      <p className="text-xs text-gray-600">
                                        Latency: <span className="font-mono font-semibold">{data.latency.toFixed(2)}</span>
                                      </p>
                                      <p className="text-xs text-gray-600">
                                        Score: <span className="font-mono">{data.objective_score?.toFixed(4) || 'N/A'}</span>
                                      </p>
                                    </div>
                                    <div className="mt-2 pt-2 border-t border-gray-200">
                                      <p className="text-xs font-semibold text-gray-700 mb-1">Parameters:</p>
                                      {Object.entries(data.parameters).map(([key, value]) => (
                                        <p key={key} className="text-xs text-gray-600">
                                          {key}: <span className="font-mono">{String(value)}</span>
                                        </p>
                                      ))}
                                    </div>
                                  </div>
                                );
                              }
                              return null;
                            }}
                          />
                          <Legend />

                          {/* All points (blue) */}
                          <Scatter
                            name="All Experiments"
                            data={paretoData}
                            fill="#3b82f6"
                            shape={(props: any) => {
                              const { cx, cy, payload } = props;
                              const onFrontier = paretoFrontier.some(p => p.experiment_id === payload.experiment_id);
                              if (onFrontier) return <></> as any; // Don't render if on frontier
                              return (
                                <circle
                                  cx={cx}
                                  cy={cy}
                                  r={5}
                                  fill="#3b82f6"
                                  stroke="#2563eb"
                                  strokeWidth={1}
                                  opacity={0.6}
                                />
                              );
                            }}
                          />

                          {/* Pareto frontier points (orange) */}
                          <Scatter
                            name="Pareto Frontier"
                            data={paretoFrontier}
                            fill="#f97316"
                            shape={(props: any) => {
                              const { cx, cy } = props;
                              return (
                                <circle
                                  cx={cx}
                                  cy={cy}
                                  r={7}
                                  fill="#f97316"
                                  stroke="#ea580c"
                                  strokeWidth={2}
                                  opacity={0.9}
                                />
                              );
                            }}
                          />

                          {/* Best experiment (green star) */}
                          {paretoData.find(d => d.is_best) && (
                            <Scatter
                              name="Best"
                              data={paretoData.filter(d => d.is_best)}
                              fill="#10b981"
                              shape={(props: any) => {
                                const { cx, cy } = props;
                                return (
                                  <g>
                                    <circle
                                      cx={cx}
                                      cy={cy}
                                      r={10}
                                      fill="#10b981"
                                      stroke="#059669"
                                      strokeWidth={2}
                                      opacity={0.9}
                                    />
                                    <text x={cx} y={cy + 4} textAnchor="middle" fill="white" fontSize={12} fontWeight="bold">
                                      ⭐
                                    </text>
                                  </g>
                                );
                              }}
                            />
                          )}
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Pareto Frontier Table */}
                    <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
                      <div className="px-4 py-3 bg-gray-50 border-b">
                        <h4 className="text-lg font-semibold text-gray-900">
                          Pareto Frontier Solutions ({paretoFrontier.length})
                        </h4>
                      </div>
                      <div className="overflow-x-auto">
                        <table className="min-w-full divide-y divide-gray-200">
                          <thead className="bg-gray-50">
                            <tr>
                              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Exp ID</th>
                              <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Throughput</th>
                              <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Latency</th>
                              <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Obj Score</th>
                              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Parameters</th>
                            </tr>
                          </thead>
                          <tbody className="bg-white divide-y divide-gray-200">
                            {paretoFrontier.map((point) => (
                              <tr key={point.experiment_id} className={point.is_best ? 'bg-green-50' : 'hover:bg-gray-50'}>
                                <td className="px-4 py-3 whitespace-nowrap">
                                  <div className="flex items-center">
                                    <span className="text-sm font-medium text-gray-900">#{point.experiment_id}</span>
                                    {point.is_best && (
                                      <span className="ml-2 text-xs bg-green-100 text-green-800 px-2 py-0.5 rounded-full">
                                        Best
                                      </span>
                                    )}
                                  </div>
                                </td>
                                <td className="px-4 py-3 text-right text-sm font-mono text-gray-900">
                                  {point.throughput.toFixed(2)}
                                </td>
                                <td className="px-4 py-3 text-right text-sm font-mono text-gray-900">
                                  {point.latency.toFixed(2)}
                                </td>
                                <td className="px-4 py-3 text-right text-sm font-mono text-gray-900">
                                  {point.objective_score?.toFixed(4) || 'N/A'}
                                </td>
                                <td className="px-4 py-3 text-sm text-gray-700">
                                  {Object.entries(point.parameters).map(([key, value]) => (
                                    <span key={key} className="mr-2">
                                      {key}={String(value)}
                                    </span>
                                  ))}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </>
                )}
              </div>
            )}

            {/* Close Button */}
            <div className="flex justify-end pt-4 border-t border-gray-200">
              <button
                onClick={onClose}
                className="px-6 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
