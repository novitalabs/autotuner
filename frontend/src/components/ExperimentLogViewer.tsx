import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../services/api';
import { useEscapeKey } from '@/hooks/useEscapeKey';

interface ExperimentLogViewerProps {
  taskId: number;
  experimentId: number;
  onClose: () => void;
}

export default function ExperimentLogViewer({ taskId, experimentId, onClose }: ExperimentLogViewerProps) {
  // Handle Escape key to close modal
  useEscapeKey(onClose);
  const [autoScroll, setAutoScroll] = useState(false);

  // Fetch task logs
  const { data: logData, isLoading } = useQuery({
    queryKey: ['task-logs', taskId],
    queryFn: () => apiClient.getTaskLogs(taskId),
    refetchInterval: 3000, // Refresh every 3 seconds
  });

  // Filter logs for this specific experiment
  const filterExperimentLogs = (logs: string) => {
    if (!logs) return '';

    const lines = logs.split('\n');
    const experimentStatusMarker = `[Experiment ${experimentId}]`;

    // Find the start of this experiment (the header with ====)
    let startIndex = -1;
    let endIndex = lines.length;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];

      // Look for experiment start: "Experiment N" followed by "===="
      if (line.includes(experimentStatusMarker) && line.includes('Status: DEPLOYING')) {
        startIndex = i;
      }

      // Look for next experiment start to mark our end
      if (startIndex !== -1 && i > startIndex) {
        // Check if this is a new experiment starting
        const match = line.match(/\[Experiment (\d+)\].*Status: DEPLOYING/);
        if (match && parseInt(match[1]) !== experimentId) {
          endIndex = i;
          break;
        }
        // Also check for completion
        if (line.includes(experimentStatusMarker) && (line.includes('Status: SUCCESS') || line.includes('Status: FAILED'))) {
          endIndex = i + 1;
          break;
        }
      }
    }

    if (startIndex === -1) {
      return `No logs found for Experiment ${experimentId}.\n\nThis could mean:\n- The experiment hasn't started yet\n- The experiment was skipped\n- Logs are still being written`;
    }

    // Extract all lines between start and end
    const experimentLines = lines.slice(startIndex, endIndex);

    return experimentLines.join('\n');
  };

  const filteredLogs = logData?.logs ? filterExperimentLogs(logData.logs) : '';

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (autoScroll) {
      const logContainer = document.getElementById('experiment-log-container');
      if (logContainer) {
        logContainer.scrollTop = logContainer.scrollHeight;
      }
    }
  }, [filteredLogs, autoScroll]);

  return (
    <div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-5xl w-full max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h2 className="text-xl font-bold text-gray-900">
              Experiment {experimentId} Logs
            </h2>
            <label className="flex items-center gap-2 text-sm text-gray-600">
              <input
                type="checkbox"
                checked={autoScroll}
                onChange={(e) => setAutoScroll(e.target.checked)}
                className="rounded border-gray-300 text-blue-600 focus:ring-emerald-500"
              />
              Auto-scroll
            </label>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-500 transition-colors"
          >
            <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Log Content */}
        <div className="flex-1 flex flex-col p-6 min-h-0">
          {isLoading ? (
            <div className="flex items-center justify-center flex-1">
              <div className="text-center">
                <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-current border-r-transparent"></div>
                <p className="mt-2 text-sm text-gray-600">Loading logs...</p>
              </div>
            </div>
          ) : (
            <div
              id="experiment-log-container"
              className="flex-1 overflow-y-auto bg-gray-900 text-gray-100 p-4 rounded-lg font-mono text-xs"
            >
              <pre className="whitespace-pre-wrap break-words">{filteredLogs || 'No logs available'}</pre>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-200 flex items-center justify-between bg-gray-50">
          <div className="text-sm text-gray-600">
            Showing logs for Experiment {experimentId} from Task ID {taskId}
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => {
                const logContainer = document.getElementById('experiment-log-container');
                if (logContainer) {
                  logContainer.scrollTop = logContainer.scrollHeight;
                }
              }}
              className="px-3 py-1 text-sm border border-gray-300 rounded-md hover:bg-gray-100 transition-colors"
            >
              Scroll to Bottom
            </button>
            <button
              onClick={onClose}
              className="px-4 py-2 bg-emerald-500 text-white rounded-md hover:bg-emerald-600 transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
