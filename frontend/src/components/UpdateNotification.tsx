/**
 * UpdateNotification Component
 *
 * Displays a notification banner when a new version is available.
 * Features:
 * - Checks for updates on mount
 * - Periodic checking (every 60 minutes)
 * - Dismissible notification
 * - Shows version comparison and release notes
 * - Links to GitHub release page
 */

import { useState, useEffect } from 'react';
import { checkForUpdates, type UpdateInfo } from '@/services/version';

interface UpdateNotificationProps {
  githubRepo: string;
}

export function UpdateNotification({ githubRepo }: UpdateNotificationProps) {
  const [updateInfo, setUpdateInfo] = useState<UpdateInfo | null>(null);
  const [dismissed, setDismissed] = useState(false);
  const [showFullNotes, setShowFullNotes] = useState(false);

  // Check for updates on mount
  useEffect(() => {
    if (!githubRepo || githubRepo === 'your-org/inference-autotuner' || githubRepo === 'novitalabs/inference-autotuner') {
      // Skip check if repo not configured or using placeholder
      return;
    }

    checkForUpdates(githubRepo).then(setUpdateInfo);
  }, [githubRepo]);

  // Periodic checking (every 60 minutes)
  useEffect(() => {
    if (!githubRepo || githubRepo === 'your-org/inference-autotuner' || githubRepo === 'novitalabs/inference-autotuner') {
      return;
    }

    const interval = setInterval(() => {
      checkForUpdates(githubRepo).then(setUpdateInfo);
    }, 60 * 60 * 1000); // 60 minutes

    return () => clearInterval(interval);
  }, [githubRepo]);

  // Don't show notification if no update available or dismissed
  if (!updateInfo?.hasUpdate || dismissed) {
    return null;
  }

  // Truncate release notes for preview
  const previewLength = 200;
  const releaseNotes = updateInfo.releaseNotes || '';
  const needsTruncation = releaseNotes.length > previewLength;
  const displayNotes = showFullNotes
    ? releaseNotes
    : releaseNotes.substring(0, previewLength) + (needsTruncation ? '...' : '');

  return (
    <div className="fixed top-16 right-4 z-40 max-w-md">
      <div className="bg-blue-50 border-l-4 border-blue-500 rounded-lg shadow-lg p-4">
        <div className="flex items-start">
          {/* Icon */}
          <div className="flex-shrink-0">
            <svg
              className="w-6 h-6 text-blue-600"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>

          {/* Content */}
          <div className="ml-3 flex-1">
            <h3 className="text-sm font-medium text-blue-900">
              New Version Available
            </h3>

            <div className="mt-2 text-sm text-blue-800">
              <p>
                Version <strong>{updateInfo.latestVersion}</strong> is now available.
                You are currently running version <strong>{updateInfo.currentVersion}</strong>.
              </p>

              {releaseNotes && (
                <div className="mt-2">
                  <p className="font-medium mb-1">What's new:</p>
                  <div className="bg-white bg-opacity-50 rounded p-2 text-xs whitespace-pre-wrap">
                    {displayNotes}
                  </div>
                  {needsTruncation && (
                    <button
                      onClick={() => setShowFullNotes(!showFullNotes)}
                      className="mt-1 text-xs text-blue-700 hover:text-blue-900 font-medium"
                    >
                      {showFullNotes ? 'Show less' : 'Show more'}
                    </button>
                  )}
                </div>
              )}
            </div>

            {/* Actions */}
            <div className="mt-3 flex gap-2">
              <a
                href={updateInfo.downloadUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center px-3 py-1.5 text-xs font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 transition-colors"
              >
                View Release
                <svg
                  className="w-3 h-3 ml-1"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                  />
                </svg>
              </a>

              <button
                onClick={() => setDismissed(true)}
                className="inline-flex items-center px-3 py-1.5 text-xs font-medium text-blue-700 bg-blue-100 rounded-md hover:bg-blue-200 transition-colors"
              >
                Dismiss
              </button>
            </div>
          </div>

          {/* Close button */}
          <button
            onClick={() => setDismissed(true)}
            className="ml-2 flex-shrink-0 text-blue-400 hover:text-blue-600"
          >
            <svg
              className="w-5 h-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}
