import React, { useState, useCallback, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Loader2, CheckCircle, XCircle, RefreshCw, ChevronDown, ChevronUp, Cpu } from 'lucide-react';

interface LoadingLog {
  timestamp: number;
  message: string;
}

interface ModelLoadingStatusProps {
  modelName?: string;
  showButton?: boolean;
  buttonVariant?: 'default' | 'outline' | 'ghost';
  buttonSize?: 'default' | 'sm' | 'lg';
  autoPreload?: boolean;
  onLoadingStateChange?: (isLoading: boolean, isLoaded: boolean) => void;
}

// Global state management (shared across components)
let globalLoadingState = {
  isLoading: false,
  isLoaded: false,
  logs: [] as LoadingLog[],
  lastCheckTime: 0,
};

const globalListeners: Set<() => void> = new Set();

const notifyListeners = () => {
  globalListeners.forEach(listener => listener());
};

export const ModelLoadingStatus: React.FC<ModelLoadingStatusProps> = ({
  modelName = 'lc0/BT4-1024x15x32h',
  showButton = true,
  buttonVariant = 'outline',
  buttonSize = 'sm',
  autoPreload = false,
  onLoadingStateChange,
}) => {
  const [isLoading, setIsLoading] = useState(globalLoadingState.isLoading);
  const [isLoaded, setIsLoaded] = useState(globalLoadingState.isLoaded);
  const [loadingLogs, setLoadingLogs] = useState<LoadingLog[]>(globalLoadingState.logs);
  const [showLogsDialog, setShowLogsDialog] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const logsContainerRef = useRef<HTMLDivElement>(null);

  // Subscribe to global state changes
  useEffect(() => {
    const listener = () => {
      setIsLoading(globalLoadingState.isLoading);
      setIsLoaded(globalLoadingState.isLoaded);
      setLoadingLogs([...globalLoadingState.logs]);
    };
    globalListeners.add(listener);
    return () => {
      globalListeners.delete(listener);
    };
  }, []);

  // Notify parent components when loading state changes
  useEffect(() => {
    onLoadingStateChange?.(isLoading, isLoaded);
  }, [isLoading, isLoaded, onLoadingStateChange]);

  // Automatically scroll logs to the bottom
  useEffect(() => {
    if (logsContainerRef.current) {
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight;
    }
  }, [loadingLogs]);

  // Fetch loading logs from backend
  const fetchLoadingLogs = useCallback(async () => {
    try {
      const url = `${import.meta.env.VITE_BACKEND_URL}/circuit/loading_logs?model_name=${encodeURIComponent(modelName)}`;
      const response = await fetch(url);
      if (response.ok) {
        const data = await response.json();
        globalLoadingState.logs = data.logs || [];
        setLoadingLogs(data.logs || []);
        notifyListeners();
        return data.logs || [];
      }
    } catch (error) {
      console.error('Failed to fetch loading logs:', error);
    }
    return [];
  }, [modelName]);

  // Check whether the model is already available
  const checkLoadingStatus = useCallback(async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/circuit_trace/status`);
      if (response.ok) {
        const data = await response.json();
        return data.available === true;
      }
    } catch (error) {
      console.error('Failed to check loading status:', error);
    }
    return false;
  }, []);

  // Preload models
  const preloadModels = useCallback(async () => {
    if (globalLoadingState.isLoading) {
      console.log('⏳ Model is already loading, skipping duplicate request.');
      return;
    }

    setError(null);
    globalLoadingState.isLoading = true;
    globalLoadingState.logs = [];
    setIsLoading(true);
    setLoadingLogs([]);
    notifyListeners();

    try {
      console.log('🔍 Start preloading Transcoders and Lorsas for model:', modelName);

      // Start polling logs
      pollIntervalRef.current = setInterval(async () => {
        await fetchLoadingLogs();
      }, 500);

      // Send preload request
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/circuit/preload_models`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model_name: modelName }),
      });

      // Wait a bit to fetch final logs
      await new Promise(resolve => setTimeout(resolve, 1000));
      await fetchLoadingLogs();

      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }

      if (response.ok) {
        const data = await response.json();
        console.log('✅ Preload completed:', data);
        globalLoadingState.isLoaded = true;
        setIsLoaded(true);
      } else {
        const errorText = await response.text();
        console.error('❌ Preload failed:', errorText);
        setError(errorText);
      }
    } catch (error) {
      console.error('❌ Preload error:', error);
      setError(error instanceof Error ? error.message : 'Unknown error');
    } finally {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
      globalLoadingState.isLoading = false;
      setIsLoading(false);
      notifyListeners();
    }
  }, [modelName, fetchLoadingLogs]);

  // Auto-preload when enabled
  useEffect(() => {
    if (autoPreload && !globalLoadingState.isLoaded && !globalLoadingState.isLoading) {
      // Check first whether the model is already loaded
      const checkAndLoad = async () => {
        // Avoid frequent status checks
        const now = Date.now();
        if (now - globalLoadingState.lastCheckTime < 5000) {
          return;
        }
        globalLoadingState.lastCheckTime = now;

        const isAvailable = await checkLoadingStatus();
        if (isAvailable) {
          globalLoadingState.isLoaded = true;
          setIsLoaded(true);
          notifyListeners();
        } else {
          preloadModels();
        }
      };
      checkAndLoad();
    }
  }, [autoPreload, checkLoadingStatus, preloadModels]);

  // Cleanup polling interval
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  // Get status icon
  const getStatusIcon = () => {
    if (isLoading) {
      return <Loader2 className="w-4 h-4 animate-spin text-blue-500" />;
    }
    if (isLoaded) {
      return <CheckCircle className="w-4 h-4 text-green-500" />;
    }
    if (error) {
      return <XCircle className="w-4 h-4 text-red-500" />;
    }
    return <Cpu className="w-4 h-4 text-gray-400" />;
  };

  // Get user-facing status text
  const getStatusText = () => {
    if (isLoading) {
      return 'Loading...';
    }
    if (isLoaded) {
      return 'TC/Lorsa is ready';
    }
    if (error) {
      return 'Load failed';
    }
    return 'TC/Lorsa not loaded';
  };

  // Render logs list
  const renderLogs = () => (
    <div
      ref={logsContainerRef}
      className="max-h-64 overflow-y-auto bg-gray-900 text-gray-100 rounded-lg p-3 font-mono text-xs space-y-1"
    >
      {loadingLogs.length === 0 ? (
        <div className="text-gray-500 text-center py-4">No loading logs yet.</div>
      ) : (
        loadingLogs.map((log, idx) => (
          <div key={idx} className="flex">
            <span className="text-gray-500 mr-2 flex-shrink-0">
              [{new Date(log.timestamp * 1000).toLocaleTimeString()}]
            </span>
            <span className={
              log.message.includes('✅') ? 'text-green-400' :
              log.message.includes('❌') ? 'text-red-400' :
              log.message.includes('⚠️') ? 'text-yellow-400' :
              log.message.includes('🔍') ? 'text-blue-400' :
              'text-gray-100'
            }>
              {log.message}
            </span>
          </div>
        ))
      )}
    </div>
  );

  // Button-only mode: show a compact status button and open dialog on click
  if (showButton) {
    return (
      <>
        <Button
          variant={buttonVariant}
          size={buttonSize}
          onClick={() => setShowLogsDialog(true)}
          className="flex items-center space-x-2"
        >
          {getStatusIcon()}
          <span>{getStatusText()}</span>
        </Button>

        <Dialog open={showLogsDialog} onOpenChange={setShowLogsDialog}>
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle className="flex items-center space-x-2">
                <Cpu className="w-5 h-5" />
                <span>Model Loading Status</span>
              </DialogTitle>
            </DialogHeader>

            <div className="space-y-4">
              {/* Status card */}
              <Card>
                <CardContent className="pt-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      {getStatusIcon()}
                      <div>
                        <div className="font-medium">{getStatusText()}</div>
                        <div className="text-sm text-gray-500">{modelName}</div>
                      </div>
                    </div>
                    <div className="flex space-x-2">
                      {!isLoading && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={preloadModels}
                        >
                          <RefreshCw className="w-4 h-4 mr-1" />
                          {isLoaded ? 'Reload models' : 'Start loading'}
                        </Button>
                      )}
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={fetchLoadingLogs}
                      >
                        Refresh logs
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Error display */}
              {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-red-700 text-sm">
                  {error}
                </div>
              )}

              {/* Loading logs */}
              <Card>
                <CardHeader className="py-3">
                  <CardTitle className="text-sm flex items-center justify-between">
                    <span>Loading logs ({loadingLogs.length})</span>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setIsCollapsed(!isCollapsed)}
                    >
                      {isCollapsed ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
                    </Button>
                  </CardTitle>
                </CardHeader>
                {!isCollapsed && (
                  <CardContent className="pt-0">
                    {renderLogs()}
                  </CardContent>
                )}
              </Card>

              {/* Helper text */}
              <div className="text-xs text-gray-500 space-y-1">
                <p>• Transcoders (TC) and Lorsas are model components required for circuit trace analysis.</p>
                <p>• The first load may take a few minutes; once loaded, the models are cached.</p>
                <p>• This loading status is shared between the Play Game and Search Circuits pages.</p>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </>
    );
  }

  // Full card mode
  return (
    <Card>
      <CardHeader className="py-3">
        <CardTitle className="text-sm flex items-center justify-between">
          <div className="flex items-center space-x-2">
            {getStatusIcon()}
            <span>Model Loading Status</span>
          </div>
          <div className="flex space-x-2">
            {!isLoading && (
              <Button
                variant="outline"
                size="sm"
                onClick={preloadModels}
              >
                <RefreshCw className="w-4 h-4 mr-1" />
                {isLoaded ? 'Reload models' : 'Start loading'}
              </Button>
            )}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsCollapsed(!isCollapsed)}
            >
              {isCollapsed ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
            </Button>
          </div>
        </CardTitle>
      </CardHeader>
      {!isCollapsed && (
        <CardContent className="pt-0 space-y-3">
          <div className="text-sm">
            <span className="text-gray-500">Model:</span>
            <span className="ml-2 font-mono">{modelName}</span>
          </div>
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-2 text-red-700 text-xs">
              {error}
            </div>
          )}
          {renderLogs()}
        </CardContent>
      )}
    </Card>
  );
};

// Hook to consume the shared model loading state
export const useModelLoadingStatus = () => {
  const [isLoading, setIsLoading] = useState(globalLoadingState.isLoading);
  const [isLoaded, setIsLoaded] = useState(globalLoadingState.isLoaded);

  useEffect(() => {
    const listener = () => {
      setIsLoading(globalLoadingState.isLoading);
      setIsLoaded(globalLoadingState.isLoaded);
    };
    globalListeners.add(listener);
    return () => {
      globalListeners.delete(listener);
    };
  }, []);

  return { isLoading, isLoaded };
};

export default ModelLoadingStatus;

