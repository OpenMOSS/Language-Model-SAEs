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

// 全局状态管理（跨组件共享）
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

  // 订阅全局状态变化
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

  // 通知父组件状态变化
  useEffect(() => {
    onLoadingStateChange?.(isLoading, isLoaded);
  }, [isLoading, isLoaded, onLoadingStateChange]);

  // 自动滚动到底部
  useEffect(() => {
    if (logsContainerRef.current) {
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight;
    }
  }, [loadingLogs]);

  // 获取加载日志
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
      console.error('获取加载日志出错:', error);
    }
    return [];
  }, [modelName]);

  // 检查模型加载状态
  const checkLoadingStatus = useCallback(async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/circuit_trace/status`);
      if (response.ok) {
        const data = await response.json();
        return data.available === true;
      }
    } catch (error) {
      console.error('检查加载状态出错:', error);
    }
    return false;
  }, []);

  // 预加载模型
  const preloadModels = useCallback(async () => {
    if (globalLoadingState.isLoading) {
      console.log('⏳ 模型正在加载中，跳过重复请求');
      return;
    }

    setError(null);
    globalLoadingState.isLoading = true;
    globalLoadingState.logs = [];
    setIsLoading(true);
    setLoadingLogs([]);
    notifyListeners();

    try {
      console.log('🔍 开始预加载 Transcoders 和 LoRSAs:', modelName);

      // 开始轮询日志
      pollIntervalRef.current = setInterval(async () => {
        await fetchLoadingLogs();
      }, 500);

      // 发送预加载请求
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/circuit/preload_models`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model_name: modelName }),
      });

      // 等待一段时间获取最后的日志
      await new Promise(resolve => setTimeout(resolve, 1000));
      await fetchLoadingLogs();

      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }

      if (response.ok) {
        const data = await response.json();
        console.log('✅ 预加载完成:', data);
        globalLoadingState.isLoaded = true;
        setIsLoaded(true);
      } else {
        const errorText = await response.text();
        console.error('❌ 预加载失败:', errorText);
        setError(errorText);
      }
    } catch (error) {
      console.error('❌ 预加载出错:', error);
      setError(error instanceof Error ? error.message : '未知错误');
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

  // 自动预加载
  useEffect(() => {
    if (autoPreload && !globalLoadingState.isLoaded && !globalLoadingState.isLoading) {
      // 先检查是否已经加载
      const checkAndLoad = async () => {
        // 避免频繁检查
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

  // 清理轮询
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  // 获取状态图标
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

  // 获取状态文本
  const getStatusText = () => {
    if (isLoading) {
      return '加载中...';
    }
    if (isLoaded) {
      return 'TC/LoRSA 已就绪';
    }
    if (error) {
      return '加载失败';
    }
    return 'TC/LoRSA 未加载';
  };

  // 渲染日志列表
  const renderLogs = () => (
    <div
      ref={logsContainerRef}
      className="max-h-64 overflow-y-auto bg-gray-900 text-gray-100 rounded-lg p-3 font-mono text-xs space-y-1"
    >
      {loadingLogs.length === 0 ? (
        <div className="text-gray-500 text-center py-4">暂无加载日志</div>
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

  // 如果只显示按钮
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
                <span>模型加载状态</span>
              </DialogTitle>
            </DialogHeader>

            <div className="space-y-4">
              {/* 状态卡片 */}
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
                          {isLoaded ? '重新加载' : '开始加载'}
                        </Button>
                      )}
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={fetchLoadingLogs}
                      >
                        刷新日志
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* 错误信息 */}
              {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-red-700 text-sm">
                  {error}
                </div>
              )}

              {/* 加载日志 */}
              <Card>
                <CardHeader className="py-3">
                  <CardTitle className="text-sm flex items-center justify-between">
                    <span>加载日志 ({loadingLogs.length})</span>
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

              {/* 说明 */}
              <div className="text-xs text-gray-500 space-y-1">
                <p>• Transcoders (TC) 和 LoRSAs 是 Circuit Trace 分析所需的模型组件</p>
                <p>• 首次加载可能需要几分钟，加载完成后会被缓存</p>
                <p>• 此加载状态在 Play Game 和 Search Circuits 页面之间共享</p>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </>
    );
  }

  // 渲染完整卡片
  return (
    <Card>
      <CardHeader className="py-3">
        <CardTitle className="text-sm flex items-center justify-between">
          <div className="flex items-center space-x-2">
            {getStatusIcon()}
            <span>模型加载状态</span>
          </div>
          <div className="flex space-x-2">
            {!isLoading && (
              <Button
                variant="outline"
                size="sm"
                onClick={preloadModels}
              >
                <RefreshCw className="w-4 h-4 mr-1" />
                {isLoaded ? '重新加载' : '开始加载'}
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
            <span className="text-gray-500">模型:</span>
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

// 导出一个 hook 用于获取加载状态
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

