import React, { useEffect, useState, useCallback } from "react";

import { Button } from "@/components/ui/button";
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from "@/components/ui/select";

interface SaeCombo {
  id: string;
  label: string;
  tc_base_path: string;
  lorsa_base_path: string;
}

interface SaeComboLoaderProps {
  title?: string;
  className?: string;
}

interface PreloadResponse {
  status: string;
  message: string;
  model_name: string;
  sae_combo_id: string;
}

interface LoadingLogsResponse {
  model_name: string;
  sae_combo_id: string;
  logs: { timestamp: number; message: string }[];
  total_count: number;
  is_loaded?: boolean;  // 新增：后端实际缓存检查结果
}

const LOCAL_STORAGE_KEY = "bt4_sae_combo_id";

export const SaeComboLoader: React.FC<SaeComboLoaderProps> = ({ title, className }) => {
  const [combos, setCombos] = useState<SaeCombo[]>([]);
  const [defaultId, setDefaultId] = useState<string | null>(null);
  const [currentServerId, setCurrentServerId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loadedId, setLoadedId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [logs, setLogs] = useState<{ timestamp: number; message: string }[]>([]);

  // 前端展示的日志条数上限（只保留最近 N 条，避免面板无限增长）
  const MAX_VISIBLE_LOGS = 200;

  const backendBase = import.meta.env.VITE_BACKEND_URL ?? "";

  // 拉取可用组合信息
  useEffect(() => {
    const fetchCombos = async () => {
      try {
        const res = await fetch(`${backendBase}/sae/combos`);
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const data = await res.json();
        const backendCombos: SaeCombo[] = data.combos ?? [];
        setCombos(backendCombos);
        setDefaultId(data.default_id ?? null);
        setCurrentServerId(data.current_id ?? null);

        const stored = window.localStorage.getItem(LOCAL_STORAGE_KEY);
        const initialId =
          (stored && backendCombos.some((c) => c.id === stored) && stored) ||
          data.current_id ||
          data.default_id ||
          (backendCombos.length > 0 ? backendCombos[0].id : null);

        setSelectedId(initialId);
        
        // 立即获取该组合的历史日志（跨页面共享），并验证是否真的已加载
        if (initialId) {
          try {
            const logParams = new URLSearchParams({
              model_name: "lc0/BT4-1024x15x32h",
              sae_combo_id: initialId,
            });
            const logRes = await fetch(`${backendBase}/circuit/loading_logs?${logParams.toString()}`);
            if (logRes.ok) {
              const logData: LoadingLogsResponse & { is_loading?: boolean } = await logRes.json();
              const allLogs = logData.logs ?? [];
              const sliced = allLogs.slice(-MAX_VISIBLE_LOGS);
              setLogs(sliced);
              
              // 同步加载状态
              if (logData.is_loading) {
                setIsLoading(true);
              } else {
                setIsLoading(false);
              }
              
              // 使用后端返回的 is_loaded 字段来判断是否真的已加载（实际缓存检查）
              // 这是最可靠的判断方式，因为后端会实际检查缓存是否存在
              if (logData.is_loaded === true && !logData.is_loading) {
                // 确认已加载，设置 loadedId
                setLoadedId(initialId);
                console.log('✅ 确认 SAE 组合已加载（缓存验证）:', initialId);
              } else {
                // 未加载或状态不明确，不设置 loadedId
                setLoadedId(null);
                console.log('⚠️ SAE 组合未加载或状态不明确:', initialId, {
                  is_loaded: logData.is_loaded,
                  is_loading: logData.is_loading
                });
              }
            } else {
              // 如果获取日志失败，不设置 loadedId
              setLoadedId(null);
            }
          } catch (logErr) {
            console.warn("Failed to fetch initial loading logs:", logErr);
            // 获取日志失败时，不设置 loadedId
            setLoadedId(null);
          }
        } else {
          setLoadedId(null);
        }
      } catch (err) {
        console.error("Failed to fetch SAE combos:", err);
      }
    };
    fetchCombos();
  }, [backendBase]);

  // 轮询日志（即使不在加载中也要轮询，以便显示历史日志和实时更新）
  useEffect(() => {
    if (!selectedId) return;

    let cancelled = false;

    const poll = async () => {
      try {
        const params = new URLSearchParams({
          model_name: "lc0/BT4-1024x15x32h",
          sae_combo_id: selectedId,
        });
        const res = await fetch(`${backendBase}/circuit/loading_logs?${params.toString()}`);
        if (!res.ok) return;
        const data: LoadingLogsResponse & { is_loading?: boolean } = await res.json();
        if (!cancelled) {
          const allLogs = data.logs ?? [];
          // 只保留最近 MAX_VISIBLE_LOGS 条
          const sliced = allLogs.slice(-MAX_VISIBLE_LOGS);
          setLogs(sliced);
          // 如果后端显示正在加载，但前端状态不是，更新前端状态
          if (data.is_loading && !isLoading) {
            setIsLoading(true);
          } else if (!data.is_loading && isLoading) {
            setIsLoading(false);
          }
          // 同步 loadedId 状态（基于实际缓存检查）
          if (data.is_loaded === true && !data.is_loading) {
            // 确认已加载，设置 loadedId
            if (selectedId && selectedId !== loadedId) {
              setLoadedId(selectedId);
            }
          } else if (data.is_loaded === false) {
            // 如果缓存不存在，且当前选中的组合被认为是已加载的，清除 loadedId
            if (loadedId === selectedId) {
              setLoadedId(null);
            }
          }
        }
      } catch (err) {
        console.error("Failed to fetch loading logs:", err);
      }
    };

    // 立即执行一次，然后开始轮询（这样切换页面时能立即看到日志）
    poll();
    const timer = window.setInterval(poll, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [selectedId, isLoading, backendBase]);

  const handleCancel = useCallback(async () => {
    try {
      const body = {
        model_name: "lc0/BT4-1024x15x32h",
        sae_combo_id: loadedId || selectedId,
      };
      await fetch(`${backendBase}/circuit/cancel_loading`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      });
    } catch (err) {
      console.error("Failed to cancel loading:", err);
    }
  }, [loadedId, selectedId, backendBase]);

  const handleReload = useCallback(async () => {
    if (!selectedId) return;
    
    // 如果当前有其他组合正在加载，先中断它
    if (loadedId && loadedId !== selectedId && isLoading) {
      try {
        await fetch(`${backendBase}/circuit/cancel_loading`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model_name: "lc0/BT4-1024x15x32h",
            sae_combo_id: loadedId,
          }),
        });
        // 等待一小段时间让中断生效
        await new Promise((resolve) => setTimeout(resolve, 500));
      } catch (err) {
        console.warn("Failed to cancel previous loading:", err);
      }
    }
    
    // 每次重新加载前清空本地日志缓存，避免遗留旧组合的日志
    setLogs([]);
    setIsLoading(true);
    
    try {
      const body = {
        model_name: "lc0/BT4-1024x15x32h",
        sae_combo_id: selectedId,
      };
      const res = await fetch(`${backendBase}/circuit/preload_models`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`HTTP ${res.status}: ${text}`);
      }
      const data: PreloadResponse = await res.json();
      setCurrentServerId(data.sae_combo_id);
      window.localStorage.setItem(LOCAL_STORAGE_KEY, data.sae_combo_id);
      // 注意：不在这里直接设置 loadedId 和 isLoading
      // 让轮询机制通过检查后端的 is_loading 和 is_loaded 字段来更新状态
      // 这样可以确保只有在实际缓存存在时才显示"已加载"
    } catch (err) {
      console.error("Failed to preload SAE models:", err);
      // 加载失败时，清除状态
      setIsLoading(false);
      if (loadedId === selectedId) {
        setLoadedId(null);
      }
    }
    // 注意：不在 finally 中设置 isLoading = false
    // 因为加载是异步的，轮询机制会根据后端的 is_loading 字段来更新状态
  }, [selectedId, loadedId, isLoading, backendBase]);

  const canReload = selectedId != null && selectedId !== loadedId && !isLoading;

  if (combos.length === 0) {
    return null;
  }

  return (
    <div
      className={`mb-4 rounded-md border border-blue-200 bg-blue-50 p-3 text-sm text-blue-900 ${className ?? ""}`}
    >
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-col gap-1">
          <div className="font-semibold">
            {title ?? "BT4 SAE 组合选择（LoRSA / Transcoder）"}
          </div>
          <div className="text-xs text-blue-800">
            当前服务端组合：
            <span className="font-mono">
              {currentServerId ?? defaultId ?? loadedId ?? selectedId ?? "未知"}
            </span>
          </div>
        </div>
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
          <Select
            value={selectedId ?? undefined}
            onValueChange={async (value) => {
              setSelectedId(value);
              // 切换组合时立即获取该组合的日志
              try {
                const params = new URLSearchParams({
                  model_name: "lc0/BT4-1024x15x32h",
                  sae_combo_id: value,
                });
                const res = await fetch(`${backendBase}/circuit/loading_logs?${params.toString()}`);
                if (res.ok) {
                  const data: LoadingLogsResponse & { is_loading?: boolean } = await res.json();
                  const allLogs = data.logs ?? [];
                  const sliced = allLogs.slice(-MAX_VISIBLE_LOGS);
                  setLogs(sliced);
                  if (data.is_loading) {
                    setIsLoading(true);
                  } else {
                    setIsLoading(false);
                  }
                  // 同步 loadedId 状态（基于实际缓存检查）
                  if (data.is_loaded === true && !data.is_loading) {
                    setLoadedId(value);
                  } else {
                    setLoadedId(null);
                  }
                }
              } catch (err) {
                console.warn("Failed to fetch logs when switching combo:", err);
              }
            }}
          >
            <SelectTrigger className="w-56 bg-white">
              <SelectValue placeholder="选择 SAE 组合" />
            </SelectTrigger>
            <SelectContent>
              {combos.map((c) => (
                <SelectItem key={c.id} value={c.id}>
                  {c.label}{" "}
                  {c.id === defaultId ? "(默认)" : undefined}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            type="button"
            size="sm"
            disabled={!canReload}
            onClick={handleReload}
            className="whitespace-nowrap"
          >
            {isLoading
              ? "加载中..."
              : loadedId === selectedId
              ? "已加载"
              : "加载 / 重新加载"}
          </Button>
          {isLoading && (
            <Button
              type="button"
              size="sm"
              variant="destructive"
              onClick={handleCancel}
              className="whitespace-nowrap"
            >
              中断加载
            </Button>
          )}
        </div>
      </div>
      <div className="mt-2 max-h-40 overflow-y-auto rounded bg-blue-100 p-2 text-xs font-mono leading-relaxed">
        {logs.length === 0 ? (
          <div className="text-blue-700 opacity={0.8}">
            暂无加载日志。请选择组合并点击“加载 / 重新加载”开始加载 LoRSA / Transcoder。
          </div>
        ) : (
          logs.map((log, idx) => (
            <div key={`${log.timestamp}-${idx}`}>
              {new Date(log.timestamp * 1000).toLocaleTimeString()} - {log.message}
            </div>
          ))
        )}
      </div>
    </div>
  );
};


