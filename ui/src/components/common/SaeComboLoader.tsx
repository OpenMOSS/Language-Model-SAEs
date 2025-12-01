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

  // 拉取可用组合信息
  useEffect(() => {
    const fetchCombos = async () => {
      try {
        const res = await fetch("/sae/combos");
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
        setLoadedId(data.current_id ?? initialId);
      } catch (err) {
        console.error("Failed to fetch SAE combos:", err);
      }
    };
    fetchCombos();
  }, []);

  // 轮询日志
  useEffect(() => {
    if (!isLoading || !selectedId) return;

    let cancelled = false;

    const poll = async () => {
      try {
        const params = new URLSearchParams({
          model_name: "lc0/BT4-1024x15x32h",
          sae_combo_id: selectedId,
        });
        const res = await fetch(`/circuit/loading_logs?${params.toString()}`);
        if (!res.ok) return;
        const data: LoadingLogsResponse = await res.json();
        if (!cancelled) {
          setLogs(data.logs ?? []);
        }
      } catch (err) {
        console.error("Failed to fetch loading logs:", err);
      }
    };

    poll();
    const timer = window.setInterval(poll, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [isLoading, selectedId]);

  const handleReload = useCallback(async () => {
    if (!selectedId) return;
    setIsLoading(true);
    try {
      const body = {
        model_name: "lc0/BT4-1024x15x32h",
        sae_combo_id: selectedId,
      };
      const res = await fetch("/circuit/preload_models", {
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
      setLoadedId(data.sae_combo_id);
      setCurrentServerId(data.sae_combo_id);
      window.localStorage.setItem(LOCAL_STORAGE_KEY, data.sae_combo_id);
    } catch (err) {
      console.error("Failed to preload SAE models:", err);
    } finally {
      setIsLoading(false);
    }
  }, [selectedId]);

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
            onValueChange={(value) => setSelectedId(value)}
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


