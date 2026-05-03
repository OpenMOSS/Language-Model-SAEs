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

const LOCAL_STORAGE_KEY = "bt4_sae_combo_id";

export const SaeComboLoader: React.FC<SaeComboLoaderProps> = ({ title, className }) => {
  const [combos, setCombos] = useState<SaeCombo[]>([]);
  const [defaultId, setDefaultId] = useState<string | null>(null);
  const [currentServerId, setCurrentServerId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loadedId, setLoadedId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const backendBase = import.meta.env.VITE_BACKEND_URL ?? "";

  const fetchCombos = useCallback(async (preserveSelection: boolean = true) => {
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

      setSelectedId((prev) => (preserveSelection && prev ? prev : initialId));
      setLoadedId(data.current_id ?? stored ?? null);
    } catch (err) {
      console.error("Failed to fetch SAE combos:", err);
    }
  }, [backendBase]);

  // Fetch available combo information
  useEffect(() => {
    fetchCombos(false);
  }, [fetchCombos]);

  // Keep local UI in sync with combo changes within the same tab and across tabs.
  useEffect(() => {
    const syncFromStorage = () => {
      const stored = window.localStorage.getItem(LOCAL_STORAGE_KEY);
      if (stored) {
        setLoadedId(stored);
        setCurrentServerId((prev) => prev ?? stored);
      }
    };

    window.addEventListener("storage", syncFromStorage);
    const interval = window.setInterval(syncFromStorage, 1000);

    return () => {
      window.removeEventListener("storage", syncFromStorage);
      window.clearInterval(interval);
    };
  }, []);

  // Heal stale loading state if the selected combo is already known as loaded.
  useEffect(() => {
    if (isLoading && selectedId && (selectedId === loadedId || selectedId === currentServerId)) {
      setIsLoading(false);
    }
  }, [isLoading, selectedId, loadedId, currentServerId]);

  // 不再轮询 /circuit/loading_logs，避免持续打日志

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

    // If another combo is currently loading, cancel it first
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
        // Wait a short time for the cancellation to take effect
        await new Promise((resolve) => setTimeout(resolve, 500));
      } catch (err) {
        console.warn("Failed to cancel previous loading:", err);
      }
    }

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
      setLoadedId(data.sae_combo_id);
      window.localStorage.setItem(LOCAL_STORAGE_KEY, data.sae_combo_id);
      await fetchCombos();
      window.alert(`SAE combo ${data.sae_combo_id} 已加载完成`);
    } catch (err) {
      console.error("Failed to preload SAE models:", err);
      setIsLoading(false);
      if (loadedId === selectedId) {
        setLoadedId(null);
      }
      window.alert("加载 SAE combo 失败，请稍后重试");
    } finally {
      setIsLoading(false);
    }
  }, [selectedId, loadedId, isLoading, backendBase, fetchCombos]);

  const canReload = selectedId != null && !isLoading;

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
            {title ?? "BT4 SAE Combo Selection (Lorsa / Transcoder)"}
          </div>
          <div className="text-xs text-blue-800">
            Current server combo:{" "}
            <span className="font-mono">
              {currentServerId ?? defaultId ?? loadedId ?? selectedId ?? "Unknown"}
            </span>
          </div>
        </div>
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
          <Select
            value={selectedId ?? undefined}
            onValueChange={(value) => {
              setSelectedId(value);
              // 切换 combo 时只更新前端状态，不再请求 loading_logs
              if (value === currentServerId) {
                setLoadedId(value);
              } else {
                setLoadedId(null);
              }
            }}
          >
            <SelectTrigger className="w-56 bg-white">
              <SelectValue placeholder="Select SAE combo" />
            </SelectTrigger>
            <SelectContent>
              {combos.map((c) => (
                <SelectItem key={c.id} value={c.id}>
                  {c.label}{" "}
                  {c.id === defaultId ? "(default)" : undefined}
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
              ? "Loading..."
              : loadedId === selectedId
              ? "Loaded"
              : "Load / Reload"}
          </Button>
          {isLoading && (
            <Button
              type="button"
              size="sm"
              variant="destructive"
              onClick={handleCancel}
              className="whitespace-nowrap"
            >
              Cancel loading
            </Button>
          )}
        </div>
      </div>
    </div>
  );
};
