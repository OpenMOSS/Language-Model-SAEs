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
  is_loaded?: boolean;  // Added: backend-actual cache check result
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

  // Max number of logs to display on the frontend (keep only the latest N)
  const MAX_VISIBLE_LOGS = 200;

  const backendBase = import.meta.env.VITE_BACKEND_URL ?? "";

  // Fetch available combo information
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

        // Fetch historical logs for this combo (shared across pages) and verify whether it's really loaded
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

              // Sync loading state
              if (logData.is_loading) {
                setIsLoading(true);
              } else {
                setIsLoading(false);
              }

              // Use the backend's is_loaded field to judge whether the combo is really loaded (actual cache check)
              // This is the most reliable way, because the backend checks whether the cache exists
              if (logData.is_loaded === true && !logData.is_loading) {
                // Confirm loaded, set loadedId
                setLoadedId(initialId);
                console.log('✅ Confirmed SAE combo is loaded (cache verified):', initialId);
              } else {
                // Not loaded or state unclear, do not set loadedId
                setLoadedId(null);
                console.log('⚠️ SAE combo not loaded or state unclear:', initialId, {
                  is_loaded: logData.is_loaded,
                  is_loading: logData.is_loading
                });
              }
            } else {
              // If fetching logs fails, do not set loadedId
              setLoadedId(null);
            }
          } catch (logErr) {
            console.warn("Failed to fetch initial loading logs:", logErr);
            // If getting logs fails, do not set loadedId
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

  // Poll logs (poll even when not loading, to show historical logs and real-time updates)
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
          // Keep only the latest MAX_VISIBLE_LOGS entries
          const sliced = allLogs.slice(-MAX_VISIBLE_LOGS);
          setLogs(sliced);
          // If the backend reports loading is in progress, but the frontend state is not, update frontend state
          if (data.is_loading && !isLoading) {
            setIsLoading(true);
          } else if (!data.is_loading && isLoading) {
            setIsLoading(false);
          }
          // Sync loadedId state (based on actual cache check)
          if (data.is_loaded === true && !data.is_loading) {
            // Confirm loaded, set loadedId
            if (selectedId && selectedId !== loadedId) {
              setLoadedId(selectedId);
            }
          } else if (data.is_loaded === false) {
            // If the cache does not exist, and the currently selected combo is considered loaded, clear loadedId
            if (loadedId === selectedId) {
              setLoadedId(null);
            }
          }
        }
      } catch (err) {
        console.error("Failed to fetch loading logs:", err);
      }
    };

    // Execute once immediately, then start polling (so logs appear right away when switching pages)
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

    // Clear local log cache before each reload to avoid old combo's logs sticking around
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
    } catch (err) {
      console.error("Failed to preload SAE models:", err);
      // Clear state when loading fails
      setIsLoading(false);
      if (loadedId === selectedId) {
        setLoadedId(null);
      }
    }
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
            onValueChange={async (value) => {
              setSelectedId(value);
              // When switching combo, immediately fetch logs for the new combo
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
                  // Sync loadedId state (based on actual cache check)
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
      <div className="mt-2 max-h-40 overflow-y-auto rounded bg-blue-100 p-2 text-xs font-mono leading-relaxed">
        {logs.length === 0 ? (
          <div className="text-blue-700 opacity={0.8}">
            No loading logs yet. Select a combo and click "Load / Reload" to start loading Lorsa / Transcoder.
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


