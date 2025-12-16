import React, { useState, useCallback, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Loader2, Settings } from 'lucide-react';
import { LinkGraphContainer } from './link-graph-container';
import { NodeConnections } from './node-connections';
import { FeatureCard } from '@/components/feature/feature-card';
import { ChessBoard } from '@/components/chess/chess-board';
import { Feature } from '@/types/feature';
import { transformCircuitData } from './link-graph/utils';

interface CircuitTracingProps {
  gameFen: string; // move之前的FEN
  previousFen?: string | null; // 上一个FEN状态
  currentFen?: string; // 当前FEN状态
  gameHistory: string[];
  lastMove?: string | null; // 最后一个移动
  onCircuitTraceStart?: () => void;
  onCircuitTraceEnd?: () => void;
  isTracing?: boolean;
}

export const CircuitTracing: React.FC<CircuitTracingProps> = ({
  gameFen,
  previousFen: _previousFen,
  currentFen,
  gameHistory,
  lastMove,
  onCircuitTraceStart,
  onCircuitTraceEnd,
  isTracing = false,
}) => {
  const [circuitTraceResult, setCircuitTraceResult] = useState<any>(null);
  const [circuitVisualizationData, setCircuitVisualizationData] = useState<any>(null);
  const [clickedNodeId, setClickedNodeId] = useState<string | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [pinnedNodeIds, setPinnedNodeIds] = useState<string[]>([]);
  const [hiddenNodeIds, setHiddenNodeIds] = useState<string[]>([]);
  const [selectedFeature, setSelectedFeature] = useState<Feature | null>(null);
  const [connectedFeatures, setConnectedFeatures] = useState<Feature[]>([]);
  const [, setIsLoadingConnectedFeatures] = useState(false);

  // Circuit Trace 参数状态
  const [showParamsDialog, setShowParamsDialog] = useState(false);
  const [circuitParams, setCircuitParams] = useState({
    max_feature_nodes: 4096,
    node_threshold: 0.73,
    edge_threshold: 0.57,
    max_act_times: null as number | null,
  });

  // 移动输入状态
  const [positiveMove, setPositiveMove] = useState<string>('');
  const [negativeMove, setNegativeMove] = useState<string>('');
  const [moveError, setMoveError] = useState<string>('');

  // Side选择状态
  const [traceSide, setTraceSide] = useState<'q' | 'k' | 'both'>('k');
  
  // 固定使用BT4模型
  const traceModel = 'BT4';

  // Top Activation 相关状态
  const [topActivations, setTopActivations] = useState<any[]>([]);
  const [loadingTopActivations, setLoadingTopActivations] = useState(false);

  // Circuit Trace 日志相关状态
  const [traceLogs, setTraceLogs] = useState<Array<{timestamp: number; message: string}>>([]);
  const MAX_VISIBLE_LOGS = 100;
  const LAST_TRACE_REQUEST_KEY = 'circuit_trace_last_request_v1';
  const TRACE_RESULT_CACHE_KEY = 'circuit_trace_result_cache_v1'; // localStorage key for trace result backup

  // 恢复结果相关状态
  const [isRecovering, setIsRecovering] = useState(false);
  const [lastTraceInfo, setLastTraceInfo] = useState<any>(null);
  const [showRecoveryButton, setShowRecoveryButton] = useState(false);

  // 直接使用父组件传入的上一步FEN，不再使用本地缓存覆盖
  const effectiveGameFen = gameFen;

  // 确保FEN在构造trace key和请求时是解码后的原始形式，避免重复编码导致404
  const safeDecodeFen = useCallback((fen: string): string => {
    try {
      return decodeURIComponent(fen);
    } catch {
      return fen;
    }
  }, []);

  // 本地缓存：按FEN缓存最近一次输入的UCI移动
  const MOVE_CACHE_KEY = 'circuit_move_by_fen_v1';
  const POSITIVE_MOVE_CACHE_KEY = 'circuit_positive_move_by_fen_v1';
  const NEGATIVE_MOVE_CACHE_KEY = 'circuit_negative_move_by_fen_v1';
  
  const loadCachedMove = useCallback((fen: string): string => {
    try {
      const raw = localStorage.getItem(MOVE_CACHE_KEY);
      if (!raw) return '';
      const obj = JSON.parse(raw) as Record<string, string>;
      return obj[fen] || '';
    } catch {
      return '';
    }
  }, []);

  const loadLastTraceRequest = useCallback((): any | null => {
    try {
      const raw = localStorage.getItem(LAST_TRACE_REQUEST_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  }, []);

  const saveLastTraceRequest = useCallback((payload: any) => {
    try {
      localStorage.setItem(LAST_TRACE_REQUEST_KEY, JSON.stringify(payload));
    } catch {
      /* no-op */
    }
  }, []);
  
  const loadCachedPositiveMove = useCallback((fen: string): string => {
    try {
      const raw = localStorage.getItem(POSITIVE_MOVE_CACHE_KEY);
      if (!raw) return '';
      const obj = JSON.parse(raw) as Record<string, string>;
      return obj[fen] || '';
    } catch {
      return '';
    }
  }, []);
  
  const loadCachedNegativeMove = useCallback((fen: string): string => {
    try {
      const raw = localStorage.getItem(NEGATIVE_MOVE_CACHE_KEY);
      if (!raw) return '';
      const obj = JSON.parse(raw) as Record<string, string>;
      return obj[fen] || '';
    } catch {
      return '';
    }
  }, []);
  
  const saveCachedMove = useCallback((fen: string, move: string) => {
    try {
      const raw = localStorage.getItem(MOVE_CACHE_KEY);
      const obj = raw ? (JSON.parse(raw) as Record<string, string>) : {};
      obj[fen] = move;
      localStorage.setItem(MOVE_CACHE_KEY, JSON.stringify(obj));
    } catch {
      /* no-op */
    }
  }, []);
  
  const saveCachedPositiveMove = useCallback((fen: string, move: string) => {
    try {
      const raw = localStorage.getItem(POSITIVE_MOVE_CACHE_KEY);
      const obj = raw ? (JSON.parse(raw) as Record<string, string>) : {};
      obj[fen] = move;
      localStorage.setItem(POSITIVE_MOVE_CACHE_KEY, JSON.stringify(obj));
    } catch {
      /* no-op */
    }
  }, []);
  
  const saveCachedNegativeMove = useCallback((fen: string, move: string) => {
    try {
      const raw = localStorage.getItem(NEGATIVE_MOVE_CACHE_KEY);
      const obj = raw ? (JSON.parse(raw) as Record<string, string>) : {};
      obj[fen] = move;
      localStorage.setItem(NEGATIVE_MOVE_CACHE_KEY, JSON.stringify(obj));
    } catch {
      /* no-op */
    }
  }, []);

  // 节点激活数据接口
  interface NodeActivationData {
    activations?: number[];
    zPatternIndices?: any;
    zPatternValues?: number[];
    nodeType?: string;
    clerp?: string;
  }

  // 轮询circuit trace日志
  useEffect(() => {
    // 如果不在tracing且没有日志，不轮询
    if (!isTracing && traceLogs.length === 0) return;

    let cancelled = false;
    let pollCount = 0;
    const MAX_POLL_AFTER_COMPLETE = 5; // tracing完成后继续轮询5次以确保获取所有日志

    const poll = async () => {
      try {
        const params = new URLSearchParams({
          model_name: "lc0/BT4-1024x15x32h",
          fen: effectiveGameFen,
        });
        const res = await fetch(`${import.meta.env.VITE_BACKEND_URL}/circuit_trace/logs?${params.toString()}`);
        if (!res.ok) return;
        const data = await res.json();
        if (!cancelled) {
          const allLogs = data.logs ?? [];
          // 只保留最近 MAX_VISIBLE_LOGS 条
          const sliced = allLogs.slice(-MAX_VISIBLE_LOGS);
          setTraceLogs(sliced);
          
          // 如果tracing已完成且已轮询足够次数，停止轮询
          if (!isTracing && !data.is_tracing) {
            pollCount++;
            if (pollCount >= MAX_POLL_AFTER_COMPLETE) {
              cancelled = true;
            }
          } else {
            pollCount = 0; // 重置计数
          }
        }
      } catch (err) {
        console.error("Failed to fetch circuit trace logs:", err);
      }
    };

    // 立即执行一次，然后开始轮询
    poll();
    const timer = window.setInterval(() => {
      if (!cancelled) {
        poll();
      } else {
        window.clearInterval(timer);
      }
    }, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [isTracing, effectiveGameFen, traceLogs.length]);

  // 保存trace结果到localStorage作为备份
  const saveTraceResultToLocalStorage = useCallback((traceKey: string, result: any) => {
    try {
      const cacheData = {
        trace_key: traceKey,
        result: result,
        saved_at: Date.now(),
      };
      localStorage.setItem(TRACE_RESULT_CACHE_KEY, JSON.stringify(cacheData));
      console.log('✅ Trace结果已备份到localStorage');
    } catch (error) {
      console.error('⚠️ 保存trace结果到localStorage失败:', error);
      // localStorage可能已满，尝试清理旧数据
      try {
        localStorage.removeItem(TRACE_RESULT_CACHE_KEY);
        localStorage.setItem(TRACE_RESULT_CACHE_KEY, JSON.stringify({
          trace_key: traceKey,
          result: result,
          saved_at: Date.now(),
        }));
      } catch (e) {
        console.error('⚠️ 清理后仍无法保存到localStorage:', e);
      }
    }
  }, []);

  // 从localStorage加载trace结果
  const loadTraceResultFromLocalStorage = useCallback((traceKey: string): any | null => {
    try {
      const cached = localStorage.getItem(TRACE_RESULT_CACHE_KEY);
      if (!cached) return null;
      
      const cacheData = JSON.parse(cached);
      // 检查是否匹配当前的trace_key，且不超过7天
      if (cacheData.trace_key === traceKey && 
          Date.now() - cacheData.saved_at < 7 * 24 * 3600 * 1000) {
        return cacheData.result;
      }
      return null;
    } catch (error) {
      console.error('⚠️ 从localStorage加载trace结果失败:', error);
      return null;
    }
  }, []);

  // 新增：handleCircuitTrace函数
  const handleCircuitTraceResult = useCallback((result: any, traceKey?: string) => {
    if (result && result.nodes) {
      try {
        const transformedData = transformCircuitData(result);
        setCircuitVisualizationData(transformedData);
        setCircuitTraceResult(result);
        // 成功加载结果后隐藏恢复按钮
        setShowRecoveryButton(false);
        
        // 保存到localStorage作为备份
        if (traceKey) {
          saveTraceResultToLocalStorage(traceKey, result);
        }
      } catch (error) {
        console.error('Circuit数据转换失败:', error);
        alert('Circuit数据转换失败: ' + (error instanceof Error ? error.message : '未知错误'));
      }
    }
  }, [saveTraceResultToLocalStorage]);

  // 新增：处理节点点击 - 修复参数传递
  const handleNodeClick = useCallback((node: any, isMetaKey: boolean) => {
    const nodeId = node.nodeId || node.id;
    console.log('🔍 节点被点击:', { nodeId, isMetaKey, currentClickedId: clickedNodeId, node });
    
    if (isMetaKey) {
      // Toggle pinned state
      const newPinnedIds = pinnedNodeIds.includes(nodeId)
        ? pinnedNodeIds.filter(id => id !== nodeId)
        : [...pinnedNodeIds, nodeId];
      setPinnedNodeIds(newPinnedIds);
      console.log('📌 切换固定状态:', newPinnedIds);
    } else {
      // Set clicked node
      const newClickedId = nodeId === clickedNodeId ? null : nodeId;
      setClickedNodeId(newClickedId);
      console.log('🎯 设置选中节点:', newClickedId);
      
      // 清除之前的特征选择
      if (newClickedId === null) {
        setSelectedFeature(null);
        setConnectedFeatures([]);
      }
    }
  }, [clickedNodeId, pinnedNodeIds]);

  // 新增：处理节点悬停 - 修复参数传递
  const handleNodeHover = useCallback((nodeId: string | null) => {
    if (nodeId !== hoveredNodeId) {
      setHoveredNodeId(nodeId);
    }
  }, [hoveredNodeId]);

  // 新增：处理特征选择
  const handleFeatureSelect = useCallback((feature: Feature | null) => {
    setSelectedFeature(feature);
  }, []);

  // 新增：处理连接特征选择
  const handleConnectedFeaturesSelect = useCallback((features: Feature[]) => {
    setConnectedFeatures(features);
    setIsLoadingConnectedFeatures(false);
  }, []);

  // 新增：处理连接特征加载
  const handleConnectedFeaturesLoading = useCallback((loading: boolean) => {
    setIsLoadingConnectedFeatures(loading);
  }, []);

  // 验证移动合法性
  const validateMove = useCallback((move: string, _fen: string): boolean => {
    try {
      // 简单的UCI格式验证
      if (!/^[a-h][1-8][a-h][1-8][qrbn]?$/.test(move)) {
        setMoveError('移动格式不正确，应为UCI格式（如：e2e4）');
        return false;
      }

      // 这里可以添加更复杂的合法性检查，比如调用chess.js库
      // 暂时只做格式检查
      setMoveError('');
      return true;
    } catch (error) {
      setMoveError('移动验证失败');
      return false;
    }
  }, []);

  // 生成trace_key（与后端保持一致）
  // 后端使用 get_bt4_sae_combo(sae_combo_id)["id"] 来获取normalized_combo_id
  // 如果sae_combo_id为None或不存在，会回退到默认组合 "k_128_e_128"
  const generateTraceKey = useCallback((fen: string, moveUci: string, saeComboId: string | null | undefined): string => {
    const modelName = 'lc0/BT4-1024x15x32h';
    // 与后端保持一致：如果saeComboId为空，使用默认组合 "k_128_e_128"
    // 注意：后端会通过get_bt4_sae_combo规范化ID，但前端无法直接调用
    // 所以这里假设前端传递的saeComboId已经是正确的（在BT4_SAE_COMBOS中存在）
    // 如果不存在，后端会回退到默认值，所以前端也应该使用默认值
    const comboId = saeComboId || 'k_128_e_128'; // 与后端BT4_DEFAULT_SAE_COMBO保持一致
    const decodedFen = safeDecodeFen(fen);
    const decodedMove = safeDecodeFen(moveUci);
    return `${modelName}::${comboId}::${decodedFen}::${decodedMove}`;
  }, [safeDecodeFen]);

  // 获取已存在的trace结果（优先从localStorage，再尝试后端）
  const fetchExistingTraceResult = useCallback(
    async (fen: string, moveUci: string, saeComboId: string | null | undefined, showSuccess: boolean = false) => {
      const decodedFen = safeDecodeFen(fen);
      const decodedMove = safeDecodeFen(moveUci);
      const traceKey = generateTraceKey(decodedFen, decodedMove, saeComboId);
      
      // 1. 先尝试从localStorage加载
      const cachedResult = loadTraceResultFromLocalStorage(traceKey);
      if (cachedResult && cachedResult.nodes) {
        console.log('✅ 从localStorage恢复trace结果');
        handleCircuitTraceResult(cachedResult, traceKey);
        if (showSuccess) {
          alert(`✅ 成功从本地缓存恢复Circuit trace结果！\n\n节点数: ${cachedResult.nodes.length}\n连接数: ${cachedResult.links?.length || 0}`);
        }
        return true;
      }
      
      // 2. 尝试从后端加载
      try {
        const params = new URLSearchParams({
          fen: decodedFen,
          move_uci: decodedMove,
        });
        if (saeComboId) {
          params.set('sae_combo_id', saeComboId);
        }
        const res = await fetch(`${import.meta.env.VITE_BACKEND_URL}/circuit_trace/result?${params.toString()}`);
        if (!res.ok) {
          if (showSuccess) {
            throw new Error(`HTTP ${res.status}: ${await res.text()}`);
          }
          return false;
        }
        const data = await res.json();
        if (data?.graph_data?.nodes) {
          handleCircuitTraceResult(data.graph_data, traceKey);
          if (data.logs) {
            const sliced = (data.logs as Array<{ timestamp: number; message: string }>)?.slice(-MAX_VISIBLE_LOGS) || [];
            setTraceLogs(sliced);
          }
          if (showSuccess) {
            alert(`✅ 成功从后端恢复Circuit trace结果！\n\n节点数: ${data.graph_data.nodes.length}\n连接数: ${data.graph_data.links?.length || 0}`);
          }
          return true;
        }
        return false;
      } catch (err) {
        console.error('恢复trace结果失败:', err);
        if (showSuccess) {
          alert(`❌ 恢复失败: ${err instanceof Error ? err.message : '未知错误'}`);
        }
        return false;
      }
    },
    [handleCircuitTraceResult, generateTraceKey, loadTraceResultFromLocalStorage, safeDecodeFen],
  );

  // 手动恢复上一次trace结果
  const handleManualRecovery = useCallback(async () => {
    const lastRequest = loadLastTraceRequest();
    if (!lastRequest || !lastRequest.fen || !lastRequest.move_uci) {
      alert('❌ 没有找到可恢复的trace请求信息');
      return;
    }

    setIsRecovering(true);
    try {
      console.log('🔄 手动恢复trace结果:', lastRequest);
      const success = await fetchExistingTraceResult(
        safeDecodeFen(lastRequest.fen), 
        safeDecodeFen(lastRequest.move_uci), 
        lastRequest.sae_combo_id,
        true // 显示成功/失败消息
      );
      
      if (!success) {
        // 如果没有现成的结果，检查是否还在tracing中
        try {
          const params = new URLSearchParams({
            model_name: 'lc0/BT4-1024x15x32h',
            fen: safeDecodeFen(lastRequest.fen),
            move_uci: safeDecodeFen(lastRequest.move_uci),
          });
          if (lastRequest.sae_combo_id) params.set('sae_combo_id', lastRequest.sae_combo_id);
          
          const statusRes = await fetch(`${import.meta.env.VITE_BACKEND_URL}/circuit_trace/logs?${params.toString()}`);
          if (statusRes.ok) {
            const statusData = await statusRes.json();
            if (statusData.is_tracing) {
              alert('⏳ 后端正在执行该trace请求，请稍等完成后再试');
              // 开始轮询日志
              const sliced = (statusData.logs as Array<{ timestamp: number; message: string }>)?.slice(-MAX_VISIBLE_LOGS) || [];
              setTraceLogs(sliced);
            } else {
              alert('❌ 没有找到对应的trace结果，可能已过期或被清理');
            }
          } else {
            alert('❌ 无法检查trace状态，请重新执行trace');
          }
        } catch (statusErr) {
          console.error('检查trace状态失败:', statusErr);
          alert('❌ 检查trace状态失败，请重新执行trace');
        }
      }
    } catch (error) {
      console.error('手动恢复失败:', error);
      alert(`❌ 恢复失败: ${error instanceof Error ? error.message : '未知错误'}`);
    } finally {
      setIsRecovering(false);
    }
  }, [loadLastTraceRequest, fetchExistingTraceResult, safeDecodeFen]);

  // 检查是否有可恢复的trace信息
  const checkRecoverableTrace = useCallback(() => {
    const lastRequest = loadLastTraceRequest();
    if (lastRequest && lastRequest.fen && lastRequest.move_uci) {
      setLastTraceInfo(lastRequest);
      setShowRecoveryButton(true);
    } else {
      setLastTraceInfo(null);
      setShowRecoveryButton(false);
    }
  }, [loadLastTraceRequest]);

  // 修改handleCircuitTrace函数来支持不同的order_mode和both trace
  const handleCircuitTrace = useCallback(async (orderMode: 'positive' | 'negative' | 'both' = 'positive') => {
    // 先检查后端是否有正在进行的circuit tracing进程
    try {
      const statusResponse = await fetch(`${import.meta.env.VITE_BACKEND_URL}/circuit_trace/status`);
      if (statusResponse.ok) {
        const status = await statusResponse.json();
        if (status.is_tracing) {
          alert('后端正在执行另一个circuit tracing进程，请等待完成后再试');
          return;
        }
      }
    } catch (error) {
      console.error('检查circuit tracing状态失败:', error);
      // 如果检查失败，仍然继续执行（避免因为网络问题阻止用户操作）
    }
    
    let moveUci: string | null = null;
    const decodedFen = safeDecodeFen(effectiveGameFen);
    const lastMoveStr: string | null = lastMove ? lastMove : null;
    
    // 调试日志：记录所有状态值
    console.log('🔍 [DEBUG] handleCircuitTrace 调用:', {
      orderMode,
      positiveMove: positiveMove,
      negativeMove: negativeMove,
      lastMove: lastMove,
      gameFen: gameFen,
      cachedPositive: loadCachedPositiveMove(gameFen),
      cachedNegative: loadCachedNegativeMove(gameFen),
      cachedMove: loadCachedMove(gameFen),
    });
    
    if (orderMode === 'both') {
      // Both Trace: 需要positive move和negative move
      const posMove = positiveMove.trim() || loadCachedPositiveMove(decodedFen);
      const negMove = negativeMove.trim() || loadCachedNegativeMove(decodedFen);
      
      if (!posMove) {
        alert('Both Trace需要输入Positive Move');
        return;
      }
      if (!negMove) {
        alert('Both Trace需要输入Negative Move');
        return;
      }
      
      // 验证两个移动格式
      if (!validateMove(posMove, gameFen)) {
        setMoveError('Positive Move格式不正确');
        return;
      }
      if (!validateMove(negMove, gameFen)) {
        setMoveError('Negative Move格式不正确');
        return;
      }
      
      // Both trace使用positive move作为主要move，negative move通过order_mode传递
      moveUci = posMove;
      
      console.log('🔍 Both Circuit Trace 参数:', {
        fen: gameFen,
        positive_move: posMove,
        negative_move: negMove,
        side: 'both',
        order_mode: 'both',
        trace_model: traceModel
      });
    } else {
      // Positive/Negative Trace: 使用对应的move
      if (orderMode === 'positive') {
        const trimmedPositive = positiveMove.trim();
        const cachedPos = loadCachedPositiveMove(decodedFen);
        const cached = loadCachedMove(decodedFen);
        
        // 优先使用用户输入的移动，只有在用户没有输入时才使用缓存或lastMove
        moveUci = trimmedPositive || cachedPos || cached || lastMoveStr;
        
        // 调试日志：记录选择过程
        console.log('🔍 [DEBUG] Positive Trace 移动选择:', {
          '用户输入 (positiveMove.trim())': trimmedPositive || '(空)',
          '缓存 Positive Move': cachedPos || '(无)',
          '缓存 Move': cached || '(无)',
          '最后移动 (lastMove)': lastMoveStr || '(无)',
          '最终选择': moveUci,
        });
      } else {
        const trimmedNegative = negativeMove.trim();
        const cachedNeg = loadCachedNegativeMove(decodedFen);
        const cached = loadCachedMove(decodedFen);
        
        moveUci = trimmedNegative || cachedNeg || cached || lastMoveStr;
        
        // 调试日志：记录选择过程
        console.log('🔍 [DEBUG] Negative Trace 移动选择:', {
          '用户输入 (negativeMove.trim())': trimmedNegative || '(空)',
          '缓存 Negative Move': cachedNeg || '(无)',
          '缓存 Move': cached || '(无)',
          '最后移动 (lastMove)': lastMoveStr || '(无)',
          '最终选择': moveUci,
        });
      }
      
      if (!moveUci) {
        alert(`请输入${orderMode === 'positive' ? 'Positive' : 'Negative'} Move或先走一步棋`);
        return;
      }
      
      // 验证移动格式
      if (!validateMove(moveUci, gameFen)) {
        return;
      }
      
      console.log('🔍 Circuit Trace 参数:', {
        fen: decodedFen,
        move_uci: moveUci,
        order_mode: orderMode,
        side: traceSide,
        trace_model: 'BT4'  // 固定使用BT4模型
      });
    }
    
    onCircuitTraceStart?.();
    
    // 清空之前的日志
    setTraceLogs([]);
    
    try {
      // 固定使用BT4模型（modelName在generateTraceKey中使用）
      
      // 获取当前选中的 SAE 组合 ID（从 localStorage 读取，与 SaeComboLoader 保持一致）
      const LOCAL_STORAGE_KEY = "bt4_sae_combo_id";
      const currentSaeComboId = window.localStorage.getItem(LOCAL_STORAGE_KEY) || null;
      
      // 构建请求体
      const requestBody: any = { 
        fen: decodedFen,
        move_uci: moveUci,
        side: orderMode === 'both' ? 'both' : traceSide,
        order_mode: orderMode,
        max_feature_nodes: circuitParams.max_feature_nodes,
        node_threshold: circuitParams.node_threshold,
        edge_threshold: circuitParams.edge_threshold,
        max_act_times: circuitParams.max_act_times,
        save_activation_info: true
      };
      
      // 如果前端有选中的 SAE 组合 ID，传递给后端
      if (currentSaeComboId) {
        requestBody.sae_combo_id = currentSaeComboId;
      }
      
      // Both trace需要传递negative move
      if (orderMode === 'both') {
        const negMove = negativeMove.trim() || loadCachedNegativeMove(decodedFen);
        if (negMove) {
          requestBody.negative_move_uci = negMove;
        }
      }
      
      // 调试日志：记录实际发送的请求体
      console.log('🔍 [DEBUG] 发送 Circuit Trace 请求:', {
        requestBody,
        '实际使用的 move_uci': requestBody.move_uci,
        '用户输入的 positiveMove': positiveMove,
        '用户输入的 negativeMove': negativeMove,
        '当前 SAE 组合 ID': currentSaeComboId,
      });

      saveLastTraceRequest({
        fen: decodedFen,
        move_uci: requestBody.move_uci,
        order_mode: orderMode,
        side: requestBody.side,
        sae_combo_id: currentSaeComboId,
        timestamp: Date.now(), // 添加时间戳
      });
      
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/circuit_trace`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });
      
      if (response.ok) {
        const data = await response.json();
        // 成功后缓存移动
        if (orderMode === 'both' || orderMode === 'positive') {
          const posMove = positiveMove.trim() || loadCachedPositiveMove(decodedFen);
          if (posMove) {
            saveCachedPositiveMove(decodedFen, posMove);
          }
        }
        if (orderMode === 'both' || orderMode === 'negative') {
          const negMove = negativeMove.trim() || loadCachedNegativeMove(decodedFen);
          if (negMove) {
            saveCachedNegativeMove(decodedFen, negMove);
          }
        }
        
        // 后端已经根据 sae_combo_id 从 constants.py 设置了正确的 metadata，
        // 直接使用后端返回的值，不再覆盖
        if (data.metadata) {
          console.log('🔍 后端返回的 metadata:', {
            lorsa_analysis_name: data.metadata.lorsa_analysis_name,
            tc_analysis_name: data.metadata.tc_analysis_name,
            sae_combo_id: currentSaeComboId,
          });
        }
        
        // 生成trace_key并保存结果
        const traceKey = generateTraceKey(effectiveGameFen, requestBody.move_uci, currentSaeComboId);
        handleCircuitTraceResult(data, traceKey);
      } else {
        const errorText = await response.text();
        console.error('Circuit trace API调用失败:', response.status, response.statusText, errorText);
        alert('Circuit trace失败: ' + errorText);
      }
    } catch (error) {
      console.error('Circuit trace出错:', error);
      const errorMessage = error instanceof Error ? error.message : '未知错误';
      
      // 检查是否是网络错误
      const isNetworkError = errorMessage.toLowerCase().includes('fetch') || 
                            errorMessage.toLowerCase().includes('network') ||
                            errorMessage.toLowerCase().includes('connection');
      
      if (isNetworkError) {
        const shouldRecover = confirm(
          `❌ Circuit trace遇到网络错误: ${errorMessage}\n\n` +
          `💡 可能的解决方案:\n` +
          `1. 点击"确定"尝试恢复上一次的trace结果\n` +
          `2. 点击"取消"重新执行trace\n\n` +
          `是否尝试恢复上一次的结果？`
        );
        
        if (shouldRecover) {
          // 延迟一下再尝试恢复，给后端时间完成处理
          setTimeout(() => {
            handleManualRecovery();
          }, 2000);
        }
      } else {
        alert('Circuit trace出错: ' + errorMessage);
      }
    } finally {
      onCircuitTraceEnd?.();
    }
  }, [gameFen, currentFen, lastMove, gameHistory, positiveMove, negativeMove, validateMove, onCircuitTraceStart, onCircuitTraceEnd, handleCircuitTraceResult, circuitParams, traceSide, loadCachedMove, saveCachedMove, loadCachedPositiveMove, loadCachedNegativeMove, saveCachedPositiveMove, saveCachedNegativeMove, handleManualRecovery, generateTraceKey, safeDecodeFen]);

  // 新增：保存原始graph JSON（与后端create_graph_files一致的数据结构）
  const handleSaveGraphJson = useCallback(() => {
    try {
      const raw = circuitTraceResult || circuitVisualizationData;
      if (!raw) {
        alert('没有可保存的图数据');
        return;
      }
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      const slug = raw?.metadata?.slug || 'circuit_trace';
      // 从当前FEN解析全回合数（第6段），若解析失败则回退为基于历史长度估算
      const fenParts = effectiveGameFen.split(' ');
      const fullmove = fenParts.length >= 6 && !Number.isNaN(parseInt(fenParts[5]))
        ? parseInt(fenParts[5])
        : Math.max(1, Math.ceil(gameHistory.length / 2));
      const fileName = `${slug}_m${fullmove}_${timestamp}.json`;
      const jsonString = JSON.stringify(raw, null, 2);
      const blob = new Blob([jsonString], { type: 'application/json' });
      const url = URL.createObjectURL(blob);

      const link = document.createElement('a');
      link.href = url;
      link.download = fileName;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('保存JSON失败:', error);
      alert('保存JSON失败');
    }
  }, [circuitTraceResult, circuitVisualizationData, gameFen, gameHistory]);

  useEffect(() => {
    const last = loadLastTraceRequest();
    if (!last || !last.fen || !last.move_uci) return;

    let cancelled = false;
    let attempts = 0;
    const MAX_ATTEMPTS = 30;

    const pollExisting = async () => {
      if (cancelled) return;
      try {
        const params = new URLSearchParams({
          model_name: 'lc0/BT4-1024x15x32h',
          fen: safeDecodeFen(last.fen),
          move_uci: safeDecodeFen(last.move_uci),
        });
        if (last.sae_combo_id) params.set('sae_combo_id', last.sae_combo_id);
        const res = await fetch(`${import.meta.env.VITE_BACKEND_URL}/circuit_trace/logs?${params.toString()}`);
        if (!res.ok) return;
        const data = await res.json();
        const sliced = (data.logs as Array<{ timestamp: number; message: string }>)?.slice(-MAX_VISIBLE_LOGS) || [];
        setTraceLogs(sliced);

        if (!data.is_tracing) {
          await fetchExistingTraceResult(last.fen, last.move_uci, last.sae_combo_id);
          cancelled = true;
          return;
        }
      } catch (err) {
        console.error('恢复trace日志失败:', err);
      }
      attempts += 1;
      if (attempts >= MAX_ATTEMPTS) cancelled = true;
    };

    pollExisting();
    const timer = window.setInterval(() => {
      if (!cancelled) {
        pollExisting();
      } else {
        window.clearInterval(timer);
      }
    }, 2000);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [loadLastTraceRequest, fetchExistingTraceResult, safeDecodeFen]);

  // 组件挂载时检查可恢复的trace
  useEffect(() => {
    checkRecoverableTrace();
  }, [checkRecoverableTrace]);

  // FEN变化时重新检查可恢复的trace
  useEffect(() => {
    if (effectiveGameFen) {
      checkRecoverableTrace();
    }
  }, [effectiveGameFen, checkRecoverableTrace]);

  // 处理参数设置
  const handleParamsChange = useCallback((key: keyof typeof circuitParams, value: string) => {
    setCircuitParams(prev => ({
      ...prev,
      [key]: key === 'max_feature_nodes' ? parseInt(value) || 1024 : 
              key === 'max_act_times' ? (() => {
                if (value === '') return null;
                const num = parseInt(value);
                if (isNaN(num)) return null;
                // 限制在10M-100M范围内，按10M步长调整
                const clamped = Math.max(10000000, Math.min(100000000, num));
                // 四舍五入到最近的10M
                return Math.round(clamped / 10000000) * 10000000;
              })() :
              parseFloat(value) || prev[key]
    }));
  }, []);

  const handleSaveParams = useCallback(() => {
    setShowParamsDialog(false);
  }, []);

  // 获取 Top Activation 数据的函数
  const fetchTopActivations = useCallback(async (nodeId: string) => {
    if (!nodeId) return;
    
    setLoadingTopActivations(true);
    try {
      // 从 nodeId 解析出 feature 信息
      const parts = nodeId.split('_');
      const rawLayer = Number(parts[0]) || 0;
      const featureIndex = Number(parts[1]) || 0;
      const layerIdx = Math.floor(rawLayer / 2);
      
      // 确定节点类型和对应的字典名
      const currentNode = circuitVisualizationData?.nodes.find((n: any) => n.nodeId === nodeId);
      const isLorsa = currentNode?.feature_type?.toLowerCase() === 'lorsa';
      
      // 使用metadata信息确定字典名
      let dictionary: string;
      if (isLorsa) {
        const lorsaAnalysisName = circuitVisualizationData?.metadata?.lorsa_analysis_name;
        if (lorsaAnalysisName && lorsaAnalysisName.includes('BT4')) {
          // BT4格式: BT4_lorsa_L{layer}A
          dictionary = `BT4_lorsa_L${layerIdx}A`;
        } else {
          dictionary = lorsaAnalysisName ? lorsaAnalysisName.replace("{}", layerIdx.toString()) : `lc0-lorsa-L${layerIdx}`;
        }
      } else {
        const tcAnalysisName = (circuitVisualizationData?.metadata as any)?.tc_analysis_name || circuitVisualizationData?.metadata?.clt_analysis_name;
        if (tcAnalysisName && tcAnalysisName.includes('BT4')) {
          // BT4格式: BT4_tc_L{layer}M
          dictionary = `BT4_tc_L${layerIdx}M`;
        } else {
          dictionary = tcAnalysisName ? tcAnalysisName.replace("{}", layerIdx.toString()) : `lc0_L${layerIdx}M_16x_k30_lr2e-03_auxk_sparseadam`;
        }
      }
      
      console.log('🔍 获取 Top Activation 数据:', {
        nodeId,
        layerIdx,
        featureIndex,
        dictionary,
        isLorsa
      });
      
      // 调用后端 API 获取 feature 数据
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}/features/${featureIndex}`,
        {
          method: "GET",
          headers: {
            Accept: "application/x-msgpack",
          },
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }
      
      const arrayBuffer = await response.arrayBuffer();
      const decoded = await import("@msgpack/msgpack").then(module => module.decode(new Uint8Array(arrayBuffer)));
      const camelcaseKeys = await import("camelcase-keys").then(module => module.default);
      
      // 解析数据
      const camelData = camelcaseKeys(decoded as Record<string, unknown>, {
        deep: true,
        stopPaths: ["sample_groups.samples.context"],
      }) as any;
      
      // 提取样本数据
      const sampleGroups = camelData?.sampleGroups || camelData?.sample_groups || [];
      const allSamples: any[] = [];
      
      for (const group of sampleGroups) {
        if (group.samples && Array.isArray(group.samples)) {
          allSamples.push(...group.samples);
        }
      }
      
      // 查找包含 FEN 的样本并提取激活值
      const chessSamples: any[] = [];
      
      for (const sample of allSamples) {
        if (sample.text) {
          const lines = sample.text.split('\n');
          
          for (const line of lines) {
            const trimmed = line.trim();
            
            // 检查是否包含 FEN 格式
            if (trimmed.includes('/')) {
              const parts = trimmed.split(/\s+/);
              
              if (parts.length >= 6) {
                const [boardPart, activeColor] = parts;
                const boardRows = boardPart.split('/');
                
                if (boardRows.length === 8 && /^[wb]$/.test(activeColor)) {
                  // 验证 FEN 格式
                  let isValidBoard = true;
                  let totalSquares = 0;
                  
                  for (const row of boardRows) {
                    if (!/^[rnbqkpRNBQKP1-8]+$/.test(row)) {
                      isValidBoard = false;
                      break;
                    }
                    
                    let rowSquares = 0;
                    for (const char of row) {
                      if (/\d/.test(char)) {
                        rowSquares += parseInt(char);
                      } else {
                        rowSquares += 1;
                      }
                    }
                    totalSquares += rowSquares;
                  }
                  
                  if (isValidBoard && totalSquares === 64) {
                    // 处理稀疏激活数据 - 正确映射到64格棋盘
                    let activationsArray: number[] | undefined = undefined;
                    let maxActivation = 0; // 使用最大激活值而不是总和
                    
                    if (sample.featureActsIndices && sample.featureActsValues && 
                        Array.isArray(sample.featureActsIndices) && Array.isArray(sample.featureActsValues)) {
                      
                      // 创建64格的激活数组
                      activationsArray = new Array(64).fill(0);
                      
                      // 将稀疏激活值映射到正确的棋盘位置，并找到最大激活值
                      for (let i = 0; i < Math.min(sample.featureActsIndices.length, sample.featureActsValues.length); i++) {
                        const index = sample.featureActsIndices[i];
                        const value = sample.featureActsValues[i];
                        
                        // 确保索引在有效范围内
                        if (index >= 0 && index < 64) {
                          activationsArray[index] = value;
                          // 使用最大激活值（与feature页面逻辑一致）
                          if (Math.abs(value) > Math.abs(maxActivation)) {
                            maxActivation = value;
                          }
                        }
                      }
                      
                      console.log('🔍 处理激活数据:', {
                        indicesLength: sample.featureActsIndices.length,
                        valuesLength: sample.featureActsValues.length,
                        nonZeroCount: activationsArray.filter(v => v !== 0).length,
                        maxActivation
                      });
                    }
                    
                    chessSamples.push({
                      fen: trimmed,
                      activationStrength: maxActivation, // 使用最大激活值作为排序依据
                      activations: activationsArray,
                      zPatternIndices: sample.zPatternIndices,
                      zPatternValues: sample.zPatternValues,
                      contextId: sample.contextIdx || sample.context_idx,
                      sampleIndex: sample.sampleIndex || 0
                    });
                    
                    break; // 找到一个有效 FEN 就跳出
                  }
                }
              }
            }
          }
        }
      }
      
      // 按最大激活值排序并取前8个（与feature页面逻辑一致）
      const topSamples = chessSamples
        .sort((a, b) => Math.abs(b.activationStrength) - Math.abs(a.activationStrength))
        .slice(0, 8);
      
      console.log('✅ 获取到 Top Activation 数据:', {
        totalChessSamples: chessSamples.length,
        topSamplesCount: topSamples.length
      });
      
      setTopActivations(topSamples);
      
    } catch (error) {
      console.error('❌ 获取 Top Activation 数据失败:', error);
      setTopActivations([]);
    } finally {
      setLoadingTopActivations(false);
    }
  }, [circuitVisualizationData]);

  // 当点击节点时获取 Top Activation 数据
  useEffect(() => {
    if (clickedNodeId) {
      fetchTopActivations(clickedNodeId);
    } else {
      setTopActivations([]);
    }
  }, [clickedNodeId, fetchTopActivations]);

  // 从circuit trace结果中提取FEN字符串
  const extractFenFromCircuitTrace = useCallback(() => {
    if (!circuitTraceResult?.metadata?.prompt_tokens) return null;
    
    const promptText = circuitTraceResult.metadata.prompt_tokens.join(' ');
    console.log('🔍 搜索FEN字符串:', promptText);
    
    // 更宽松的FEN格式检测
    const lines = promptText.split('\n');
    for (const line of lines) {
      const trimmed = line.trim();
      // 检查是否包含FEN格式 - 包含斜杠且有足够的字符
      if (trimmed.includes('/')) {
        const parts = trimmed.split(/\s+/);
        if (parts.length >= 6) {
          const [boardPart, activeColor] = parts;
          const boardRows = boardPart.split('/');
          
          if (boardRows.length === 8 && /^[wb]$/.test(activeColor)) {
            console.log('✅ 找到FEN字符串:', trimmed);
            return trimmed;
          }
        }
      }
    }
    
    // 如果没找到完整的FEN，尝试更简单的匹配
    const simpleMatch = promptText.match(/[rnbqkpRNBQKP1-8\/]{15,}\s+[wb]\s+[KQkqA-Za-z-]+\s+[a-h][36-]?\s*\d*\s*\d*/);
    if (simpleMatch) {
      console.log('✅ 找到简单FEN匹配:', simpleMatch[0]);
      return simpleMatch[0];
    }
    
    console.log('❌ 未找到FEN字符串');
    return null;
  }, [circuitTraceResult]);


  // 获取节点激活数据
  const getNodeActivationData = useCallback((nodeId: string | null): NodeActivationData => {
    if (!nodeId || !circuitTraceResult) {
      console.log('❌ 缺少必要参数:', { nodeId, hasCircuitTraceResult: !!circuitTraceResult });
      return { activations: undefined, zPatternIndices: undefined, zPatternValues: undefined };
    }
    
    console.log(`🔍 查找节点 ${nodeId} 的激活数据...`);
    console.log('📋 Circuit trace结果结构:', {
      hasActivationInfo: !!circuitTraceResult.activation_info,
      activationInfoKeys: circuitTraceResult.activation_info ? Object.keys(circuitTraceResult.activation_info) : [],
      hasNodes: !!circuitTraceResult.nodes,
      nodesLength: circuitTraceResult.nodes?.length || 0
    });
    
    // 解析 node_id -> rawLayer, featureOrHead, ctx(position)
    const parseFromNodeId = (id: string) => {
      const parts = id.split('_');
      const rawLayer = Number(parts[0]) || 0;
      const featureOrHead = Number(parts[1]) || 0;
      const ctxIdx = Number(parts[2]) || 0;
      // 将原始层号除以2得到真实层号
      const layerForActivation = Math.floor(rawLayer / 2);
      return { rawLayer, layerForActivation, featureOrHead, ctxIdx };
    };
    const parsed = parseFromNodeId(nodeId);

    // 首先确定节点类型
    let featureTypeForNode: string | undefined = undefined;
    if (circuitTraceResult.nodes && Array.isArray(circuitTraceResult.nodes)) {
      const nodeMeta = circuitTraceResult.nodes.find((n: any) => n?.node_id === nodeId);
      featureTypeForNode = nodeMeta?.feature_type;
    }

    console.log('🔍 节点解析信息:', {
      nodeId,
      parsed,
      featureTypeForNode
    });

    // 1) 优先从activation_info中查找激活数据
    if (circuitTraceResult.activation_info) {
      console.log('🔍 从activation_info中查找激活数据...');
      console.log('📋 activation_info结构:', {
        hasActivationInfo: !!circuitTraceResult.activation_info,
        activationInfoKeys: Object.keys(circuitTraceResult.activation_info),
        traceSide,
        hasDirectFeatures: !!circuitTraceResult.activation_info.features,
        sideActivationInfo: circuitTraceResult.activation_info[traceSide]
      });
      
      // 检查是否是合并后的激活信息（直接包含features）
      let featuresToSearch = null;
      if (circuitTraceResult.activation_info.features && Array.isArray(circuitTraceResult.activation_info.features)) {
        // 这是合并后的激活信息，直接使用
        featuresToSearch = circuitTraceResult.activation_info.features;
        console.log(`🔍 使用合并后的激活信息，找到${featuresToSearch.length}个特征`);
      } else {
        // 这是原始的q/k分支结构，根据traceSide选择
        const sideActivationInfo = circuitTraceResult.activation_info[traceSide];
        if (sideActivationInfo && sideActivationInfo.features && Array.isArray(sideActivationInfo.features)) {
          featuresToSearch = sideActivationInfo.features;
          console.log(`🔍 在${traceSide}侧找到${featuresToSearch.length}个特征的激活信息`);
        }
      }
      
      if (featuresToSearch) {
        // 在features数组中查找匹配的特征
        for (const featureInfo of featuresToSearch) {
          const matchesLayer = featureInfo.layer === parsed.layerForActivation;
          const matchesPosition = featureInfo.position === parsed.ctxIdx;
          
          let matchesIndex = false;
          if (featureTypeForNode) {
            const t = featureTypeForNode.toLowerCase();
            if (t === 'lorsa') {
              matchesIndex = featureInfo.head_idx === parsed.featureOrHead;
            } else if (t === 'cross layer transcoder' || t.includes('transcoder')) {
              matchesIndex = featureInfo.feature_idx === parsed.featureOrHead;
            }
          } else {
            // 回退：尝试匹配任一索引
            matchesIndex = (featureInfo.head_idx === parsed.featureOrHead) || 
                          (featureInfo.feature_idx === parsed.featureOrHead);
          }
          
          if (matchesLayer && matchesPosition && matchesIndex) {
            console.log('✅ 在activation_info中找到匹配的特征:', {
              featureId: featureInfo.featureId,
              type: featureInfo.type,
              layer: featureInfo.layer,
              position: featureInfo.position,
              head_idx: featureInfo.head_idx,
              feature_idx: featureInfo.feature_idx,
              hasActivations: !!featureInfo.activations,
              hasZPattern: !!(featureInfo.zPatternIndices && featureInfo.zPatternValues)
            });
            
            return {
              activations: featureInfo.activations,
              zPatternIndices: featureInfo.zPatternIndices,
              zPatternValues: featureInfo.zPatternValues,
              nodeType: featureInfo.type,
              clerp: undefined // activation_info中没有clerp信息
            };
          }
        }
        
        console.log('❌ 在activation_info中未找到匹配的特征');
      } else {
        console.log(`❌ ${traceSide}侧没有activation_info或features数组`);
      }
    }

    // 2) 回退到原有的节点内联字段检查
    let nodesToSearch: any[] = [];
    if (circuitTraceResult.nodes && Array.isArray(circuitTraceResult.nodes)) {
      nodesToSearch = circuitTraceResult.nodes;
    } else if (Array.isArray(circuitTraceResult)) {
      nodesToSearch = circuitTraceResult;
    }

    if (nodesToSearch.length > 0) {
      const exactMatch = nodesToSearch.find(node => node?.node_id === nodeId);
      if (exactMatch) {
        const inlineActs = exactMatch.activations;
        const inlineZIdx = exactMatch.zPatternIndices;
        const inlineZVal = exactMatch.zPatternValues;
        console.log('✅ 节点内联字段检查:', {
          hasInlineActivations: !!inlineActs,
          hasInlineZIdx: !!inlineZIdx,
          hasInlineZVal: !!inlineZVal,
        });
        if (inlineActs || (inlineZIdx && inlineZVal)) {
          return {
            activations: inlineActs,
            zPatternIndices: inlineZIdx,
            zPatternValues: inlineZVal,
            nodeType: exactMatch.feature_type,
            clerp: exactMatch.clerp,
          };
        } 
      }
    }

    // 3) 深度扫描激活记录集合
    const candidateRecords: any[] = [];
    const pushCandidateArrays = (obj: any) => {
      if (!obj) return;
      if (Array.isArray(obj)) {
        for (const item of obj) {
          if (item && typeof item === 'object') {
            const hasActivationShape = ('layer' in item) && ('position' in item) && ('activations' in item);
            const hasZShape = ('zPatternIndices' in item) && ('zPatternValues' in item);
            const hasIndexKey = ('head_idx' in item) || ('feature_idx' in item);
            if (hasActivationShape || hasZShape || hasIndexKey) {
              candidateRecords.push(item);
            }
          }
        }
      } else if (typeof obj === 'object') {
        for (const v of Object.values(obj)) pushCandidateArrays(v);
      }
    };
    pushCandidateArrays(circuitTraceResult);

    console.log('🧭 候选记录数:', candidateRecords.length);

    // 定义匹配函数
    const tryMatchRecord = (rec: any, featureType?: string) => {
      const recLayer = Number(rec?.layer);
      const recPos = Number(rec?.position);
      const recHead = rec?.head_idx;
      const recFeatIdx = rec?.feature_idx;

      const layerOk = !Number.isNaN(recLayer) && recLayer === parsed.layerForActivation;
      const posOk = !Number.isNaN(recPos) && recPos === parsed.ctxIdx;

      let indexOk = false;
      if (featureType) {
        const t = featureType.toLowerCase();
        if (t === 'lorsa') indexOk = recHead === parsed.featureOrHead;
        else if (t === 'cross layer transcoder' || t.includes('transcoder')) indexOk = recFeatIdx === parsed.featureOrHead;
        else indexOk = (recHead === parsed.featureOrHead) || (recFeatIdx === parsed.featureOrHead);
      } else {
        indexOk = (recHead === parsed.featureOrHead) || (recFeatIdx === parsed.featureOrHead);
      }

      return layerOk && posOk && indexOk;
    };

    const matched = candidateRecords.find(rec => tryMatchRecord(rec, featureTypeForNode));
    if (matched) {
      console.log('✅ 通过解析匹配到activation记录:', {
        nodeId,
        layerForActivation: parsed.layerForActivation,
        ctxIdx: parsed.ctxIdx,
        featureOrHead: parsed.featureOrHead,
        featureTypeForNode,
      });
      return {
        activations: matched.activations,
        zPatternIndices: matched.zPatternIndices,
        zPatternValues: matched.zPatternValues,
        nodeType: featureTypeForNode,
        clerp: (nodesToSearch.find(n => n?.node_id === nodeId) || {}).clerp,
      };
    }

    // 4) 最后的模糊匹配
    if (nodesToSearch.length > 0) {
      const fuzzyMatches = nodesToSearch.filter(node => node?.node_id && node.node_id.includes(nodeId.split('_')[0]));
      if (fuzzyMatches.length > 0) {
        const firstMatch = fuzzyMatches[0];
        console.log('🔍 使用模糊匹配节点:', {
          node_id: firstMatch.node_id,
          hasActivations: !!firstMatch.activations,
        });
        return {
          activations: firstMatch.activations,
          zPatternIndices: firstMatch.zPatternIndices,
          zPatternValues: firstMatch.zPatternValues,
          nodeType: firstMatch.feature_type,
          clerp: firstMatch.clerp,
        };
      }
    }

    console.log('❌ 未找到任何匹配的节点/记录');
    return { activations: undefined, zPatternIndices: undefined, zPatternValues: undefined };
  }, [circuitTraceResult, traceSide]);

  // 当lastMove变化时，更新positiveMove（如果为空）
  useEffect(() => {
    if (lastMove && !positiveMove) {
      setPositiveMove(lastMove);
    }
  }, [lastMove, positiveMove]);

  // 当有效分析FEN变化时：只在用户没有输入时才清空待分析移动，避免覆盖用户输入
  const prevEffectiveGameFenRef = useRef<string>(effectiveGameFen);
  useEffect(() => {
    // 只在 FEN 真正变化时才处理
    if (prevEffectiveGameFenRef.current !== effectiveGameFen) {
      // 如果用户已经输入了移动，不清空（保留用户输入）
      // 只有在用户没有输入时才清空，避免自动带入旧移动/缓存
      if (!positiveMove.trim() && !negativeMove.trim()) {
        setPositiveMove('');
        setNegativeMove('');
        setMoveError('');
      }
      prevEffectiveGameFenRef.current = effectiveGameFen;
    }
  }, [effectiveGameFen, positiveMove, negativeMove]);

  return (
    <div className="space-y-6">
      {/* Circuit Trace 控制面板 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Circuit Trace 分析</span>
            <div className="flex gap-2">
              {showRecoveryButton && (
                <Button
                  onClick={handleManualRecovery}
                  disabled={isRecovering}
                  variant="outline"
                  size="sm"
                  className="bg-yellow-50 border-yellow-300 text-yellow-800 hover:bg-yellow-100"
                  title={lastTraceInfo ? `恢复上次trace: ${lastTraceInfo.move_uci} (${new Date(lastTraceInfo.timestamp || 0).toLocaleTimeString()})` : '恢复上次trace结果'}
                >
                  {isRecovering ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      恢复中...
                    </>
                  ) : (
                    <>
                      🔄 恢复结果
                    </>
                  )}
                </Button>
              )}
              <Button
                onClick={() => setShowParamsDialog(true)}
                variant="outline"
                size="sm"
              >
                <Settings className="w-4 h-4 mr-2" />
                参数设置
              </Button>
              <Button
                onClick={() => handleCircuitTrace('positive')}
                disabled={isTracing}
                variant={isTracing ? 'destructive' : 'default'}
                size="sm"
              >
                {isTracing ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Tracing中...
                  </>
                ) : (
                  'Positive Trace'
                )}
              </Button>
              <Button
                onClick={() => handleCircuitTrace('negative')}
                disabled={isTracing}
                variant={isTracing ? 'destructive' : 'outline'}
                size="sm"
              >
                {isTracing ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Tracing中...
                  </>
                ) : (
                  'Negative Trace'
                )}
              </Button>
              <Button
                onClick={() => handleCircuitTrace('both')}
                disabled={isTracing}
                variant={isTracing ? 'destructive' : 'outline'}
                size="sm"
                className="bg-purple-500 hover:bg-purple-600 text-white"
              >
                {isTracing ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Tracing中...
                  </>
                ) : (
                  'Both Trace'
                )}
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {/* Circuit Trace 日志显示 */}
          {(isTracing || traceLogs.length > 0) && (
            <div className="mb-4 rounded-md border border-blue-200 bg-blue-50 p-3 text-sm text-blue-900">
              <div className="mb-2 font-semibold flex items-center justify-between">
                <span>
                  {isTracing ? '🔍 Circuit Tracing 日志' : '📋 Circuit Tracing 日志（已完成）'}
                </span>
                {!isTracing && traceLogs.length > 0 && !circuitVisualizationData && (
                  <span className="text-xs text-blue-600 bg-blue-100 px-2 py-1 rounded">
                    💡 如果结果未显示，请点击上方"🔄 恢复结果"按钮
                  </span>
                )}
              </div>
              <div className="max-h-40 overflow-y-auto rounded bg-blue-100 p-2 text-xs font-mono leading-relaxed">
                {traceLogs.length === 0 ? (
                  <div className="text-blue-700 opacity-80">
                    {isTracing ? '等待日志...' : '暂无日志'}
                  </div>
                ) : (
                  traceLogs.map((log, idx) => (
                    <div key={`${log.timestamp}-${idx}`} className="mb-1">
                      {new Date(log.timestamp * 1000).toLocaleTimeString()} - {log.message}
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
          <div className="space-y-4">
            {/* Side选择框 */}
            <div className="space-y-2">
              <Label htmlFor="side-select" className="text-sm font-medium text-gray-700">
                分析侧选择
              </Label>
              <Select value={traceSide} onValueChange={(v: 'q' | 'k' | 'both') => setTraceSide(v)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="q">Q侧 (Query)</SelectItem>
                  <SelectItem value="k">K侧 (Key)</SelectItem>
                  <SelectItem value="both">Q+K侧 (合并)</SelectItem>
                </SelectContent>
              </Select>
              <div className="text-xs text-gray-500">
                选择要分析的注意力机制侧
              </div>
            </div>
            
            {/* Positive Move输入框 */}
            <div className="space-y-2">
              <Label htmlFor="positive-move-input" className="text-sm font-medium text-gray-700">
                Positive Move (UCI格式，如：e2e4)
              </Label>
              <div className="flex gap-2">
                <Input
                  id="positive-move-input"
                  type="text"
                  placeholder="输入要促进的UCI移动"
                  value={positiveMove}
                  onChange={(e) => {
                    setPositiveMove(e.target.value);
                    setMoveError('');
                    saveCachedPositiveMove(safeDecodeFen(effectiveGameFen), e.target.value);
                  }}
                  className={`font-mono ${moveError && moveError.includes('Positive') ? 'border-red-500' : ''}`}
                />
                <Button
                  onClick={() => {
                    const move = lastMove || '';
                    setPositiveMove(move);
                    if (move) {
                      saveCachedPositiveMove(safeDecodeFen(effectiveGameFen), move);
                    }
                  }}
                  variant="outline"
                  size="sm"
                  disabled={!lastMove}
                >
                  使用最后移动
                </Button>
              </div>
              <div className="text-xs text-gray-500">
                用于Positive Trace和Both Trace（促进此移动）
              </div>
            </div>
            
            {/* Negative Move输入框 */}
            <div className="space-y-2">
              <Label htmlFor="negative-move-input" className="text-sm font-medium text-gray-700">
                Negative Move (UCI格式，如：e2e4)
              </Label>
              <div className="flex gap-2">
                <Input
                  id="negative-move-input"
                  type="text"
                  placeholder="输入要抑制的UCI移动"
                  value={negativeMove}
                  onChange={(e) => {
                    setNegativeMove(e.target.value);
                    setMoveError('');
                    saveCachedNegativeMove(safeDecodeFen(effectiveGameFen), e.target.value);
                  }}
                  className={`font-mono ${moveError && moveError.includes('Negative') ? 'border-red-500' : ''}`}
                />
              </div>
              <div className="text-xs text-gray-500">
                用于Negative Trace和Both Trace（抑制此移动）
              </div>
              {moveError && (
                <p className="text-sm text-red-600">{moveError}</p>
              )}
            </div>
            
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium text-gray-700">分析FEN (移动前):</span>
                <div className="font-mono text-xs bg-blue-50 p-2 rounded mt-1 break-all border border-blue-200">
                  {gameFen}
                </div>
              </div>
              <div>
                <span className="font-medium text-gray-700">当前FEN (移动后):</span>
                <div className="font-mono text-xs bg-green-50 p-2 rounded mt-1 break-all border border-green-200">
                  {currentFen || effectiveGameFen}
                </div>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium text-gray-700">Positive Move:</span>
                <div className="font-mono text-xs bg-green-50 p-2 rounded mt-1 border border-green-200">
                  {positiveMove || lastMove || '暂无移动'}
                </div>
              </div>
              <div>
                <span className="font-medium text-gray-700">Negative Move:</span>
                <div className="font-mono text-xs bg-red-50 p-2 rounded mt-1 border border-red-200">
                  {negativeMove || '暂无移动'}
                </div>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium text-gray-700">将要分析的移动:</span>
                <div className="font-mono text-xs bg-yellow-50 p-2 rounded mt-1 border border-yellow-200">
                  {positiveMove || negativeMove || lastMove || '暂无移动'}
                </div>
              </div>
              <div>
                <span className="font-medium text-gray-700">移动历史:</span>
                <div className="font-mono text-xs bg-gray-100 p-2 rounded mt-1">
                  {gameHistory.length > 0 ? gameHistory.join(' ') : '暂无移动'}
                </div>
              </div>
            </div>
            
            {/* 当前参数显示 */}
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium text-gray-700">最大特征节点数:</span>
                <div className="font-mono text-xs bg-blue-50 p-2 rounded mt-1 border border-blue-200">
                  {circuitParams.max_feature_nodes}
                </div>
              </div>
              <div>
                <span className="font-medium text-gray-700">节点阈值:</span>
                <div className="font-mono text-xs bg-green-50 p-2 rounded mt-1 border border-green-200">
                  {circuitParams.node_threshold}
                </div>
              </div>
              <div>
                <span className="font-medium text-gray-700">边阈值:</span>
                <div className="font-mono text-xs bg-purple-50 p-2 rounded mt-1 border border-purple-200">
                  {circuitParams.edge_threshold}
                </div>
              </div>
              <div>
                <span className="font-medium text-gray-700">最大激活次数:</span>
                <div className="font-mono text-xs bg-orange-50 p-2 rounded mt-1 border border-orange-200">
                  {circuitParams.max_act_times === null ? '无限制' : 
                   circuitParams.max_act_times >= 1000000 ? 
                   `${(circuitParams.max_act_times / 1000000).toFixed(0)}M` : 
                   circuitParams.max_act_times.toLocaleString()}
                </div>
              </div>
            </div>
            
            {!positiveMove && !negativeMove && !lastMove && (
              <div className="text-center py-4 text-gray-500 bg-yellow-50 rounded-lg border border-yellow-200">
                <p>请输入Positive Move或Negative Move（UCI格式）或先走一步棋</p>
                <p className="text-sm mt-1">例如：e2e4, Nf3, O-O (王车易位用e1g1), O-O-O (后翼易位用e1c1)</p>
                <p className="text-sm mt-1 text-purple-600">Both Trace需要同时输入Positive Move和Negative Move</p>
              </div>
            )}

            {/* 显示上次trace信息（仅在有恢复按钮且没有当前可视化时显示） */}
            {showRecoveryButton && lastTraceInfo && !circuitVisualizationData && (
              <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg text-sm">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-yellow-800">💾 发现上次trace记录</span>
                  <span className="text-xs text-yellow-600">
                    {lastTraceInfo.timestamp ? new Date(lastTraceInfo.timestamp).toLocaleString() : '时间未知'}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs text-yellow-700">
                  <div>
                    <span className="font-medium">FEN:</span>
                    <div className="font-mono bg-yellow-100 p-1 rounded mt-1 break-all">
                      {lastTraceInfo.fen}
                    </div>
                  </div>
                  <div>
                    <span className="font-medium">移动:</span>
                    <div className="font-mono bg-yellow-100 p-1 rounded mt-1">
                      {lastTraceInfo.move_uci} ({lastTraceInfo.order_mode}模式, {lastTraceInfo.side}侧)
                    </div>
                  </div>
                </div>
                <div className="text-xs text-yellow-600 mt-2 text-center">
                  点击上方"🔄 恢复结果"按钮尝试恢复该trace的可视化结果
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Circuit可视化区域 */}
      {circuitVisualizationData && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Circuit Trace 可视化</span>
              <div className="flex gap-2">
                <Button
                  onClick={handleSaveGraphJson}
                  variant="outline"
                  size="sm"
                >
                  保存JSON
                </Button>
                <Button
                  onClick={() => {
                    setCircuitVisualizationData(null);
                    setCircuitTraceResult(null);
                    setClickedNodeId(null);
                    setHoveredNodeId(null);
                    setPinnedNodeIds([]);
                    setHiddenNodeIds([]);
                    setSelectedFeature(null);
                    setConnectedFeatures([]);
                    // 清除可视化后重新检查恢复按钮
                    checkRecoverableTrace();
                  }}
                  variant="outline"
                  size="sm"
                >
                  清除可视化
                </Button>
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {/* Circuit Visualization Layout */}
            <div className="space-y-6 w-full max-w-full overflow-hidden">
              {/* Top Row: Link Graph and Node Connections side by side */}
              <div className="flex gap-6 h-[900px] w-full max-w-full overflow-hidden">
                {/* Link Graph Component - Left Side */}
                <div className="flex-1 min-w-0 max-w-full border rounded-lg p-4 bg-white shadow-sm overflow-hidden">
                  <div className="w-full h-full overflow-hidden relative">
                    <LinkGraphContainer 
                      data={circuitVisualizationData} 
                      onNodeClick={handleNodeClick}
                      onNodeHover={handleNodeHover}
                      onFeatureSelect={handleFeatureSelect}
                      onConnectedFeaturesSelect={handleConnectedFeaturesSelect}
                      onConnectedFeaturesLoading={handleConnectedFeaturesLoading}
                      clickedId={clickedNodeId}
                      hoveredId={hoveredNodeId}
                      pinnedIds={pinnedNodeIds}
                    />
                  </div>
                </div>

                {/* Node Connections Component - Right Side */}
                <div className="w-96 flex-shrink-0 border rounded-lg p-4 bg-white shadow-sm overflow-hidden">
                  <NodeConnections
                    data={circuitVisualizationData}
                    clickedId={clickedNodeId}
                    hoveredId={hoveredNodeId}
                    pinnedIds={pinnedNodeIds}
                    hiddenIds={hiddenNodeIds}
                    onFeatureClick={handleNodeClick}
                    onFeatureSelect={handleFeatureSelect}
                    onFeatureHover={handleNodeHover}
                  />
                </div>
              </div>

              {/* Chess Board Display */}
              {(() => {
                const fen = extractFenFromCircuitTrace();
                const nodeActivationData = getNodeActivationData(clickedNodeId);
                
                if (!fen) return null;
                
                return (
                  <div className="w-full border rounded-lg p-4 bg-white shadow-sm">
                    <h3 className="text-lg font-semibold mb-4 text-center">
                      Circuit Trace 棋盘状态
                      {clickedNodeId && nodeActivationData && (
                        <span className="text-sm font-normal text-blue-600 ml-2">
                          (节点: {clickedNodeId}{nodeActivationData.nodeType ? ` - ${nodeActivationData.nodeType.toUpperCase()}` : ''})
                        </span>
                      )}
                    </h3>
                    {clickedNodeId && nodeActivationData && nodeActivationData.activations && (
                      <div className="text-center mb-2 text-sm text-purple-600">
                        激活数据: {nodeActivationData.activations.filter((v: number) => v !== 0).length} 个非零激活
                        {nodeActivationData.zPatternIndices && nodeActivationData.zPatternValues && 
                          `, ${nodeActivationData.zPatternValues.length} 个Z模式连接`
                        }
                      </div>
                    )}
                    <div className="flex justify-center">
                      <ChessBoard
                        fen={fen}
                        size="medium"
                        showCoordinates={true}
                        activations={nodeActivationData?.activations}
                        zPatternIndices={nodeActivationData?.zPatternIndices}
                        zPatternValues={nodeActivationData?.zPatternValues}
                        flip_activation={Boolean(fen && fen.split(' ')[1] === 'b')}
                        sampleIndex={clickedNodeId ? parseInt(clickedNodeId.split('_')[1]) : undefined}
                        analysisName={`${nodeActivationData?.nodeType || 'Circuit Node'} (${traceSide.toUpperCase()}侧)`}
                      />
                    </div>
                  </div>
                );
              })()}

              {/* Bottom Row: Feature Card - 只在没有Top Activation时显示 */}
              {clickedNodeId && topActivations.length === 0 && (() => {
                const currentNode = circuitVisualizationData.nodes.find((node: any) => node.nodeId === clickedNodeId);
                
                if (!currentNode) {
                  console.log('❌ 未找到节点:', clickedNodeId);
                  return null;
                }
                
                console.log('✅ 找到节点:', currentNode);
                
                // 从node_id解析真正的feature ID (格式: layer_featureId_ctxIdx)
                const parseNodeId = (nodeId: string) => {
                  const parts = nodeId.split('_');
                  if (parts.length >= 2) {
                    const rawLayer = parseInt(parts[0]) || 0;
                    return {
                      layerIdx: Math.floor(rawLayer / 2), // 除以2得到实际模型层数
                      featureIndex: parseInt(parts[1]) || 0
                    };
                  }
                  return { layerIdx: 0, featureIndex: 0 };
                };
                
                const { layerIdx, featureIndex } = parseNodeId(currentNode.nodeId);
                const isLorsa = currentNode.feature_type?.toLowerCase() === 'lorsa';
                
                // 根据节点类型构建正确的dictionary名
                let dictionary: string;
                if (isLorsa) {
                  const lorsaAnalysisName = circuitVisualizationData?.metadata?.lorsa_analysis_name;
                  if (lorsaAnalysisName && lorsaAnalysisName.includes('BT4')) {
                    // BT4格式: BT4_lorsa_L{layer}A
                    dictionary = `BT4_lorsa_L${layerIdx}A`;
                  } else {
                    dictionary = lorsaAnalysisName ? lorsaAnalysisName.replace("{}", layerIdx.toString()) : `lc0-lorsa-L${layerIdx}`;
                  }
                } else {
                  const tcAnalysisName = (circuitVisualizationData?.metadata as any)?.tc_analysis_name || circuitVisualizationData?.metadata?.clt_analysis_name;
                  if (tcAnalysisName && tcAnalysisName.includes('BT4')) {
                    // BT4格式: BT4_tc_L{layer}M
                    dictionary = `BT4_tc_L${layerIdx}M`;
                  } else {
                    dictionary = tcAnalysisName ? tcAnalysisName.replace("{}", layerIdx.toString()) : `lc0_L${layerIdx}M_16x_k30_lr2e-03_auxk_sparseadam`;
                  }
                }
                
                const nodeTypeDisplay = isLorsa ? 'LORSA' : 'SAE';
                
                return (
                  <div className="w-full border rounded-lg p-4 bg-white shadow-sm">
                    <div className="flex justify-between items-center mb-4">
                      <h3 className="text-lg font-semibold">Selected Feature Details</h3>
                      <div className="flex items-center space-x-4">
                        {connectedFeatures.length > 0 && (
                          <div className="flex items-center space-x-2">
                            <span className="text-sm text-gray-600">Connected features:</span>
                            <span className="px-2 py-1 bg-green-100 text-green-800 text-sm font-medium rounded-full">
                              {connectedFeatures.length}
                            </span>
                          </div>
                        )}
                        {/* 跳转到Feature页面的链接 */}
                        {currentNode && featureIndex !== undefined && (
                          <Link
                            to={`/features?dictionary=${encodeURIComponent(dictionary)}&featureIndex=${featureIndex}`}
                            className="inline-flex items-center px-3 py-2 bg-blue-500 text-white text-sm font-medium rounded-md hover:bg-blue-600 transition-colors"
                            title={`跳转到L${layerIdx} ${nodeTypeDisplay} Feature #${featureIndex}`}
                          >
                            <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                            </svg>
                            查看L{layerIdx} {nodeTypeDisplay} #{featureIndex}
                          </Link>
                        )}
                      </div>
                    </div>
                    
                    {/* 节点基本信息 */}
                    <div className="mb-4 p-3 bg-gray-50 rounded-lg">
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="font-medium text-gray-700">节点ID:</span>
                          <span className="ml-2 font-mono text-blue-600">{currentNode.nodeId}</span>
                        </div>
                        <div>
                          <span className="font-medium text-gray-700">特征类型:</span>
                          <span className="ml-2 px-2 py-1 bg-blue-100 text-blue-800 text-xs font-medium rounded-full">
                            {currentNode.feature_type || 'Unknown'}
                          </span>
                        </div>
                        <div>
                          <span className="font-medium text-gray-700">层数:</span>
                          <span className="ml-2">{layerIdx}</span>
                        </div>
                        <div>
                          <span className="font-medium text-gray-700">特征索引:</span>
                          <span className="ml-2">{featureIndex}</span>
                        </div>
                        {currentNode.sourceLinks && (
                          <div>
                            <span className="font-medium text-gray-700">出边数:</span>
                            <span className="ml-2">{currentNode.sourceLinks.length}</span>
                          </div>
                        )}
                        {currentNode.targetLinks && (
                          <div>
                            <span className="font-medium text-gray-700">入边数:</span>
                            <span className="ml-2">{currentNode.targetLinks.length}</span>
                          </div>
                        )}
                      </div>
                    </div>
                    
                    {selectedFeature ? (
                      <FeatureCard feature={selectedFeature} />
                    ) : (
                      <div className="flex items-center justify-center p-8 bg-gray-50 border rounded-lg">
                        <div className="text-center">
                          <p className="text-gray-600 mb-2">No feature is available for this node</p>
                          <p className="text-sm text-gray-500">
                            点击上方的"查看L{layerIdx} {nodeTypeDisplay} #{featureIndex}"链接查看详细信息
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })()}
            </div>
          </CardContent>
        </Card>
      )}

      {/* 保留原有的简单circuitTraceResult显示，但移除跳转按钮 */}
      {circuitTraceResult && !circuitVisualizationData && (
        <Card>
          <CardHeader>
            <CardTitle>Circuit Trace Result</CardTitle>
          </CardHeader>
          <CardContent>
            {circuitTraceResult.nodes ? (
              <div className="space-y-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="font-medium text-blue-900 mb-2">分析摘要</h4>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-blue-700">节点数量:</span>
                      <span className="ml-2 font-mono">{circuitTraceResult.nodes.length}</span>
                    </div>
                    <div>
                      <span className="text-blue-700">连接数量:</span>
                      <span className="ml-2 font-mono">{circuitTraceResult.links?.length || 0}</span>
                    </div>
                    {circuitTraceResult.metadata?.target_move && (
                      <div>
                        <span className="text-blue-700">目标移动:</span>
                        <span className="ml-2 font-mono text-green-600">{circuitTraceResult.metadata.target_move}</span>
                      </div>
                    )}
                  </div>
                </div>
                
                <div className="space-y-2">
                  <h4 className="font-medium">关键节点 (前10个)</h4>
                  {circuitTraceResult.nodes.slice(0, 10).map((node: any, index: number) => (
                    <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                      <div className="flex items-center space-x-2">
                        <span className="font-mono text-sm">{node.node_id}</span>
                        <Badge variant="outline" className="text-xs">
                          {node.feature_type}
                        </Badge>
                      </div>
                      {node.node_id && (
                        <Link
                          to={`/features?nodeId=${encodeURIComponent(node.node_id)}`}
                          className="text-blue-600 underline text-sm hover:text-blue-800"
                        >
                          查看Feature
                        </Link>
                      )}
                    </div>
                  ))}
                  {circuitTraceResult.nodes.length > 10 && (
                    <div className="text-center text-sm text-gray-500">
                      还有 {circuitTraceResult.nodes.length - 10} 个节点
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="text-center py-4 text-gray-500">
                无节点数据
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Top Activation Section */}
      {clickedNodeId && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Top Activation 棋盘</span>
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-600">节点: {clickedNodeId}</span>
                {loadingTopActivations && (
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                    <span className="text-sm text-gray-500">加载中...</span>
                  </div>
                )}
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {loadingTopActivations ? (
              <div className="flex items-center justify-center py-8">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
                  <p className="text-gray-600">正在获取 Top Activation 数据...</p>
                </div>
              </div>
            ) : topActivations.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {topActivations.map((sample, index) => (
                  <div key={index} className="bg-gray-50 rounded-lg p-3 border">
                    <div className="text-center mb-2">
                      <div className="text-sm font-medium text-gray-700">
                        Top #{index + 1}
                      </div>
                      <div className="text-xs text-gray-500">
                        最大激活值: {sample.activationStrength.toFixed(3)}
                      </div>
                    </div>
                    <ChessBoard
                      fen={sample.fen}
                      size="small"
                      showCoordinates={false}
                      activations={sample.activations}
                      zPatternIndices={sample.zPatternIndices}
                      zPatternValues={sample.zPatternValues}
                      sampleIndex={sample.sampleIndex}
                      analysisName={`Context ${sample.contextId}`}
                      flip_activation={Boolean(sample.fen && sample.fen.split(' ')[1] === 'b')}
                      autoFlipWhenBlack={true}
                    />
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <p>未找到包含棋盘的激活样本</p>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* 参数设置对话框 */}
      <Dialog open={showParamsDialog} onOpenChange={setShowParamsDialog}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Settings className="w-5 h-5" />
              Circuit Trace 参数设置
            </DialogTitle>
          </DialogHeader>
          
          <div className="space-y-6 py-4">
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="max_feature_nodes">最大特征节点数 (Max Feature Nodes)</Label>
                <Input
                  id="max_feature_nodes"
                  type="number"
                  min="1"
                  max="10000"
                  step="1"
                  value={circuitParams.max_feature_nodes}
                  onChange={(e) => handleParamsChange('max_feature_nodes', e.target.value)}
                  className="font-mono"
                />
                <p className="text-xs text-gray-500">
                  控制circuit trace中考虑的最大特征节点数量。默认值: 4096
                </p>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="node_threshold">节点阈值 (Node Threshold)</Label>
                <Input
                  id="node_threshold"
                  type="number"
                  min="0"
                  max="1"
                  step="0.01"
                  value={circuitParams.node_threshold}
                  onChange={(e) => handleParamsChange('node_threshold', e.target.value)}
                  className="font-mono"
                />
                <p className="text-xs text-gray-500">
                  节点重要性阈值，用于过滤不重要的节点。默认值: 0.73
                </p>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="edge_threshold">边阈值 (Edge Threshold)</Label>
                <Input
                  id="edge_threshold"
                  type="number"
                  min="0"
                  max="1"
                  step="0.01"
                  value={circuitParams.edge_threshold}
                  onChange={(e) => handleParamsChange('edge_threshold', e.target.value)}
                  className="font-mono"
                />
                <p className="text-xs text-gray-500">
                  边重要性阈值，用于过滤不重要的连接。默认值: 0.57
                </p>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="max_act_times">最大激活次数 (Max Activation Times)</Label>
                <Input
                  id="max_act_times"
                  type="number"
                  min="10000000"
                  max="100000000"
                  step="10000000"
                  value={circuitParams.max_act_times || ''}
                  onChange={(e) => handleParamsChange('max_act_times', e.target.value)}
                  className="font-mono"
                  placeholder="留空表示无限制"
                />
                <p className="text-xs text-gray-500">
                  过滤dense feature。范围：10M-100M，留空表示无限制
                </p>
              </div>
            </div>
            
            {/* 当前参数预览 */}
            <div className="bg-gray-50 p-4 rounded-lg space-y-2">
              <h4 className="font-medium text-sm text-gray-700">当前参数预览:</h4>
              <div className="grid grid-cols-1 gap-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-600">最大特征节点数:</span>
                  <span className="font-mono text-blue-600">{circuitParams.max_feature_nodes}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">节点阈值:</span>
                  <span className="font-mono text-green-600">{circuitParams.node_threshold}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">边阈值:</span>
                  <span className="font-mono text-purple-600">{circuitParams.edge_threshold}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">最大激活次数:</span>
                  <span className="font-mono text-orange-600">
                    {circuitParams.max_act_times === null ? '无限制' : 
                     circuitParams.max_act_times >= 1000000 ? 
                     `${(circuitParams.max_act_times / 1000000).toFixed(0)}M` : 
                     circuitParams.max_act_times.toLocaleString()}
                  </span>
                </div>
              </div>
            </div>
          </div>
          
          <DialogFooter className="flex gap-2">
            <Button
              variant="outline"
              onClick={() => setShowParamsDialog(false)}
            >
              取消
            </Button>
            <Button
              onClick={() => {
                // 重置为默认值
                setCircuitParams({
                  max_feature_nodes: 4096,
                  node_threshold: 0.73,
                  edge_threshold: 0.57,
                  max_act_times: null,
                });
              }}
              variant="outline"
            >
              重置默认
            </Button>
            <Button
              onClick={handleSaveParams}
            >
              保存设置
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};