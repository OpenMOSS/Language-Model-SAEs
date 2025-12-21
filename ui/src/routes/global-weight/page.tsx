import React, { useState, useEffect, useCallback } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { AppNavbar } from '@/components/app/navbar';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Loader2, ArrowLeft, ChevronDown, ChevronUp } from 'lucide-react';
import { SaeComboLoader } from '@/components/common/SaeComboLoader';
import { ChessBoard } from '@/components/chess/chess-board';
import { CircuitInterpretationCard } from '@/components/circuits/circuit-interpretation-card';

interface GlobalWeightFeature {
  name: string;
  weight: number;
  clerp?: string;
  rank?: number;
}

interface GlobalWeightResult {
  feature_type: string;
  layer_idx: number;
  feature_idx: number;
  activation_type?: string;
  feature_name: string;
  features_in: GlobalWeightFeature[];
  features_out: GlobalWeightFeature[];
}

const LOCAL_STORAGE_KEY = "bt4_sae_combo_id";

export const GlobalWeightPage: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  
  // 从URL参数获取初始值，如果没有则从localStorage读取
  const initialFeatureType = searchParams.get('feature_type') || 'tc';
  const initialLayerIdx = parseInt(searchParams.get('layer_idx') || '0');
  const initialFeatureIdx = parseInt(searchParams.get('feature_idx') || '0');
  const urlSaeComboId = searchParams.get('sae_combo_id');
  const storedSaeComboId = typeof window !== 'undefined' ? window.localStorage.getItem(LOCAL_STORAGE_KEY) : null;
  const initialSaeComboId = urlSaeComboId || storedSaeComboId || undefined;
  
  const [featureType, setFeatureType] = useState<'tc' | 'lorsa'>(initialFeatureType as 'tc' | 'lorsa');
  const [layerIdx, setLayerIdx] = useState<number>(initialLayerIdx);
  const [featureIdx, setFeatureIdx] = useState<number>(initialFeatureIdx);
  const [saeComboId, setSaeComboId] = useState<string | undefined>(initialSaeComboId);
  const [k, setK] = useState<number>(100);
  const [activationType, setActivationType] = useState<'max' | 'mean'>('mean');
  
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState<string | null>(null);
  const [result, setResult] = useState<GlobalWeightResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  // 特征详情相关状态
  const [selectedFeatureName, setSelectedFeatureName] = useState<string | null>(null);
  const [selectedFeatureInfo, setSelectedFeatureInfo] = useState<{
    layerIdx: number;
    featureIdx: number;
    featureType: 'tc' | 'lorsa';
  } | null>(null);
  const [topActivations, setTopActivations] = useState<any[]>([]);
  const [loadingTopActivations, setLoadingTopActivations] = useState(false);
  const [editingClerp, setEditingClerp] = useState<string>('');
  const [isSavingClerp, setIsSavingClerp] = useState(false);
  const [syncingClerp, setSyncingClerp] = useState(false);
  const [featuresWithClerp, setFeaturesWithClerp] = useState<Map<string, { clerp: string; rank: number }>>(new Map());
  const [loadingClerps, setLoadingClerps] = useState(false);
  
  // 当前特征的折叠状态
  const [currentFeatureExpanded, setCurrentFeatureExpanded] = useState<boolean>(false);
  const [currentFeatureTopActivations, setCurrentFeatureTopActivations] = useState<any[]>([]);
  const [loadingCurrentFeatureTopActivations, setLoadingCurrentFeatureTopActivations] = useState(false);
  const [currentFeatureInterpretation, setCurrentFeatureInterpretation] = useState<string>('');
  const [isSavingCurrentFeatureInterpretation, setIsSavingCurrentFeatureInterpretation] = useState(false);
  const [syncingCurrentFeatureInterpretation, setSyncingCurrentFeatureInterpretation] = useState(false);
  
  // 特征查询相关状态
  const [searchFeatureType, setSearchFeatureType] = useState<'tc' | 'lorsa'>('tc');
  const [searchLayerIdx, setSearchLayerIdx] = useState<number>(0);
  const [searchFeatureIdx, setSearchFeatureIdx] = useState<number>(0);
  const [searchResult, setSearchResult] = useState<{
    foundInFeaturesIn: boolean;
    foundInFeaturesOut: boolean;
    featuresInInfo: { rank: number; weight: number; name: string } | null;
    featuresOutInfo: { rank: number; weight: number; name: string } | null;
  } | null>(null);

  // 当URL参数变化时更新状态
  useEffect(() => {
    const featureTypeParam = searchParams.get('feature_type');
    const layerIdxParam = searchParams.get('layer_idx');
    const featureIdxParam = searchParams.get('feature_idx');
    const saeComboIdParam = searchParams.get('sae_combo_id');
    
    if (featureTypeParam) setFeatureType(featureTypeParam as 'tc' | 'lorsa');
    if (layerIdxParam) setLayerIdx(parseInt(layerIdxParam));
    if (featureIdxParam) setFeatureIdx(parseInt(featureIdxParam));
    if (saeComboIdParam) {
      setSaeComboId(saeComboIdParam);
    } else {
      // 如果URL中没有sae_combo_id，从localStorage读取
      const stored = typeof window !== 'undefined' ? window.localStorage.getItem(LOCAL_STORAGE_KEY) : null;
      if (stored) {
        setSaeComboId(stored);
      }
    }
  }, [searchParams]);

  // 监听localStorage变化，以便在SaeComboLoader加载新组合时自动更新
  useEffect(() => {
    const handleStorageChange = () => {
      const stored = typeof window !== 'undefined' ? window.localStorage.getItem(LOCAL_STORAGE_KEY) : null;
      if (stored && stored !== saeComboId) {
        setSaeComboId(stored);
      }
    };

    // 监听storage事件（跨标签页同步）
    window.addEventListener('storage', handleStorageChange);
    
    // 轮询检查localStorage（同标签页内，因为storage事件只在跨标签页时触发）
    const interval = setInterval(() => {
      handleStorageChange();
    }, 1000);

    return () => {
      window.removeEventListener('storage', handleStorageChange);
      clearInterval(interval);
    };
  }, [saeComboId]);

  const preloadModels = async (comboId: string): Promise<void> => {
    setLoadingMessage('正在检查模型加载状态...');
    
    // 先检查是否正在加载
    const checkLoadingStatus = async (): Promise<{ isLoading: boolean; logs?: Array<{ timestamp: number; message: string }> }> => {
      try {
        const logParams = new URLSearchParams({
          model_name: 'lc0/BT4-1024x15x32h',
          sae_combo_id: comboId,
        });
        const logRes = await fetch(`${import.meta.env.VITE_BACKEND_URL}/circuit/loading_logs?${logParams.toString()}`);
        if (logRes.ok) {
          const logData: { is_loading?: boolean; logs?: Array<{ timestamp: number; message: string }> } = await logRes.json();
          return { isLoading: logData.is_loading ?? false, logs: logData.logs };
        }
      } catch (err) {
        console.warn('Failed to check loading status:', err);
      }
      return { isLoading: false };
    };

    // 如果正在加载，等待加载完成
    let status = await checkLoadingStatus();
    if (status.isLoading) {
      setLoadingMessage('检测到模型正在加载中，等待加载完成...');
      // 轮询等待加载完成
      const maxWaitTime = 300000; // 最多等待5分钟
      const startTime = Date.now();
      let lastLogCount = status.logs?.length ?? 0;
      while (status.isLoading && Date.now() - startTime < maxWaitTime) {
        await new Promise(resolve => setTimeout(resolve, 2000)); // 每2秒检查一次
        status = await checkLoadingStatus();
        // 如果有新的日志，显示最后一条
        if (status.logs && status.logs.length > lastLogCount) {
          const lastLog = status.logs[status.logs.length - 1];
          setLoadingMessage(`加载中: ${lastLog.message}`);
          lastLogCount = status.logs.length;
        }
      }
      if (status.isLoading) {
        throw new Error('模型加载超时，请稍后重试');
      }
      setLoadingMessage('模型加载完成，准备计算全局权重...');
      return; // 加载已完成
    }

    // 调用预加载接口
    setLoadingMessage('开始预加载模型...');
    const preloadRes = await fetch(`${import.meta.env.VITE_BACKEND_URL}/circuit/preload_models`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model_name: 'lc0/BT4-1024x15x32h',
        sae_combo_id: comboId,
      }),
    });

    if (!preloadRes.ok) {
      const errorText = await preloadRes.text();
      throw new Error(`预加载失败: HTTP ${preloadRes.status}: ${errorText}`);
    }

    const preloadData = await preloadRes.json();
    
    // 如果返回 already_loaded，说明已经加载完成
    if (preloadData.status === 'already_loaded') {
      setLoadingMessage('模型已加载，准备计算全局权重...');
      return;
    }

    // 如果返回 loaded 或 loading，等待加载完成
    if (preloadData.status === 'loaded' || preloadData.status === 'loading') {
      setLoadingMessage('等待模型加载完成...');
      const maxWaitTime = 300000; // 最多等待5分钟
      const startTime = Date.now();
      let lastLogCount = 0;
      while (Date.now() - startTime < maxWaitTime) {
        status = await checkLoadingStatus();
        // 如果有新的日志，显示最后一条
        if (status.logs && status.logs.length > lastLogCount) {
          const lastLog = status.logs[status.logs.length - 1];
          setLoadingMessage(`加载中: ${lastLog.message}`);
          lastLogCount = status.logs.length;
        }
        if (!status.isLoading) {
          // 再等待一下确保加载完成
          await new Promise(resolve => setTimeout(resolve, 1000));
          setLoadingMessage('模型加载完成，准备计算全局权重...');
          return;
        }
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
      throw new Error('模型加载超时，请稍后重试');
    }
  };

  const fetchGlobalWeight = async () => {
    setLoading(true);
    setLoadingMessage(null);
    setError(null);
    
    try {
      // 确保使用最新的 sae_combo_id（优先使用状态中的，否则从 localStorage 读取）
      const currentSaeComboId = saeComboId || (typeof window !== 'undefined' ? window.localStorage.getItem(LOCAL_STORAGE_KEY) : null);
      
      if (!currentSaeComboId) {
        throw new Error('请先选择 SAE 组合');
      }

      const params = new URLSearchParams({
        model_name: 'lc0/BT4-1024x15x32h',
        feature_type: featureType,
        layer_idx: layerIdx.toString(),
        feature_idx: featureIdx.toString(),
        k: k.toString(),
        activation_type: activationType,
      });
      
      params.append('sae_combo_id', currentSaeComboId);
      
      let response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/global_weight?${params.toString()}`);
      
      // 如果返回503错误（模型未加载），自动尝试预加载
      if (response.status === 503) {
        const errorText = await response.text();
        console.log('检测到模型未加载，开始自动预加载...', errorText);
        
        try {
          setLoadingMessage('模型未加载，正在自动加载...');
          await preloadModels(currentSaeComboId);
          console.log('预加载完成，重试获取全局权重...');
          setLoadingMessage('正在计算全局权重...');
          
          // 重试请求
          response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/global_weight?${params.toString()}`);
        } catch (preloadErr: any) {
          throw new Error(`自动预加载失败: ${preloadErr.message}`);
        }
      }
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }
      
      const data = await response.json() as GlobalWeightResult;
      setResult(data);
      
      // 批量获取所有 features 的 clerp 和计算排名
      await fetchBatchClerpsAndRanks(data.features_in, data.features_out);
      
      // 更新URL参数（使用实际使用的 sae_combo_id）
      const newParams = new URLSearchParams({
        feature_type: featureType,
        layer_idx: layerIdx.toString(),
        feature_idx: featureIdx.toString(),
      });
      newParams.append('sae_combo_id', currentSaeComboId);
      // 同步更新状态
      setSaeComboId(currentSaeComboId);
      setSearchParams(newParams);
    } catch (err: any) {
      setError(err.message || '获取全局权重失败');
      console.error('Error fetching global weight:', err);
    } finally {
      setLoading(false);
      setLoadingMessage(null);
    }
  };

  useEffect(() => {
    // 如果URL中有参数，自动加载
    if (searchParams.get('layer_idx') && searchParams.get('feature_idx')) {
      fetchGlobalWeight();
    }
  }, []); // 只在组件挂载时执行一次

  // 解析特征名称，提取层、特征索引和类型
  const parseFeatureName = useCallback((featureName: string): {
    layerIdx: number;
    featureIdx: number;
    featureType: 'tc' | 'lorsa';
  } | null => {
    // 格式: BT4_tc_L0M_k30_e16#123 或 BT4_lorsa_L0A_k30_e16#123
    const match = featureName.match(/BT4_(tc|lorsa)_L(\d+)(M|A)_k30_e16#(\d+)/);
    if (match) {
      const [, type, layer, , idx] = match;
      return {
        layerIdx: parseInt(layer),
        featureIdx: parseInt(idx),
        featureType: type as 'tc' | 'lorsa',
      };
    }
    return null;
  }, []);

  // 获取字典名（根据层和类型）
  const getDictionaryName = useCallback((layerIdx: number, isLorsa: boolean): string => {
    // 根据组合ID构建字典名
    // 格式: BT4_lorsa_L{layer}A_k30_e16 或 BT4_tc_L{layer}M_k30_e16
    if (isLorsa) {
      return `BT4_lorsa_L${layerIdx}A_k30_e16`;
    } else {
      return `BT4_tc_L${layerIdx}M_k30_e16`;
    }
  }, []);

  // 获取 SAE 名称（用于 Circuit Interpretation）
  const getSaeNameForCircuit = useCallback((layer: number, isLorsa: boolean): string => {
    return getDictionaryName(layer, isLorsa);
  }, [getDictionaryName]);

  // 获取 Top Activation 数据（用于选中的特征）
  const fetchTopActivations = useCallback(async (layerIdx: number, featureIdx: number, isLorsa: boolean) => {
    setLoadingTopActivations(true);
    try {
      const dictionary = getDictionaryName(layerIdx, isLorsa);
      
      console.log('🔍 获取 Top Activation 数据:', {
        layerIdx,
        featureIdx,
        dictionary,
        isLorsa
      });
      
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}/features/${featureIdx}`,
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
      
      const camelData = camelcaseKeys(decoded as Record<string, unknown>, {
        deep: true,
        stopPaths: ["sample_groups.samples.context"],
      }) as any;
      
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
            
            if (trimmed.includes('/')) {
              const parts = trimmed.split(/\s+/);
              
              if (parts.length >= 6) {
                const [boardPart, activeColor] = parts;
                const boardRows = boardPart.split('/');
                
                if (boardRows.length === 8 && /^[wb]$/.test(activeColor)) {
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
                    let activationsArray: number[] | undefined = undefined;
                    let maxActivation = 0;
                    
                    if (sample.featureActsIndices && sample.featureActsValues && 
                        Array.isArray(sample.featureActsIndices) && Array.isArray(sample.featureActsValues)) {
                      
                      activationsArray = new Array(64).fill(0);
                      
                      for (let i = 0; i < Math.min(sample.featureActsIndices.length, sample.featureActsValues.length); i++) {
                        const index = sample.featureActsIndices[i];
                        const value = sample.featureActsValues[i];
                        
                        if (index >= 0 && index < 64) {
                          activationsArray[index] = value;
                          if (Math.abs(value) > Math.abs(maxActivation)) {
                            maxActivation = value;
                          }
                        }
                      }
                    }
                    
                    chessSamples.push({
                      fen: trimmed,
                      activationStrength: maxActivation,
                      activations: activationsArray,
                      zPatternIndices: sample.zPatternIndices,
                      zPatternValues: sample.zPatternValues,
                      contextId: sample.contextIdx || sample.context_idx,
                      sampleIndex: sample.sampleIndex || 0
                    });
                    
                    break;
                  }
                }
              }
            }
          }
        }
      }
      
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
  }, [getDictionaryName]);

  // 获取 Top Activation 数据（用于当前特征或指定特征）
  const fetchTopActivationsForFeature = useCallback(async (
    layerIdx: number, 
    featureIdx: number, 
    isLorsa: boolean,
    isCurrentFeature: boolean = false
  ) => {
    if (isCurrentFeature) {
      setLoadingCurrentFeatureTopActivations(true);
    } else {
      setLoadingTopActivations(true);
    }
    
    try {
      const dictionary = getDictionaryName(layerIdx, isLorsa);
      
      console.log('🔍 获取 Top Activation 数据:', {
        layerIdx,
        featureIdx,
        dictionary,
        isLorsa,
        isCurrentFeature
      });
      
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}/features/${featureIdx}`,
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
      
      const camelData = camelcaseKeys(decoded as Record<string, unknown>, {
        deep: true,
        stopPaths: ["sample_groups.samples.context"],
      }) as any;
      
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
            
            if (trimmed.includes('/')) {
              const parts = trimmed.split(/\s+/);
              
              if (parts.length >= 6) {
                const [boardPart, activeColor] = parts;
                const boardRows = boardPart.split('/');
                
                if (boardRows.length === 8 && /^[wb]$/.test(activeColor)) {
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
                    let activationsArray: number[] | undefined = undefined;
                    let maxActivation = 0;
                    
                    if (sample.featureActsIndices && sample.featureActsValues && 
                        Array.isArray(sample.featureActsIndices) && Array.isArray(sample.featureActsValues)) {
                      
                      activationsArray = new Array(64).fill(0);
                      
                      for (let i = 0; i < Math.min(sample.featureActsIndices.length, sample.featureActsValues.length); i++) {
                        const index = sample.featureActsIndices[i];
                        const value = sample.featureActsValues[i];
                        
                        if (index >= 0 && index < 64) {
                          activationsArray[index] = value;
                          if (Math.abs(value) > Math.abs(maxActivation)) {
                            maxActivation = value;
                          }
                        }
                      }
                    }
                    
                    chessSamples.push({
                      fen: trimmed,
                      activationStrength: maxActivation,
                      activations: activationsArray,
                      zPatternIndices: sample.zPatternIndices,
                      zPatternValues: sample.zPatternValues,
                      contextId: sample.contextIdx || sample.context_idx,
                      sampleIndex: sample.sampleIndex || 0
                    });
                    
                    break;
                  }
                }
              }
            }
          }
        }
      }
      
      const topSamples = chessSamples
        .sort((a, b) => Math.abs(b.activationStrength) - Math.abs(a.activationStrength))
        .slice(0, 8);
      
      console.log('✅ 获取到 Top Activation 数据:', {
        totalChessSamples: chessSamples.length,
        topSamplesCount: topSamples.length,
        isCurrentFeature
      });
      
      if (isCurrentFeature) {
        setCurrentFeatureTopActivations(topSamples);
      } else {
        setTopActivations(topSamples);
      }
      
    } catch (error) {
      console.error('❌ 获取 Top Activation 数据失败:', error);
      if (isCurrentFeature) {
        setCurrentFeatureTopActivations([]);
      } else {
        setTopActivations([]);
      }
    } finally {
      if (isCurrentFeature) {
        setLoadingCurrentFeatureTopActivations(false);
      } else {
        setLoadingTopActivations(false);
      }
    }
  }, [getDictionaryName]);

  // 批量获取 interpretation 并计算排名
  const fetchBatchClerpsAndRanks = useCallback(async (
    featuresIn: GlobalWeightFeature[],
    featuresOut: GlobalWeightFeature[]
  ) => {
    setLoadingClerps(true);
    try {
      // 解析所有 features，构建 nodes 数组
      const allFeatures = [...featuresIn, ...featuresOut];
      const nodes: Array<{
        node_id: string;
        feature: number;
        layer: number;
        feature_type: string;
      }> = [];
      const featureMap = new Map<string, { layerIdx: number; featureIdx: number; featureType: 'tc' | 'lorsa' }>();

      for (const feature of allFeatures) {
        const parsed = parseFeatureName(feature.name);
        if (parsed) {
          const nodeId = `${parsed.layerIdx * 2}_${parsed.featureIdx}_0`;
          nodes.push({
            node_id: nodeId,
            feature: parsed.featureIdx,
            layer: parsed.layerIdx,
            feature_type: parsed.featureType
          });
          featureMap.set(feature.name, parsed);
        }
      }

      // 按 feature_type 分组
      const lorsaNodes = nodes.filter(n => n.feature_type === 'lorsa');
      const tcNodes = nodes.filter(n => n.feature_type === 'tc');

      const clerpMap = new Map<string, string>();
      const rankMap = new Map<string, number>();

      // 批量获取 Lorsa clerps
      if (lorsaNodes.length > 0) {
        try {
          const lorsaResponse = await fetch(
            `${import.meta.env.VITE_BACKEND_URL}/circuit/sync_interpretations_to_clerps`,
            {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                nodes: lorsaNodes,
                lorsa_analysis_name: 'BT4_lorsa_k30_e16',
              })
            }
          );

          if (lorsaResponse.ok) {
            const lorsaResult = await lorsaResponse.json();
            if (lorsaResult.updated_nodes) {
              for (const node of lorsaResult.updated_nodes) {
                const featureName = allFeatures.find(f => {
                  const parsed = parseFeatureName(f.name);
                  return parsed && `${parsed.layerIdx * 2}_${parsed.featureIdx}_0` === node.node_id;
                })?.name;
                if (featureName && node.clerp) {
                  clerpMap.set(featureName, node.clerp);
                }
              }
            }
          }
        } catch (err) {
          console.warn('获取 Lorsa clerps 失败:', err);
        }
      }

      // 批量获取 TC clerps
      if (tcNodes.length > 0) {
        try {
          const tcResponse = await fetch(
            `${import.meta.env.VITE_BACKEND_URL}/circuit/sync_interpretations_to_clerps`,
            {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                nodes: tcNodes,
                tc_analysis_name: 'BT4_tc_k30_e16',
              })
            }
          );

          if (tcResponse.ok) {
            const tcResult = await tcResponse.json();
            if (tcResult.updated_nodes) {
              for (const node of tcResult.updated_nodes) {
                const featureName = allFeatures.find(f => {
                  const parsed = parseFeatureName(f.name);
                  return parsed && `${parsed.layerIdx * 2}_${parsed.featureIdx}_0` === node.node_id;
                })?.name;
                if (featureName && node.clerp) {
                  clerpMap.set(featureName, node.clerp);
                }
              }
            }
          }
        } catch (err) {
          console.warn('获取 TC clerps 失败:', err);
        }
      }

      // 计算排名（Lorsa 和 Transcoder 分开）
      // 对于 features_in 和 features_out 分别计算排名
      const lorsaFeaturesIn = featuresIn.filter(f => {
        const parsed = parseFeatureName(f.name);
        return parsed && parsed.featureType === 'lorsa';
      }).sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));

      const tcFeaturesIn = featuresIn.filter(f => {
        const parsed = parseFeatureName(f.name);
        return parsed && parsed.featureType === 'tc';
      }).sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));

      const lorsaFeaturesOut = featuresOut.filter(f => {
        const parsed = parseFeatureName(f.name);
        return parsed && parsed.featureType === 'lorsa';
      }).sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));

      const tcFeaturesOut = featuresOut.filter(f => {
        const parsed = parseFeatureName(f.name);
        return parsed && parsed.featureType === 'tc';
      }).sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));

      // 设置排名
      lorsaFeaturesIn.forEach((f, idx) => {
        rankMap.set(f.name, idx + 1);
      });
      tcFeaturesIn.forEach((f, idx) => {
        rankMap.set(f.name, idx + 1);
      });
      lorsaFeaturesOut.forEach((f, idx) => {
        rankMap.set(f.name, idx + 1);
      });
      tcFeaturesOut.forEach((f, idx) => {
        rankMap.set(f.name, idx + 1);
      });

      // 合并 clerp 和排名信息
      const combinedMap = new Map<string, { clerp: string; rank: number }>();
      for (const feature of allFeatures) {
        const clerp = clerpMap.get(feature.name) || '';
        const rank = rankMap.get(feature.name) || 0;
        combinedMap.set(feature.name, { clerp, rank });
      }

      setFeaturesWithClerp(combinedMap);
    } catch (error) {
      console.error('❌ 批量获取 clerp 和排名失败:', error);
    } finally {
      setLoadingClerps(false);
    }
  }, [parseFeatureName]);

  // 获取 Interpretation 从 MongoDB（用于选中的特征）
  const fetchClerp = useCallback(async (layerIdx: number, featureIdx: number, isLorsa: boolean) => {
    setSyncingClerp(true);
    try {
      // 构建 analysis_name
      const analysisName = isLorsa ? 'BT4_lorsa_k30_e16' : 'BT4_tc_k30_e16';
      
      // 构建节点数据
      const node = {
        node_id: `${layerIdx * 2}_${featureIdx}_0`,
        feature: featureIdx,
        layer: layerIdx,
        feature_type: isLorsa ? 'lorsa' : 'tc'
      };
      
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/circuit/sync_interpretations_to_clerps`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            nodes: [node],
            lorsa_analysis_name: isLorsa ? analysisName : undefined,
            tc_analysis_name: !isLorsa ? analysisName : undefined
          })
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }
      
      const result = await response.json();
      
      // 查找对应的 clerp
      const updatedNode = result.updated_nodes?.find((n: any) => n.node_id === node.node_id);
      if (updatedNode && updatedNode.clerp) {
        setEditingClerp(updatedNode.clerp);
      } else {
        setEditingClerp('');
      }
      
    } catch (error) {
      console.error('❌ 获取 interpretation 失败:', error);
      setEditingClerp('');
    } finally {
      setSyncingClerp(false);
    }
  }, [saeComboId]);

  // 获取 Interpretation 从 MongoDB（用于当前特征或选中的特征）
  const fetchInterpretationForFeature = useCallback(async (
    layerIdx: number, 
    featureIdx: number, 
    isLorsa: boolean,
    isCurrentFeature: boolean = false
  ) => {
    if (isCurrentFeature) {
      setSyncingCurrentFeatureInterpretation(true);
    } else {
      setSyncingClerp(true);
    }
    
    try {
      // 构建 analysis_name
      const analysisName = isLorsa ? 'BT4_lorsa_k30_e16' : 'BT4_tc_k30_e16';
      
      // 构建节点数据
      const node = {
        node_id: `${layerIdx * 2}_${featureIdx}_0`,
        feature: featureIdx,
        layer: layerIdx,
        feature_type: isLorsa ? 'lorsa' : 'tc'
      };
      
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/circuit/sync_interpretations_to_clerps`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            nodes: [node],
            lorsa_analysis_name: isLorsa ? analysisName : undefined,
            tc_analysis_name: !isLorsa ? analysisName : undefined
          })
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }
      
      const result = await response.json();
      
      // 查找对应的 interpretation
      const updatedNode = result.updated_nodes?.find((n: any) => n.node_id === node.node_id);
      const interpretation = updatedNode?.clerp || '';
      
      if (isCurrentFeature) {
        setCurrentFeatureInterpretation(interpretation);
      } else {
        setEditingClerp(interpretation);
      }
      
    } catch (error) {
      console.error('❌ 获取 interpretation 失败:', error);
      if (isCurrentFeature) {
        setCurrentFeatureInterpretation('');
      } else {
        setEditingClerp('');
      }
    } finally {
      if (isCurrentFeature) {
        setSyncingCurrentFeatureInterpretation(false);
      } else {
        setSyncingClerp(false);
      }
    }
  }, [saeComboId]);

  // 保存 Interpretation 到 MongoDB（用于选中的特征）
  const saveClerp = useCallback(async (layerIdx: number, featureIdx: number, isLorsa: boolean, clerpText: string) => {
    setIsSavingClerp(true);
    try {
      const analysisName = isLorsa ? 'BT4_lorsa_k30_e16' : 'BT4_tc_k30_e16';
      
      const node = {
        node_id: `${layerIdx * 2}_${featureIdx}_0`,
        clerp: clerpText,
        feature: featureIdx,
        layer: layerIdx,
        feature_type: isLorsa ? 'lorsa' : 'tc'
      };
      
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/circuit/sync_clerps_to_interpretations`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            nodes: [node],
            lorsa_analysis_name: isLorsa ? analysisName : undefined,
            tc_analysis_name: !isLorsa ? analysisName : undefined
          })
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }
      
      await response.json();
      
      // 更新 featuresWithClerp 状态
      if (selectedFeatureName) {
        setFeaturesWithClerp(prev => {
          const newMap = new Map(prev);
          const existing = newMap.get(selectedFeatureName);
          if (existing) {
            newMap.set(selectedFeatureName, {
              ...existing,
              clerp: clerpText
            });
          } else {
            newMap.set(selectedFeatureName, {
              clerp: clerpText,
              rank: 0
            });
          }
          return newMap;
        });
      }
      
      alert(`✅ Interpretation已成功保存到MongoDB！`);
      
    } catch (error) {
      console.error('❌ 保存 interpretation 失败:', error);
      alert(`❌ 保存失败: ${error instanceof Error ? error.message : '未知错误'}`);
    } finally {
      setIsSavingClerp(false);
    }
  }, [saeComboId, selectedFeatureName]);

  // 保存 Interpretation 到 MongoDB（用于当前特征或选中的特征）
  const saveInterpretationForFeature = useCallback(async (
    layerIdx: number, 
    featureIdx: number, 
    isLorsa: boolean, 
    interpretationText: string,
    isCurrentFeature: boolean = false
  ) => {
    if (isCurrentFeature) {
      setIsSavingCurrentFeatureInterpretation(true);
    } else {
      setIsSavingClerp(true);
    }
    
    try {
      const analysisName = isLorsa ? 'BT4_lorsa_k30_e16' : 'BT4_tc_k30_e16';
      
      const node = {
        node_id: `${layerIdx * 2}_${featureIdx}_0`,
        clerp: interpretationText,
        feature: featureIdx,
        layer: layerIdx,
        feature_type: isLorsa ? 'lorsa' : 'tc'
      };
      
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/circuit/sync_clerps_to_interpretations`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            nodes: [node],
            lorsa_analysis_name: isLorsa ? analysisName : undefined,
            tc_analysis_name: !isLorsa ? analysisName : undefined
          })
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }
      
      await response.json();
      
      // 更新 featuresWithClerp 状态（如果当前特征在列表中）
      if (isCurrentFeature && result) {
        const allFeatures = [...result.features_in, ...result.features_out];
        const currentFeatureName = allFeatures.find(f => {
          const parsed = parseFeatureName(f.name);
          return parsed && parsed.layerIdx === layerIdx && parsed.featureIdx === featureIdx;
        })?.name;
        
        if (currentFeatureName) {
          setFeaturesWithClerp(prev => {
            const newMap = new Map(prev);
            const existing = newMap.get(currentFeatureName);
            if (existing) {
              newMap.set(currentFeatureName, {
                ...existing,
                clerp: interpretationText
              });
            } else {
              newMap.set(currentFeatureName, {
                clerp: interpretationText,
                rank: 0
              });
            }
            return newMap;
          });
        }
      } else if (selectedFeatureName) {
        // 更新 featuresWithClerp 状态
        setFeaturesWithClerp(prev => {
          const newMap = new Map(prev);
          const existing = newMap.get(selectedFeatureName);
          if (existing) {
            newMap.set(selectedFeatureName, {
              ...existing,
              clerp: interpretationText
            });
          } else {
            newMap.set(selectedFeatureName, {
              clerp: interpretationText,
              rank: 0
            });
          }
          return newMap;
        });
      }
      
      alert(`✅ Interpretation已成功保存到MongoDB！`);
      
    } catch (error) {
      console.error('❌ 保存 interpretation 失败:', error);
      alert(`❌ 保存失败: ${error instanceof Error ? error.message : '未知错误'}`);
    } finally {
      if (isCurrentFeature) {
        setIsSavingCurrentFeatureInterpretation(false);
      } else {
        setIsSavingClerp(false);
      }
    }
  }, [saeComboId, result, parseFeatureName, selectedFeatureName]);

  // 当 result 更新时，自动加载当前特征的 Top Activation 和 Interpretation
  useEffect(() => {
    if (result) {
      const isLorsa = result.feature_type === 'lorsa';
      fetchTopActivationsForFeature(result.layer_idx, result.feature_idx, isLorsa, true);
      fetchInterpretationForFeature(result.layer_idx, result.feature_idx, isLorsa, true);
    }
  }, [result, fetchTopActivationsForFeature, fetchInterpretationForFeature]);

  const handleFeatureClick = (featureName: string) => {
    const parsed = parseFeatureName(featureName);
    if (parsed) {
      setSelectedFeatureName(featureName);
      setSelectedFeatureInfo(parsed);
      
      // 获取 Top Activation 和 clerp
      fetchTopActivations(parsed.layerIdx, parsed.featureIdx, parsed.featureType === 'lorsa');
      fetchClerp(parsed.layerIdx, parsed.featureIdx, parsed.featureType === 'lorsa');
    }
  };

  // 查询特征在 Features In 和 Features Out 中的排名和权重
  const searchFeature = useCallback(() => {
    if (!result) {
      alert('请先计算全局权重');
      return;
    }

    // 构建特征名称
    const featureTypePrefix = searchFeatureType === 'lorsa' ? 'BT4_lorsa' : 'BT4_tc';
    const layerSuffix = searchFeatureType === 'lorsa' ? 'A' : 'M';
    const featureName = `${featureTypePrefix}_L${searchLayerIdx}${layerSuffix}_k30_e16#${searchFeatureIdx}`;

    // 在 Features In 中查找
    const featuresInMatch = result.features_in.find(f => f.name === featureName);
    const featuresInInfo = featuresInMatch ? {
      rank: featuresWithClerp.get(featureName)?.rank || 0,
      weight: featuresInMatch.weight,
      name: featureName
    } : null;

    // 在 Features Out 中查找
    const featuresOutMatch = result.features_out.find(f => f.name === featureName);
    const featuresOutInfo = featuresOutMatch ? {
      rank: featuresWithClerp.get(featureName)?.rank || 0,
      weight: featuresOutMatch.weight,
      name: featureName
    } : null;

    setSearchResult({
      foundInFeaturesIn: !!featuresInMatch,
      foundInFeaturesOut: !!featuresOutMatch,
      featuresInInfo,
      featuresOutInfo
    });

    // 如果找到了，自动选中该特征
    if (featuresInMatch || featuresOutMatch) {
      const parsed = parseFeatureName(featureName);
      if (parsed) {
        setSelectedFeatureName(featureName);
        setSelectedFeatureInfo(parsed);
        
        // 获取 Top Activation 和 clerp
        fetchTopActivations(parsed.layerIdx, parsed.featureIdx, parsed.featureType === 'lorsa');
        fetchClerp(parsed.layerIdx, parsed.featureIdx, parsed.featureType === 'lorsa');
      }
    }
  }, [result, searchFeatureType, searchLayerIdx, searchFeatureIdx, featuresWithClerp, parseFeatureName, fetchTopActivations, fetchClerp]);

  return (
    <div className="min-h-screen bg-background">
      <AppNavbar />
      <div className="container mx-auto p-6 space-y-6">
        {/* 全局 BT4 SAE 组合选择（LoRSA / Transcoder） */}
        <SaeComboLoader />
        
        <div className="flex items-center gap-4">
          <Button
            variant="outline"
            onClick={() => navigate(-1)}
            className="flex items-center gap-2"
          >
            <ArrowLeft className="w-4 h-4" />
            返回
          </Button>
          <h1 className="text-3xl font-bold">全局权重分析</h1>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>参数设置</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
              <div className="space-y-2">
                <Label htmlFor="feature_type">特征类型</Label>
                <Select
                  value={featureType}
                  onValueChange={(value) => setFeatureType(value as 'tc' | 'lorsa')}
                >
                  <SelectTrigger id="feature_type">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="tc">TC (Transcoder)</SelectItem>
                    <SelectItem value="lorsa">LoRSA</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="layer_idx">层索引</Label>
                <Input
                  id="layer_idx"
                  type="number"
                  min="0"
                  max="14"
                  value={layerIdx}
                  onChange={(e) => setLayerIdx(parseInt(e.target.value) || 0)}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="feature_idx">特征索引</Label>
                <Input
                  id="feature_idx"
                  type="number"
                  min="0"
                  value={featureIdx}
                  onChange={(e) => setFeatureIdx(parseInt(e.target.value) || 0)}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="activation_type">激活类型</Label>
                <Select
                  value={activationType}
                  onValueChange={(value) => setActivationType(value as 'max' | 'mean')}
                >
                  <SelectTrigger id="activation_type">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="max">Max Activation</SelectItem>
                    <SelectItem value="mean">Mean Activation</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="k">Top K</Label>
                <Input
                  id="k"
                  type="number"
                  min="1"
                  max="500"
                  value={k}
                  onChange={(e) => setK(parseInt(e.target.value) || 100)}
                />
              </div>
            </div>

            <div className="mt-4 space-y-2">
              <Button onClick={fetchGlobalWeight} disabled={loading}>
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    {loadingMessage || '计算中...'}
                  </>
                ) : (
                  '计算全局权重'
                )}
              </Button>
              {loadingMessage && loadingMessage !== '计算中...' && (
                <p className="text-sm text-muted-foreground">{loadingMessage}</p>
              )}
            </div>
          </CardContent>
        </Card>

        {error && (
          <Card className="border-red-500">
            <CardContent className="pt-6">
              <p className="text-red-500">{error}</p>
            </CardContent>
          </Card>
        )}

        {result && (
          <div className="space-y-6">
            {/* 特征查询卡片 */}
            <Card className="border-blue-200 bg-blue-50/50">
              <CardHeader>
                <CardTitle>查询特征排名和权重</CardTitle>
                <p className="text-sm text-muted-foreground">
                  在 Features In 和 Features Out 中查找指定特征的排名和权重
                </p>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="search_feature_type">特征类型</Label>
                    <Select
                      value={searchFeatureType}
                      onValueChange={(value) => setSearchFeatureType(value as 'tc' | 'lorsa')}
                    >
                      <SelectTrigger id="search_feature_type">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="tc">TC (Transcoder)</SelectItem>
                        <SelectItem value="lorsa">LoRSA</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="search_layer_idx">层索引</Label>
                    <Input
                      id="search_layer_idx"
                      type="number"
                      min="0"
                      max="14"
                      value={searchLayerIdx}
                      onChange={(e) => setSearchLayerIdx(parseInt(e.target.value) || 0)}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="search_feature_idx">特征索引</Label>
                    <Input
                      id="search_feature_idx"
                      type="number"
                      min="0"
                      value={searchFeatureIdx}
                      onChange={(e) => setSearchFeatureIdx(parseInt(e.target.value) || 0)}
                    />
                  </div>

                  <div className="space-y-2 flex items-end">
                    <Button onClick={searchFeature} className="w-full">
                      查询
                    </Button>
                  </div>
                </div>

                {/* 查询结果 */}
                {searchResult && (
                  <div className="mt-4 p-4 bg-white rounded-lg border border-blue-200">
                    <h4 className="font-semibold mb-3 text-blue-900">查询结果</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className={`p-3 rounded ${searchResult.foundInFeaturesIn ? 'bg-green-50 border border-green-300' : 'bg-gray-50 border border-gray-300'}`}>
                        <div className="font-medium text-sm mb-2">
                          Features In (输入特征)
                        </div>
                        {searchResult.foundInFeaturesIn && searchResult.featuresInInfo ? (
                          <div className="space-y-1 text-sm">
                            <div>
                              <span className="font-medium">特征名称:</span>{' '}
                              <span className="font-mono text-xs">{searchResult.featuresInInfo.name}</span>
                            </div>
                            <div>
                              <span className="font-medium">排名:</span>{' '}
                              <span className="text-green-700 font-semibold">
                                #{searchResult.featuresInInfo.rank > 0 ? searchResult.featuresInInfo.rank : '未计算'}
                              </span>
                            </div>
                            <div>
                              <span className="font-medium">权重:</span>{' '}
                              <span className="text-green-700 font-semibold">
                                {searchResult.featuresInInfo.weight.toFixed(6)}
                              </span>
                            </div>
                          </div>
                        ) : (
                          <div className="text-sm text-gray-500">未找到该特征</div>
                        )}
                      </div>

                      <div className={`p-3 rounded ${searchResult.foundInFeaturesOut ? 'bg-green-50 border border-green-300' : 'bg-gray-50 border border-gray-300'}`}>
                        <div className="font-medium text-sm mb-2">
                          Features Out (输出特征)
                        </div>
                        {searchResult.foundInFeaturesOut && searchResult.featuresOutInfo ? (
                          <div className="space-y-1 text-sm">
                            <div>
                              <span className="font-medium">特征名称:</span>{' '}
                              <span className="font-mono text-xs">{searchResult.featuresOutInfo.name}</span>
                            </div>
                            <div>
                              <span className="font-medium">排名:</span>{' '}
                              <span className="text-green-700 font-semibold">
                                #{searchResult.featuresOutInfo.rank > 0 ? searchResult.featuresOutInfo.rank : '未计算'}
                              </span>
                            </div>
                            <div>
                              <span className="font-medium">权重:</span>{' '}
                              <span className="text-green-700 font-semibold">
                                {searchResult.featuresOutInfo.weight.toFixed(6)}
                              </span>
                            </div>
                          </div>
                        ) : (
                          <div className="text-sm text-gray-500">未找到该特征</div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>当前特征: {result.feature_name}</CardTitle>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setCurrentFeatureExpanded(!currentFeatureExpanded)}
                    className="flex items-center gap-2"
                  >
                    {currentFeatureExpanded ? (
                      <>
                        <ChevronUp className="w-4 h-4" />
                        折叠
                      </>
                    ) : (
                      <>
                        <ChevronDown className="w-4 h-4" />
                        展开
                      </>
                    )}
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <p><strong>类型:</strong> {result.feature_type === 'tc' ? 'TC (Transcoder)' : 'LoRSA'}</p>
                  <p><strong>层:</strong> {result.layer_idx}</p>
                  <p><strong>特征索引:</strong> {result.feature_idx}</p>
                  {result.activation_type && (
                    <p><strong>激活类型:</strong> {result.activation_type === 'max' ? 'Max Activation' : 'Mean Activation'}</p>
                  )}
                </div>
                
                {currentFeatureExpanded && (
                  <div className="mt-6 space-y-6 pt-6 border-t">
                    {/* Top Activation 部分 */}
                    <div>
                      <h3 className="text-lg font-semibold mb-4">Top Activation 棋盘</h3>
                      {loadingCurrentFeatureTopActivations ? (
                        <div className="flex items-center justify-center py-8">
                          <div className="text-center">
                            <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4" />
                            <p className="text-gray-600">正在获取 Top Activation 数据...</p>
                          </div>
                        </div>
                      ) : currentFeatureTopActivations.length > 0 ? (
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                          {currentFeatureTopActivations.map((sample, index) => (
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
                    </div>

                    {/* Circuit Interpretation 部分 */}
                    {result && saeComboId && (
                      <div>
                        <CircuitInterpretationCard
                          node={{
                            nodeId: `${result.layer_idx * 2}_${result.feature_idx}_0`,
                            layer: result.layer_idx,
                            feature: result.feature_idx,
                            feature_type: result.feature_type,
                          }}
                          saeComboId={saeComboId}
                          saeSeries="BT4-exp128"
                          getSaeName={getSaeNameForCircuit}
                        />
                      </div>
                    )}

                    {/* Interpretation Editor 部分 */}
                    <div>
                      <h3 className="text-lg font-semibold mb-4">Interpretation Editor</h3>
                      <div className="space-y-4">
                        <div className="flex justify-between items-center">
                          <Label htmlFor="current-feature-interpretation">Interpretation 内容</Label>
                          <div className="text-xs text-gray-500">
                            字符数: {currentFeatureInterpretation.length}
                          </div>
                        </div>
                        <textarea
                          id="current-feature-interpretation"
                          value={currentFeatureInterpretation}
                          onChange={(e) => setCurrentFeatureInterpretation(e.target.value)}
                          className="w-full h-32 px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 font-mono text-sm"
                          placeholder="输入或编辑特征的 interpretation 内容..."
                        />
                        <div className="flex justify-end space-x-2">
                          <Button
                            variant="outline"
                            onClick={() => {
                              fetchInterpretationForFeature(
                                result.layer_idx,
                                result.feature_idx,
                                result.feature_type === 'lorsa',
                                true
                              );
                            }}
                            disabled={syncingCurrentFeatureInterpretation}
                          >
                            {syncingCurrentFeatureInterpretation ? (
                              <>
                                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                同步中...
                              </>
                            ) : (
                              '从 MongoDB 同步'
                            )}
                          </Button>
                          <Button
                            onClick={() => {
                              saveInterpretationForFeature(
                                result.layer_idx,
                                result.feature_idx,
                                result.feature_type === 'lorsa',
                                currentFeatureInterpretation,
                                true
                              );
                            }}
                            disabled={isSavingCurrentFeatureInterpretation}
                          >
                            {isSavingCurrentFeatureInterpretation ? (
                              <>
                                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                保存中...
                              </>
                            ) : (
                              '保存到 MongoDB'
                            )}
                          </Button>
                        </div>
                        <div className="text-xs text-blue-600 bg-blue-50 p-3 rounded border-l-4 border-blue-200">
                          <div className="font-medium mb-1">💡 使用说明:</div>
                          <ul className="list-disc list-inside space-y-1 text-blue-700">
                            <li>编辑 interpretation 内容后点击"保存到 MongoDB"将更改同步到数据库</li>
                            <li>点击"从 MongoDB 同步"可以从数据库读取最新的 interpretation 内容</li>
                            <li>Interpretation 会保存到对应特征的 interpretation 字段中</li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>输入特征 (Features In)</CardTitle>
                  <p className="text-sm text-muted-foreground">
                    影响当前特征的前序特征（Top {result.features_in.length}）
                  </p>
                </CardHeader>
                <CardContent>
                  {loadingClerps && (
                    <div className="text-sm text-muted-foreground mb-2 flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      正在加载 Interpretation 和排名...
                    </div>
                  )}
                  <div className="space-y-2 max-h-[600px] overflow-y-auto">
                    {result.features_in.map((feature, idx) => {
                      const featureInfo = featuresWithClerp.get(feature.name);
                      const clerp = featureInfo?.clerp || '';
                      const rank = featureInfo?.rank || 0;
                      const parsed = parseFeatureName(feature.name);
                      const featureTypeLabel = parsed?.featureType === 'lorsa' ? 'LoRSA' : 'TC';
                      
                      // 检查是否是查询结果
                      const isSearchResult = searchResult && (
                        (searchResult.foundInFeaturesIn && searchResult.featuresInInfo?.name === feature.name)
                      );
                      
                      return (
                        <div
                          key={idx}
                          className={`p-2 rounded border hover:bg-muted cursor-pointer transition-colors ${
                            selectedFeatureName === feature.name ? 'bg-blue-100 border-blue-500' : ''
                          } ${
                            isSearchResult ? 'bg-green-100 border-green-500 ring-2 ring-green-300' : ''
                          }`}
                          onClick={() => handleFeatureClick(feature.name)}
                          title="点击查看该特征的 Top Activation"
                        >
                          <div className="flex items-center justify-between mb-1">
                            <span className="font-mono text-sm flex-1 break-all">{feature.name}</span>
                            <div className="flex items-center gap-3 ml-4 whitespace-nowrap">
                              {rank > 0 && (
                                <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded">
                                  {featureTypeLabel} #{rank}
                                </span>
                              )}
                              <span className="text-right font-semibold">
                                {feature.weight.toFixed(4)}
                              </span>
                            </div>
                          </div>
                          {clerp && (
                            <div className="text-xs text-muted-foreground mt-1 line-clamp-2">
                              {clerp}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>输出特征 (Features Out)</CardTitle>
                  <p className="text-sm text-muted-foreground">
                    受当前特征影响的后序特征（Top {result.features_out.length}）
                  </p>
                </CardHeader>
                <CardContent>
                  {loadingClerps && (
                    <div className="text-sm text-muted-foreground mb-2 flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      正在加载 Interpretation 和排名...
                    </div>
                  )}
                  <div className="space-y-2 max-h-[600px] overflow-y-auto">
                    {result.features_out.map((feature, idx) => {
                      const featureInfo = featuresWithClerp.get(feature.name);
                      const clerp = featureInfo?.clerp || '';
                      const rank = featureInfo?.rank || 0;
                      const parsed = parseFeatureName(feature.name);
                      const featureTypeLabel = parsed?.featureType === 'lorsa' ? 'LoRSA' : 'TC';
                      
                      // 检查是否是查询结果
                      const isSearchResult = searchResult && (
                        (searchResult.foundInFeaturesOut && searchResult.featuresOutInfo?.name === feature.name)
                      );
                      
                      return (
                        <div
                          key={idx}
                          className={`p-2 rounded border hover:bg-muted cursor-pointer transition-colors ${
                            selectedFeatureName === feature.name ? 'bg-blue-100 border-blue-500' : ''
                          } ${
                            isSearchResult ? 'bg-green-100 border-green-500 ring-2 ring-green-300' : ''
                          }`}
                          onClick={() => handleFeatureClick(feature.name)}
                          title="点击查看该特征的 Top Activation"
                        >
                          <div className="flex items-center justify-between mb-1">
                            <span className="font-mono text-sm flex-1 break-all">{feature.name}</span>
                            <div className="flex items-center gap-3 ml-4 whitespace-nowrap">
                              {rank > 0 && (
                                <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded">
                                  {featureTypeLabel} #{rank}
                                </span>
                              )}
                              <span className="text-right font-semibold">
                                {feature.weight.toFixed(4)}
                              </span>
                            </div>
                          </div>
                          {clerp && (
                            <div className="text-xs text-muted-foreground mt-1 line-clamp-2">
                              {clerp}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        )}

        {/* 特征详情面板：Top Activation 和 Clerp 编辑器 */}
        {selectedFeatureName && selectedFeatureInfo && (
          <Card className="mt-6">
            <CardHeader>
              <CardTitle>特征详情: {selectedFeatureName}</CardTitle>
              <p className="text-sm text-muted-foreground">
                层: {selectedFeatureInfo.layerIdx}, 特征索引: {selectedFeatureInfo.featureIdx}, 
                类型: {selectedFeatureInfo.featureType === 'tc' ? 'TC (Transcoder)' : 'LoRSA'}
              </p>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Top Activation 部分 */}
              <div>
                <h3 className="text-lg font-semibold mb-4">Top Activation 棋盘</h3>
                {loadingTopActivations ? (
                  <div className="flex items-center justify-center py-8">
                    <div className="text-center">
                      <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4" />
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
              </div>

              {/* Circuit Interpretation 部分 */}
              {selectedFeatureInfo && saeComboId && (
                <div>
                  <CircuitInterpretationCard
                    node={{
                      nodeId: `${selectedFeatureInfo.layerIdx * 2}_${selectedFeatureInfo.featureIdx}_0`,
                      layer: selectedFeatureInfo.layerIdx,
                      feature: selectedFeatureInfo.featureIdx,
                      feature_type: selectedFeatureInfo.featureType,
                    }}
                    saeComboId={saeComboId}
                    saeSeries="BT4-exp128"
                    getSaeName={getSaeNameForCircuit}
                  />
                </div>
              )}

              {/* Interpretation Editor 部分 */}
              <div>
                <h3 className="text-lg font-semibold mb-4">Interpretation Editor</h3>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <Label htmlFor="interpretation-editor">Interpretation 内容</Label>
                    <div className="text-xs text-gray-500">
                      字符数: {editingClerp.length}
                    </div>
                  </div>
                  <textarea
                    id="interpretation-editor"
                    value={editingClerp}
                    onChange={(e) => setEditingClerp(e.target.value)}
                    className="w-full h-32 px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 font-mono text-sm"
                    placeholder="输入或编辑特征的 interpretation 内容..."
                  />
                  <div className="flex justify-end space-x-2">
                    <Button
                      variant="outline"
                      onClick={() => {
                        fetchClerp(
                          selectedFeatureInfo.layerIdx,
                          selectedFeatureInfo.featureIdx,
                          selectedFeatureInfo.featureType === 'lorsa'
                        );
                      }}
                      disabled={syncingClerp}
                    >
                      {syncingClerp ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          同步中...
                        </>
                      ) : (
                        '从 MongoDB 同步'
                      )}
                    </Button>
                    <Button
                      onClick={() => {
                        saveClerp(
                          selectedFeatureInfo.layerIdx,
                          selectedFeatureInfo.featureIdx,
                          selectedFeatureInfo.featureType === 'lorsa',
                          editingClerp
                        );
                      }}
                      disabled={isSavingClerp}
                    >
                      {isSavingClerp ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          保存中...
                        </>
                      ) : (
                        '保存到 MongoDB'
                      )}
                    </Button>
                  </div>
                  <div className="text-xs text-blue-600 bg-blue-50 p-3 rounded border-l-4 border-blue-200">
                    <div className="font-medium mb-1">💡 使用说明:</div>
                    <ul className="list-disc list-inside space-y-1 text-blue-700">
                      <li>编辑 interpretation 内容后点击"保存到 MongoDB"将更改同步到数据库</li>
                      <li>点击"从 MongoDB 同步"可以从数据库读取最新的 interpretation 内容</li>
                      <li>Interpretation 会保存到对应特征的 interpretation 字段中</li>
                    </ul>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};
