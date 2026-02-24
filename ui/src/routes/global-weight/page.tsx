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
  const [featuresInLayerFilter, setFeaturesInLayerFilter] = useState<string>('');
  const [featuresOutLayerFilter, setFeaturesOutLayerFilter] = useState<string>('');
  
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState<string | null>(null);
  const [result, setResult] = useState<GlobalWeightResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  
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
  
  const [currentFeatureExpanded, setCurrentFeatureExpanded] = useState<boolean>(false);
  const [currentFeatureTopActivations, setCurrentFeatureTopActivations] = useState<any[]>([]);
  const [loadingCurrentFeatureTopActivations, setLoadingCurrentFeatureTopActivations] = useState(false);
  const [currentFeatureInterpretation, setCurrentFeatureInterpretation] = useState<string>('');
  const [isSavingCurrentFeatureInterpretation, setIsSavingCurrentFeatureInterpretation] = useState(false);
  const [syncingCurrentFeatureInterpretation, setSyncingCurrentFeatureInterpretation] = useState(false);
  
  const [searchFeatureType, setSearchFeatureType] = useState<'tc' | 'lorsa'>('tc');
  const [searchLayerIdx, setSearchLayerIdx] = useState<number>(0);
  const [searchFeatureIdx, setSearchFeatureIdx] = useState<number>(0);
  const [searchResult, setSearchResult] = useState<{
    foundInFeaturesIn: boolean;
    foundInFeaturesOut: boolean;
    featuresInInfo: { rank: number; weight: number; name: string } | null;
    featuresOutInfo: { rank: number; weight: number; name: string } | null;
  } | null>(null);

  useEffect(() => {
    const featureTypeParam = searchParams.get('feature_type');
    const layerIdxParam = searchParams.get('layer_idx');
    const featureIdxParam = searchParams.get('feature_idx');
    const saeComboIdParam = searchParams.get('sae_combo_id');
    const featuresInLayerFilterParam = searchParams.get('features_in_layer_filter');
    const featuresOutLayerFilterParam = searchParams.get('features_out_layer_filter');

    if (featureTypeParam) setFeatureType(featureTypeParam as 'tc' | 'lorsa');
    if (layerIdxParam) setLayerIdx(parseInt(layerIdxParam));
    if (featureIdxParam) setFeatureIdx(parseInt(featureIdxParam));
    if (featuresInLayerFilterParam) setFeaturesInLayerFilter(featuresInLayerFilterParam);
    if (featuresOutLayerFilterParam) setFeaturesOutLayerFilter(featuresOutLayerFilterParam);
    if (saeComboIdParam) {
      setSaeComboId(saeComboIdParam);
    } else {
      // If there is no sae_combo_id in the URL, read from localStorage
      const stored = typeof window !== 'undefined' ? window.localStorage.getItem(LOCAL_STORAGE_KEY) : null;
      if (stored) {
        setSaeComboId(stored);
      }
    }
  }, [searchParams]);

  // Listen for localStorage changes, so that SaeComboLoader can automatically update when a new combination is loaded
  useEffect(() => {
    const handleStorageChange = () => {
      const stored = typeof window !== 'undefined' ? window.localStorage.getItem(LOCAL_STORAGE_KEY) : null;
      if (stored && stored !== saeComboId) {
        setSaeComboId(stored);
      }
    };

    // Listen for storage events (cross-tab synchronization)
    window.addEventListener('storage', handleStorageChange);
    
    // Poll localStorage (within the same tab, because the storage event is only triggered across tabs)
    const interval = setInterval(() => {
      handleStorageChange();
    }, 1000);

    return () => {
      window.removeEventListener('storage', handleStorageChange);
      clearInterval(interval);
    };
  }, [saeComboId]);

  const preloadModels = async (comboId: string): Promise<void> => {
    setLoadingMessage('Checking model loading status...');
    
    // First check if it is loading
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

    // If it is loading, wait for it to complete
    let status = await checkLoadingStatus();
    if (status.isLoading) {
      setLoadingMessage('Detected model is loading, waiting for it to complete...');
      // Poll to wait for it to complete
      const maxWaitTime = 300000; // Maximum wait time is 5 minutes
      const startTime = Date.now();
      let lastLogCount = status.logs?.length ?? 0;
      while (status.isLoading && Date.now() - startTime < maxWaitTime) {
        await new Promise(resolve => setTimeout(resolve, 2000)); // Check every 2 seconds
        status = await checkLoadingStatus();
        // If there is a new log, display the last one
        if (status.logs && status.logs.length > lastLogCount) {
          const lastLog = status.logs[status.logs.length - 1];
          setLoadingMessage(`Loading: ${lastLog.message}`);
          lastLogCount = status.logs.length;
        }
      }
      if (status.isLoading) {
        throw new Error('Model loading timeout, please try again later');
      }
      setLoadingMessage('Model loaded, preparing to calculate global weights...');
      return; // Loading completed
    }

    // Call the preload interface
    setLoadingMessage('Starting model preload...');
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
      throw new Error(`Preload failed: HTTP ${preloadRes.status}: ${errorText}`);
    }

    const preloadData = await preloadRes.json();
    
    // If it returns already_loaded, it means it has been loaded
    if (preloadData.status === 'already_loaded') {
      setLoadingMessage('Model loaded, preparing to calculate global weights...');
      return;
    }

    // If it returns loaded or loading, wait for it to complete
    if (preloadData.status === 'loaded' || preloadData.status === 'loading') {
      setLoadingMessage('Waiting for model to load...');
      const maxWaitTime = 300000; // Maximum wait time is 5 minutes
      const startTime = Date.now();
      let lastLogCount = 0;
      while (Date.now() - startTime < maxWaitTime) {
        status = await checkLoadingStatus();
        // If there is a new log, display the last one
        if (status.logs && status.logs.length > lastLogCount) {
          const lastLog = status.logs[status.logs.length - 1];
          setLoadingMessage(`Loading: ${lastLog.message}`);
          lastLogCount = status.logs.length;
        }
        if (!status.isLoading) {
          // Wait again to ensure it is loaded
          await new Promise(resolve => setTimeout(resolve, 1000));
          setLoadingMessage('Model loaded, preparing to calculate global weights...');
          return;
        }
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
      throw new Error('Model loading timeout, please try again later');
    }
  };

  const fetchGlobalWeight = async () => {
    setLoading(true);
    setLoadingMessage(null);
    setError(null);
    
    try {
      // Ensure using the latest sae_combo_id (use the one in the state first, otherwise read from localStorage)
      const currentSaeComboId = saeComboId || (typeof window !== 'undefined' ? window.localStorage.getItem(LOCAL_STORAGE_KEY) : null);
      
      if (!currentSaeComboId) {
        throw new Error('Please select a SAE combination first');
      }

      const params = new URLSearchParams({
        model_name: 'lc0/BT4-1024x15x32h',
        feature_type: featureType,
        layer_idx: layerIdx.toString(),
        feature_idx: featureIdx.toString(),
        k: k.toString(),
        activation_type: activationType,
      });

      // Add layer filter parameters
      if (featuresInLayerFilter.trim()) {
        params.append('features_in_layer_filter', featuresInLayerFilter.trim());
      }
      if (featuresOutLayerFilter.trim()) {
        params.append('features_out_layer_filter', featuresOutLayerFilter.trim());
      }

      params.append('sae_combo_id', currentSaeComboId);
      
      let response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/global_weight?${params.toString()}`);
      
      // If it returns 503 error (model not loaded), automatically try to preload
      if (response.status === 503) {
        const errorText = await response.text();
        console.log('Detected model not loaded, starting automatic preload...', errorText);
        
        try {
          setLoadingMessage('Model not loaded, starting automatic load...');
          await preloadModels(currentSaeComboId);
          console.log('Preload completed, retrying to fetch global weights...');
          setLoadingMessage('Calculating global weights...');
          
          // Retry request
          response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/global_weight?${params.toString()}`);
        } catch (preloadErr: any) {
          throw new Error(`Automatic preload failed: ${preloadErr.message}`);
        }
      }
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }
      
      const data = await response.json() as GlobalWeightResult;
      setResult(data);
      
      // Batch fetch clerp and calculate ranks for all features
      await fetchBatchClerpsAndRanks(data.features_in, data.features_out);
      
      // Update URL parameters (use the actual sae_combo_id used)
      const newParams = new URLSearchParams({
        feature_type: featureType,
        layer_idx: layerIdx.toString(),
        feature_idx: featureIdx.toString(),
      });

      // Add layer filter to URL parameters
      if (featuresInLayerFilter.trim()) {
        newParams.append('features_in_layer_filter', featuresInLayerFilter.trim());
      }
      if (featuresOutLayerFilter.trim()) {
        newParams.append('features_out_layer_filter', featuresOutLayerFilter.trim());
      }

      newParams.append('sae_combo_id', currentSaeComboId);
      // Synchronize update state
      setSaeComboId(currentSaeComboId);
      setSearchParams(newParams);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch global weights');
      console.error('Error fetching global weight:', err);
    } finally {
      setLoading(false);
      setLoadingMessage(null);
    }
  };

  useEffect(() => {
    // If there are parameters in the URL, automatically load
    if (searchParams.get('layer_idx') && searchParams.get('feature_idx')) {
      fetchGlobalWeight();
    }
  }, []); // Only execute once when the component is mounted

  // Parse feature name, extract layer, feature index and type
  const parseFeatureName = useCallback((featureName: string): {
    layerIdx: number;
    featureIdx: number;
    featureType: 'tc' | 'lorsa';
  } | null => {
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

  // Get dictionary name (based on layer and type)
  const getDictionaryName = useCallback((layerIdx: number, isLorsa: boolean): string => {
    // Build dictionary name based on combination ID
    // Format: BT4_lorsa_L{layer}A_k30_e16 or BT4_tc_L{layer}M_k30_e16
    if (isLorsa) {
      return `BT4_lorsa_L${layerIdx}A_k30_e16`;
    } else {
      return `BT4_tc_L${layerIdx}M_k30_e16`;
    }
  }, []);

  // Get SAE name (for Circuit Interpretation)
  const getSaeNameForCircuit = useCallback((layer: number, isLorsa: boolean): string => {
    return getDictionaryName(layer, isLorsa);
  }, [getDictionaryName]);

  // Get Top Activation data (for selected feature)
  const fetchTopActivations = useCallback(async (layerIdx: number, featureIdx: number, isLorsa: boolean) => {
    setLoadingTopActivations(true);
    try {
      const dictionary = getDictionaryName(layerIdx, isLorsa);
      
      console.log('Get Top Activation data:', {
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
      
      // Find samples containing FEN and extract activations
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
      
      console.log('Get Top Activation data:', {
        totalChessSamples: chessSamples.length,
        topSamplesCount: topSamples.length
      });
      
      setTopActivations(topSamples);
      
    } catch (error) {
      console.error('Failed to get Top Activation data:', error);
      setTopActivations([]);
    } finally {
      setLoadingTopActivations(false);
    }
  }, [getDictionaryName]);

  // Get Top Activation data (for current feature or specified feature)
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
      
      console.log('Get Top Activation data:', {
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
      
      // Find samples containing FEN and extract activations
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
      
      console.log('Get Top Activation data:', {
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
      console.error('Failed to get Top Activation data:', error);
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

  // Batch fetch interpretations and calculate ranks
  const fetchBatchClerpsAndRanks = useCallback(async (
    featuresIn: GlobalWeightFeature[],
    featuresOut: GlobalWeightFeature[]
  ) => {
    setLoadingClerps(true);
    try {
      // Parse all features, build nodes array
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

      // Group by feature_type
      const lorsaNodes = nodes.filter(n => n.feature_type === 'lorsa');
      const tcNodes = nodes.filter(n => n.feature_type === 'tc');

      const clerpMap = new Map<string, string>();
      const rankMap = new Map<string, number>();

      // Batch fetch Lorsa interpretations
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
          console.warn('Failed to fetch Lorsa interpretations:', err);
        }
      }

      // Batch fetch TC interpretations
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
          console.warn('Failed to fetch TC interpretations:', err);
        }
      }

      // Calculate ranks (Lorsa and Transcoder separately)
      // For features_in and features_out separately calculate ranks
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

      // Set ranks
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

      // Merge interpretation and rank information
      const combinedMap = new Map<string, { clerp: string; rank: number }>();
      for (const feature of allFeatures) {
        const clerp = clerpMap.get(feature.name) || '';
        const rank = rankMap.get(feature.name) || 0;
        combinedMap.set(feature.name, { clerp, rank });
      }

      setFeaturesWithClerp(combinedMap);
    } catch (error) {
      console.error('Failed to fetch interpretations and ranks:', error);
    } finally {
      setLoadingClerps(false);
    }
  }, [parseFeatureName]);

  // Get Interpretation from MongoDB (for selected feature)
  const fetchClerp = useCallback(async (layerIdx: number, featureIdx: number, isLorsa: boolean) => {
    setSyncingClerp(true);
    try {
      // Build analysis_name
      const analysisName = isLorsa ? 'BT4_lorsa_k30_e16' : 'BT4_tc_k30_e16';
      
      // Build node data
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
      
      const updatedNode = result.updated_nodes?.find((n: any) => n.node_id === node.node_id);
      if (updatedNode && updatedNode.clerp) {
        setEditingClerp(updatedNode.clerp);
      } else {
        setEditingClerp('');
      }
      
    } catch (error) {
      console.error('Failed to get interpretation:', error);
      setEditingClerp('');
    } finally {
      setSyncingClerp(false);
    }
  }, [saeComboId]);

  // Get Interpretation from MongoDB (for current feature or selected feature)
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
      // Build analysis_name
      const analysisName = isLorsa ? 'BT4_lorsa_k30_e16' : 'BT4_tc_k30_e16';
      
      // Build node data
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
      
      // Find corresponding interpretation
      const updatedNode = result.updated_nodes?.find((n: any) => n.node_id === node.node_id);
      const interpretation = updatedNode?.clerp || '';
      
      if (isCurrentFeature) {
        setCurrentFeatureInterpretation(interpretation);
      } else {
        setEditingClerp(interpretation);
      }
      
    } catch (error) {
      console.error('Failed to get interpretation:', error);
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

  // Save Interpretation to MongoDB (for selected feature)
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
      
      // Update featuresWithClerp state
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
      
      alert(`Interpretation successfully saved to MongoDB!`);
      
    } catch (error) {
      console.error('Failed to save interpretation:', error);
      alert(`Failed to save: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsSavingClerp(false);
    }
  }, [saeComboId, selectedFeatureName]);

  // Save Interpretation to MongoDB (for current feature or selected feature)
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
      
      alert(`Successfully saved Interpretation to MongoDB!`);
      
    } catch (error) {
      console.error('Failed to save interpretation:', error);
      alert(`Failed to save: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      if (isCurrentFeature) {
        setIsSavingCurrentFeatureInterpretation(false);
      } else {
        setIsSavingClerp(false);
      }
    }
  }, [saeComboId, result, parseFeatureName, selectedFeatureName]);

  // When result is updated, automatically load Top Activation and Interpretation for the current feature
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
      
      // Get Top Activation and clerp
      fetchTopActivations(parsed.layerIdx, parsed.featureIdx, parsed.featureType === 'lorsa');
      fetchClerp(parsed.layerIdx, parsed.featureIdx, parsed.featureType === 'lorsa');
    }
  };

  // Search for features in Features In and Features Out and get their ranks and weights
  const searchFeature = useCallback(() => {
    if (!result) {
      alert('Please calculate global weights first');
      return;
    }

    // Build feature name
    const featureTypePrefix = searchFeatureType === 'lorsa' ? 'BT4_lorsa' : 'BT4_tc';
    const layerSuffix = searchFeatureType === 'lorsa' ? 'A' : 'M';
    const featureName = `${featureTypePrefix}_L${searchLayerIdx}${layerSuffix}_k30_e16#${searchFeatureIdx}`;

    // Search in Features In
    const featuresInMatch = result.features_in.find(f => f.name === featureName);
    const featuresInInfo = featuresInMatch ? {
      rank: featuresWithClerp.get(featureName)?.rank || 0,
      weight: featuresInMatch.weight,
      name: featureName
    } : null;

    // Search in Features Out
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

    // If found, automatically select the feature
    if (featuresInMatch || featuresOutMatch) {
      const parsed = parseFeatureName(featureName);
      if (parsed) {
        setSelectedFeatureName(featureName);
        setSelectedFeatureInfo(parsed);
        
        // Get Top Activation and clerp
        fetchTopActivations(parsed.layerIdx, parsed.featureIdx, parsed.featureType === 'lorsa');
        fetchClerp(parsed.layerIdx, parsed.featureIdx, parsed.featureType === 'lorsa');
      }
    }
  }, [result, searchFeatureType, searchLayerIdx, searchFeatureIdx, featuresWithClerp, parseFeatureName, fetchTopActivations, fetchClerp]);

  return (
    <div className="min-h-screen bg-background">
      <AppNavbar />
      <div className="container mx-auto p-6 space-y-6">
        <SaeComboLoader />
        
        <div className="flex items-center gap-4">
          <Button
            variant="outline"
            onClick={() => navigate(-1)}
            className="flex items-center gap-2"
          >
            <ArrowLeft className="w-4 h-4" />
            Back
          </Button>
          <h1 className="text-3xl font-bold">Global Weight Analysis</h1>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Parameter Settings</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
              <div className="space-y-2">
                <Label htmlFor="feature_type">Feature Type</Label>
                <Select
                  value={featureType}
                  onValueChange={(value) => setFeatureType(value as 'tc' | 'lorsa')}
                >
                  <SelectTrigger id="feature_type">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="tc">TC (Transcoder)</SelectItem>
                    <SelectItem value="lorsa">Lorsa</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="layer_idx">Layer Index</Label>
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
                <Label htmlFor="feature_idx">Feature Index</Label>
                <Input
                  id="feature_idx"
                  type="number"
                  min="0"
                  value={featureIdx}
                  onChange={(e) => setFeatureIdx(parseInt(e.target.value) || 0)}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="activation_type">Activation Type</Label>
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

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="features_in_layer_filter">
                  Features In Layer Filter
                  <span className="text-xs text-muted-foreground ml-2">
                    (e.g. 4,5,8-9 means layers 4, 5, 8, 9)
                  </span>
                </Label>
                <Input
                  id="features_in_layer_filter"
                  type="text"
                  placeholder="Leave empty to filter"
                  value={featuresInLayerFilter}
                  onChange={(e) => setFeaturesInLayerFilter(e.target.value)}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="features_out_layer_filter">
                  Features Out Layer Filter
                  <span className="text-xs text-muted-foreground ml-2">
                    (e.g. 4,5,8-9 means layers 4, 5, 8, 9)
                  </span>
                </Label>
                <Input
                  id="features_out_layer_filter"
                  type="text"
                  placeholder="Leave empty to filter"
                  value={featuresOutLayerFilter}
                  onChange={(e) => setFeaturesOutLayerFilter(e.target.value)}
                />
              </div>
            </div>

            <div className="mt-4 space-y-2">
              <Button onClick={fetchGlobalWeight} disabled={loading}>
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    {loadingMessage || 'Calculating...'}
                  </>
                ) : (
                  'Calculate Global Weight'
                )}
              </Button>
              {loadingMessage && loadingMessage !== 'Calculating...' && (
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
            {/* Feature Query Card */}
            <Card className="border-blue-200 bg-blue-50/50">
              <CardHeader>
                <CardTitle>Query Feature Rank and Weight</CardTitle>
                <p className="text-sm text-muted-foreground">
                  Search for the rank and weight of a specified feature in Features In and Features Out
                </p>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="search_feature_type">Feature Type</Label>
                    <Select
                      value={searchFeatureType}
                      onValueChange={(value) => setSearchFeatureType(value as 'tc' | 'lorsa')}
                    >
                      <SelectTrigger id="search_feature_type">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="tc">TC (Transcoder)</SelectItem>
                        <SelectItem value="lorsa">Lorsa</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="search_layer_idx">Layer Index</Label>
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
                    <Label htmlFor="search_feature_idx">Feature Index</Label>
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
                      Search
                    </Button>
                  </div>
                </div>

                {/* Search Result */}
                {searchResult && (
                  <div className="mt-4 p-4 bg-white rounded-lg border border-blue-200">
                    <h4 className="font-semibold mb-3 text-blue-900">Search Result</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className={`p-3 rounded ${searchResult.foundInFeaturesIn ? 'bg-green-50 border border-green-300' : 'bg-gray-50 border border-gray-300'}`}>
                        <div className="font-medium text-sm mb-2">
                          Features In (Input Feature)
                        </div>
                        {searchResult.foundInFeaturesIn && searchResult.featuresInInfo ? (
                          <div className="space-y-1 text-sm">
                            <div>
                              <span className="font-medium">Feature Name:</span>{' '}
                              <span className="font-mono text-xs">{searchResult.featuresInInfo.name}</span>
                            </div>
                            <div>
                              <span className="font-medium">Rank:</span>{' '}
                              <span className="text-green-700 font-semibold">
                                #{searchResult.featuresInInfo.rank > 0 ? searchResult.featuresInInfo.rank : 'Not calculated'}
                              </span>
                            </div>
                            <div>
                              <span className="font-medium">Weight:</span>{' '}
                              <span className="text-green-700 font-semibold">
                                {searchResult.featuresInInfo.weight.toFixed(6)}
                              </span>
                            </div>
                          </div>
                        ) : (
                          <div className="text-sm text-gray-500">Feature not found</div>
                        )}
                      </div>

                      <div className={`p-3 rounded ${searchResult.foundInFeaturesOut ? 'bg-green-50 border border-green-300' : 'bg-gray-50 border border-gray-300'}`}>
                        <div className="font-medium text-sm mb-2">
                          Features Out (Output Feature)
                        </div>
                        {searchResult.foundInFeaturesOut && searchResult.featuresOutInfo ? (
                          <div className="space-y-1 text-sm">
                            <div>
                              <span className="font-medium">Feature Name:</span>{' '}
                              <span className="font-mono text-xs">{searchResult.featuresOutInfo.name}</span>
                            </div>
                            <div>
                              <span className="font-medium">Rank:</span>{' '}
                              <span className="text-green-700 font-semibold">
                                #{searchResult.featuresOutInfo.rank > 0 ? searchResult.featuresOutInfo.rank : 'Not calculated'}
                              </span>
                            </div>
                            <div>
                              <span className="font-medium">Weight:</span>{' '}
                              <span className="text-green-700 font-semibold">
                                {searchResult.featuresOutInfo.weight.toFixed(6)}
                              </span>
                            </div>
                          </div>
                        ) : (
                          <div className="text-sm text-gray-500">Feature not found</div>
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
                  <CardTitle>Current Feature: {result.feature_name}</CardTitle>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setCurrentFeatureExpanded(!currentFeatureExpanded)}
                    className="flex items-center gap-2"
                  >
                    {currentFeatureExpanded ? (
                      <>
                        <ChevronUp className="w-4 h-4" />
                        Collapse
                      </>
                    ) : (
                      <>
                        <ChevronDown className="w-4 h-4" />
                        Expand
                      </>
                    )}
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <p><strong>Type:</strong> {result.feature_type === 'tc' ? 'TC (Transcoder)' : 'Lorsa'}</p>
                  <p><strong>Layer:</strong> {result.layer_idx}</p>
                  <p><strong>Feature Index:</strong> {result.feature_idx}</p>
                  {result.activation_type && (
                    <p><strong>Activation Type:</strong> {result.activation_type === 'max' ? 'Max Activation' : 'Mean Activation'}</p>
                  )}
                </div>
                
                {currentFeatureExpanded && (
                  <div className="mt-6 space-y-6 pt-6 border-t">
                    {/* Top Activation  */}
                    <div>
                      <h3 className="text-lg font-semibold mb-4">Top Activation Chessboard</h3>
                      {loadingCurrentFeatureTopActivations ? (
                        <div className="flex items-center justify-center py-8">
                          <div className="text-center">
                            <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4" />
                            <p className="text-gray-600">Getting Top Activation data...</p>
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
                                  Max Activation: {sample.activationStrength.toFixed(3)}
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
                          <p>No activation samples found with chessboard</p>
                        </div>
                      )}
                    </div>

                    {/* Circuit Interpretation  */}
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

                    {/* Interpretation Editor */}
                    <div>
                      <h3 className="text-lg font-semibold mb-4">Interpretation Editor</h3>
                      <div className="space-y-4">
                        <div className="flex justify-between items-center">
                          <Label htmlFor="current-feature-interpretation">Interpretation Content</Label>
                          <div className="text-xs text-gray-500">
                            Character Count: {currentFeatureInterpretation.length}
                          </div>
                        </div>
                        <textarea
                          id="current-feature-interpretation"
                          value={currentFeatureInterpretation}
                          onChange={(e) => setCurrentFeatureInterpretation(e.target.value)}
                          className="w-full h-32 px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 font-mono text-sm"
                          placeholder="Enter or edit the interpretation content of the feature..."
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
                                Syncing...
                              </>
                            ) : (
                              'Sync from MongoDB'
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
                                Saving...
                              </>
                            ) : (
                              'Save to MongoDB'
                            )}
                          </Button>
                        </div>
                        <div className="text-xs text-blue-600 bg-blue-50 p-3 rounded border-l-4 border-blue-200">
                          <div className="font-medium mb-1">💡 Usage Instructions:</div>
                          <ul className="list-disc list-inside space-y-1 text-blue-700">
                            <li>After editing the interpretation content, click "Save to MongoDB" to synchronize the changes to the database</li>
                            <li>Click "Sync from MongoDB" to read the latest interpretation content from the database</li>
                            <li>The interpretation will be saved to the interpretation field of the corresponding feature</li>
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
                  <CardTitle>Input Features (Features In)</CardTitle>
                  <p className="text-sm text-muted-foreground">
                    Features that affect the current feature (Top {result.features_in.length})
                  </p>
                </CardHeader>
                <CardContent>
                  {loadingClerps && (
                    <div className="text-sm text-muted-foreground mb-2 flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Loading Interpretation and Rank...
                    </div>
                  )}
                  <div className="space-y-2 max-h-[600px] overflow-y-auto">
                    {result.features_in.map((feature, idx) => {
                      const featureInfo = featuresWithClerp.get(feature.name);
                      const clerp = featureInfo?.clerp || '';
                      const rank = featureInfo?.rank || 0;
                      const parsed = parseFeatureName(feature.name);
                      const featureTypeLabel = parsed?.featureType === 'lorsa' ? 'Lorsa' : 'TC';
                      
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
                          title="Click to view the Top Activation of this feature"
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
                  <CardTitle>Output Features (Features Out)</CardTitle>
                  <p className="text-sm text-muted-foreground">
                    Features that are affected by the current feature (Top {result.features_out.length})
                  </p>
                </CardHeader>
                <CardContent>
                  {loadingClerps && (
                    <div className="text-sm text-muted-foreground mb-2 flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Loading Interpretation and Rank...
                    </div>
                  )}
                  <div className="space-y-2 max-h-[600px] overflow-y-auto">
                    {result.features_out.map((feature, idx) => {
                      const featureInfo = featuresWithClerp.get(feature.name);
                      const clerp = featureInfo?.clerp || '';
                      const rank = featureInfo?.rank || 0;
                      const parsed = parseFeatureName(feature.name);
                      const featureTypeLabel = parsed?.featureType === 'lorsa' ? 'Lorsa' : 'TC';
                      
                      // Check if it is a search result
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
                          title="Click to view the Top Activation of this feature"
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

        {/* Feature Details Panel: Top Activation and Clerp Editor */}
        {selectedFeatureName && selectedFeatureInfo && (
          <Card className="mt-6">
            <CardHeader>
              <CardTitle>Feature Details: {selectedFeatureName}</CardTitle>
              <p className="text-sm text-muted-foreground">
                Layer: {selectedFeatureInfo.layerIdx}, Feature Index: {selectedFeatureInfo.featureIdx}, 
                Type: {selectedFeatureInfo.featureType === 'tc' ? 'TC (Transcoder)' : 'Lorsa'}
              </p>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Top Activation */}
              <div>
                <h3 className="text-lg font-semibold mb-4">Top Activation Chessboard</h3>
                {loadingTopActivations ? (
                  <div className="flex items-center justify-center py-8">
                    <div className="text-center">
                      <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4" />
                      <p className="text-gray-600">Getting Top Activation data...</p>
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
                            Max Activation: {sample.activationStrength.toFixed(3)}
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
                    <p>No activation samples found with chessboard</p>
                  </div>
                )}
              </div>

              {/* Circuit Interpretation  */}
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

              {/* Interpretation Editor  */}
              <div>
                <h3 className="text-lg font-semibold mb-4">Interpretation Editor</h3>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <Label htmlFor="interpretation-editor">Interpretation Content</Label>
                    <div className="text-xs text-gray-500">
                      Character Count: {editingClerp.length}
                    </div>
                  </div>
                  <textarea
                    id="interpretation-editor"
                    value={editingClerp}
                    onChange={(e) => setEditingClerp(e.target.value)}
                    className="w-full h-32 px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 font-mono text-sm"
                    placeholder="Enter or edit the interpretation content of the feature..."
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
                          Syncing...  
                        </>
                      ) : (
                        'Sync from MongoDB'
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
                          Saving...
                        </>
                      ) : (
                        'Save to MongoDB'
                      )}
                    </Button>
                  </div>
                  <div className="text-xs text-blue-600 bg-blue-50 p-3 rounded border-l-4 border-blue-200">
                    <div className="font-medium mb-1">💡 Usage Instructions:</div>
                    <ul className="list-disc list-inside space-y-1 text-blue-700">
                      <li>After editing the interpretation content, click "Save to MongoDB" to synchronize the changes to the database</li>
                      <li>Click "Sync from MongoDB" to read the latest interpretation content from the database</li>
                      <li>The interpretation will be saved to the interpretation field of the corresponding feature</li>
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
