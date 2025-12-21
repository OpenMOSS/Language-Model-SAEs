import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  CircuitAnnotation,
  CircuitFeature,
  getCircuitsByFeature,
  listCircuitAnnotations,
  createCircuitAnnotation,
  updateCircuitInterpretation,
  addFeatureToCircuit,
  removeFeatureFromCircuit,
  updateFeatureInterpretationInCircuit,
  deleteCircuitAnnotation,
} from '@/utils/api';

interface CircuitInterpretationCardProps {
  /** 当前选中的节点信息 */
  node: {
    nodeId: string;
    layer: number;
    feature: number;
    feature_type: string;
  } | null;
  /** SAE组合ID */
  saeComboId: string;
  /** SAE系列 */
  saeSeries: string;
  /** 从metadata中获取SAE名称的函数 */
  getSaeName: (layer: number, isLorsa: boolean) => string;
}

export const CircuitInterpretationCard: React.FC<CircuitInterpretationCardProps> = ({
  node,
  saeComboId,
  saeSeries,
  getSaeName,
}) => {
  const [allCircuits, setAllCircuits] = useState<CircuitAnnotation[]>([]);
  const [featureCircuits, setFeatureCircuits] = useState<CircuitAnnotation[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [creatingNew, setCreatingNew] = useState(false);
  const [newCircuitInterpretation, setNewCircuitInterpretation] = useState('');
  const [editingCircuitId, setEditingCircuitId] = useState<string | null>(null);
  const [editingInterpretation, setEditingInterpretation] = useState('');
  const [editingFeatureCircuitId, setEditingFeatureCircuitId] = useState<string | null>(null);
  const [editingFeatureIndex, setEditingFeatureIndex] = useState<number | null>(null);
  const [editingFeatureInterpretation, setEditingFeatureInterpretation] = useState('');
  const [showAllCircuits, setShowAllCircuits] = useState(false);
  const [collapsedCircuits, setCollapsedCircuits] = useState<Set<string>>(new Set());

  // 加载所有circuits
  const [loadingAllCircuits, setLoadingAllCircuits] = useState(false);
  
  const loadAllCircuits = useCallback(async () => {
    setLoadingAllCircuits(true);
    setError(null);

    try {
      const result = await listCircuitAnnotations(saeComboId, 100, 0);
      setAllCircuits(result.circuits);
    } catch (err) {
      setError(err instanceof Error ? err.message : '加载所有circuit标注失败');
      console.error('Failed to load all circuits:', err);
      setAllCircuits([]); // 确保即使出错也设置空数组，避免一直显示加载中
    } finally {
      setLoadingAllCircuits(false);
    }
  }, [saeComboId]);

  // 加载当前feature所属的circuits
  const loadFeatureCircuits = useCallback(async () => {
    if (!node) {
      setFeatureCircuits([]);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const isLorsa = node.feature_type?.toLowerCase() === 'lorsa';
      const saeName = getSaeName(node.layer, isLorsa);
      const featureType = isLorsa ? 'lorsa' : 'transcoder';

      const result = await getCircuitsByFeature(
        saeName,
        saeSeries,
        node.layer,
        node.feature,
        featureType as 'transcoder' | 'lorsa'
      );

      setFeatureCircuits(result.circuits);
    } catch (err) {
      setError(err instanceof Error ? err.message : '加载circuit标注失败');
      console.error('Failed to load feature circuits:', err);
    } finally {
      setLoading(false);
    }
  }, [node, saeSeries, getSaeName]);

  // 当节点变化时重新加载
  useEffect(() => {
    if (node) {
      loadFeatureCircuits();
    } else {
      setFeatureCircuits([]);
    }
    // 移除 loadFeatureCircuits 从依赖项，避免频繁重新创建导致的循环调用
    // 只依赖 node 的关键属性，而不是整个 node 对象
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [node?.nodeId, node?.layer, node?.feature, node?.feature_type]);

  // 当切换到"显示全部"时，自动加载所有circuits
  // 使用 ref 来跟踪是否已经加载过，避免重复加载
  const hasLoadedAllCircuitsRef = useRef(false);
  
  useEffect(() => {
    // 只在切换到"显示全部"模式且还没有加载过时，才加载一次
    if (showAllCircuits && !hasLoadedAllCircuitsRef.current && !loadingAllCircuits) {
      hasLoadedAllCircuitsRef.current = true;
      loadAllCircuits();
    }
    // 当切换回"仅当前feature"模式时，不重置标志，保留已加载的数据
    // 移除 loadAllCircuits 从依赖项，避免频繁重新创建导致的循环调用
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showAllCircuits]);

  // 刷新所有数据
  const refreshAll = useCallback(async () => {
    await Promise.all([loadAllCircuits(), loadFeatureCircuits()]);
  }, [loadAllCircuits, loadFeatureCircuits]);

  // 创建新circuit
  const handleCreateCircuit = useCallback(async () => {
    if (!node || !newCircuitInterpretation.trim()) {
      setError('请输入circuit解释');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const isLorsa = node.feature_type?.toLowerCase() === 'lorsa';
      const saeName = getSaeName(node.layer, isLorsa);
      const featureType = isLorsa ? 'lorsa' : 'transcoder';

      const feature: CircuitFeature = {
        sae_name: saeName,
        sae_series: saeSeries,
        layer: node.layer,
        feature_index: node.feature,
        feature_type: featureType as 'transcoder' | 'lorsa',
        interpretation: '',
      };

      await createCircuitAnnotation(
        newCircuitInterpretation.trim(),
        saeComboId,
        [feature]
      );

      setNewCircuitInterpretation('');
      setCreatingNew(false);
      await refreshAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : '创建circuit失败');
      console.error('Failed to create circuit:', err);
    } finally {
      setLoading(false);
    }
  }, [node, newCircuitInterpretation, saeComboId, saeSeries, getSaeName, refreshAll]);

  // 更新circuit解释
  const handleUpdateCircuitInterpretation = useCallback(async (circuitId: string) => {
    if (!editingInterpretation.trim()) {
      setError('请输入circuit解释');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      await updateCircuitInterpretation(circuitId, editingInterpretation.trim());
      setEditingCircuitId(null);
      setEditingInterpretation('');
      await refreshAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : '更新circuit解释失败');
      console.error('Failed to update circuit interpretation:', err);
    } finally {
      setLoading(false);
    }
  }, [editingInterpretation, refreshAll]);

  // 添加feature到circuit
  const handleAddFeatureToCircuit = useCallback(async (circuitId: string) => {
    if (!node) return;

    setLoading(true);
    setError(null);

    try {
      const isLorsa = node.feature_type?.toLowerCase() === 'lorsa';
      const saeName = getSaeName(node.layer, isLorsa);
      const featureType = isLorsa ? 'lorsa' : 'transcoder';

      const feature: CircuitFeature = {
        sae_name: saeName,
        sae_series: saeSeries,
        layer: node.layer,
        feature_index: node.feature,
        feature_type: featureType as 'transcoder' | 'lorsa',
        interpretation: '',
      };

      await addFeatureToCircuit(circuitId, feature);
      await refreshAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : '添加feature到circuit失败');
      console.error('Failed to add feature to circuit:', err);
    } finally {
      setLoading(false);
    }
  }, [node, saeSeries, getSaeName, refreshAll]);

  // 从circuit删除feature
  const handleRemoveFeatureFromCircuit = useCallback(async (
    circuitId: string,
    saeName: string,
    layer: number,
    featureIndex: number,
    featureType: 'transcoder' | 'lorsa'
  ) => {
    setLoading(true);
    setError(null);

    try {
      await removeFeatureFromCircuit(
        circuitId,
        saeName,
        saeSeries,
        layer,
        featureIndex,
        featureType
      );
      await refreshAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : '从circuit删除feature失败');
      console.error('Failed to remove feature from circuit:', err);
    } finally {
      setLoading(false);
    }
  }, [saeSeries, refreshAll]);

  // 更新feature解释
  const handleUpdateFeatureInterpretation = useCallback(async (
    circuitId: string,
    saeName: string,
    layer: number,
    featureIndex: number,
    featureType: 'transcoder' | 'lorsa'
  ) => {
    setLoading(true);
    setError(null);

    try {
      await updateFeatureInterpretationInCircuit(
        circuitId,
        saeName,
        saeSeries,
        layer,
        featureIndex,
        featureType,
        editingFeatureInterpretation.trim()
      );
      setEditingFeatureCircuitId(null);
      setEditingFeatureIndex(null);
      setEditingFeatureInterpretation('');
      await refreshAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : '更新feature解释失败');
      console.error('Failed to update feature interpretation:', err);
    } finally {
      setLoading(false);
    }
  }, [saeSeries, editingFeatureInterpretation, refreshAll]);

  // 删除circuit
  const handleDeleteCircuit = useCallback(async (circuitId: string) => {
    if (!confirm('确定要删除这个circuit标注吗？')) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      await deleteCircuitAnnotation(circuitId);
      await refreshAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : '删除circuit失败');
      console.error('Failed to delete circuit:', err);
    } finally {
      setLoading(false);
    }
  }, [refreshAll]);

  if (!node) {
    return (
      <div className="w-full border rounded-lg p-4 bg-white shadow-sm">
        <div className="text-center py-8 text-gray-500">
          请先选择一个节点以查看Circuit标注
        </div>
      </div>
    );
  }

  const isLorsa = node.feature_type?.toLowerCase() === 'lorsa';
  const currentSaeName = getSaeName(node.layer, isLorsa);
  const featureType = isLorsa ? 'lorsa' : 'transcoder';

  // 合并显示：当前feature所属的circuits + 所有其他circuits
  const displayCircuits = showAllCircuits ? allCircuits : featureCircuits;
  const otherCircuits = showAllCircuits 
    ? allCircuits.filter(c => !featureCircuits.some(fc => fc.circuit_id === c.circuit_id))
    : [];

  return (
    <div className="w-full border rounded-lg p-4 bg-white shadow-sm">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold">
          Circuit Interpretation
          {node && (
            <span className="text-sm font-normal text-gray-600 ml-2">
              - {featureType === 'lorsa' ? 'LoRSA' : 'Transcoder'} L{node.layer} #{node.feature}
            </span>
          )}
        </h3>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => {
              setShowAllCircuits(!showAllCircuits);
              // useEffect 会自动处理加载逻辑，这里不需要手动调用
            }}
            className={`px-3 py-1 text-sm rounded transition-colors ${
              showAllCircuits
                ? 'bg-blue-500 text-white hover:bg-blue-600'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
            title={showAllCircuits ? '只显示当前feature所属的circuits' : '显示所有circuits'}
          >
            {showAllCircuits ? '仅当前feature' : '显示全部'}
          </button>
          <button
            onClick={refreshAll}
            disabled={loading}
            className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 disabled:opacity-50 transition-colors"
            title="刷新数据"
          >
            {loading ? '刷新中...' : '刷新'}
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">
          {error}
        </div>
      )}

      {/* 创建新circuit */}
      <div className="mb-6">
        {!creatingNew ? (
          <button
            onClick={() => setCreatingNew(true)}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 text-sm"
            disabled={loading}
          >
            + 创建新 Circuit
          </button>
        ) : (
          <div className="border border-gray-300 rounded p-4">
            <h3 className="font-semibold mb-2 text-sm">创建新 Circuit</h3>
            <textarea
              value={newCircuitInterpretation}
              onChange={(e) => setNewCircuitInterpretation(e.target.value)}
              placeholder="输入circuit的整体解释..."
              className="w-full p-2 border border-gray-300 rounded mb-2 text-sm"
              rows={3}
            />
            <div className="flex gap-2">
              <button
                onClick={handleCreateCircuit}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 text-sm"
                disabled={loading || !newCircuitInterpretation.trim()}
              >
                创建
              </button>
              <button
                onClick={() => {
                  setCreatingNew(false);
                  setNewCircuitInterpretation('');
                }}
                className="px-4 py-2 bg-gray-300 text-gray-700 rounded hover:bg-gray-400 text-sm"
              >
                取消
              </button>
            </div>
          </div>
        )}
      </div>

      {/* 当前feature所属的circuits */}
      {featureCircuits.length > 0 && (
        <div className="mb-6">
          <h4 className="font-semibold mb-3 text-sm text-blue-700">
            当前feature所属的Circuits ({featureCircuits.length})
          </h4>
          <div className="space-y-4">
            {featureCircuits.map((circuit) => (
              <CircuitItem
                key={circuit.circuit_id}
                circuit={circuit}
                node={node}
                currentSaeName={currentSaeName}
                featureType={featureType}
                saeSeries={saeSeries}
                editingCircuitId={editingCircuitId}
                editingInterpretation={editingInterpretation}
                editingFeatureCircuitId={editingFeatureCircuitId}
                editingFeatureIndex={editingFeatureIndex}
                editingFeatureInterpretation={editingFeatureInterpretation}
                loading={loading}
                isCollapsed={collapsedCircuits.has(circuit.circuit_id)}
                onToggleCollapse={(circuitId) => {
                  setCollapsedCircuits(prev => {
                    const newSet = new Set(prev);
                    if (newSet.has(circuitId)) {
                      newSet.delete(circuitId);
                    } else {
                      newSet.add(circuitId);
                    }
                    return newSet;
                  });
                }}
                onSetEditingCircuitId={setEditingCircuitId}
                onSetEditingInterpretation={setEditingInterpretation}
                onSetEditingFeatureCircuitId={setEditingFeatureCircuitId}
                onSetEditingFeatureIndex={setEditingFeatureIndex}
                onSetEditingFeatureInterpretation={setEditingFeatureInterpretation}
                onUpdateCircuitInterpretation={handleUpdateCircuitInterpretation}
                onRemoveFeature={handleRemoveFeatureFromCircuit}
                onUpdateFeatureInterpretation={handleUpdateFeatureInterpretation}
                onDeleteCircuit={handleDeleteCircuit}
              />
            ))}
          </div>
        </div>
      )}

      {/* 所有其他circuits（当显示全部时） */}
      {showAllCircuits && otherCircuits.length > 0 && (
        <div className="mb-6">
          <h4 className="font-semibold mb-3 text-sm text-gray-700">
            其他Circuits ({otherCircuits.length})
          </h4>
          <div className="space-y-4">
            {otherCircuits.map((circuit) => (
              <CircuitItem
                key={circuit.circuit_id}
                circuit={circuit}
                node={node}
                currentSaeName={currentSaeName}
                featureType={featureType}
                saeSeries={saeSeries}
                editingCircuitId={editingCircuitId}
                editingInterpretation={editingInterpretation}
                editingFeatureCircuitId={editingFeatureCircuitId}
                editingFeatureIndex={editingFeatureIndex}
                editingFeatureInterpretation={editingFeatureInterpretation}
                loading={loading}
                isCollapsed={collapsedCircuits.has(circuit.circuit_id)}
                onToggleCollapse={(circuitId) => {
                  setCollapsedCircuits(prev => {
                    const newSet = new Set(prev);
                    if (newSet.has(circuitId)) {
                      newSet.delete(circuitId);
                    } else {
                      newSet.add(circuitId);
                    }
                    return newSet;
                  });
                }}
                onSetEditingCircuitId={setEditingCircuitId}
                onSetEditingInterpretation={setEditingInterpretation}
                onSetEditingFeatureCircuitId={setEditingFeatureCircuitId}
                onSetEditingFeatureIndex={setEditingFeatureIndex}
                onSetEditingFeatureInterpretation={setEditingFeatureInterpretation}
                onUpdateCircuitInterpretation={handleUpdateCircuitInterpretation}
                onAddFeature={handleAddFeatureToCircuit}
                onRemoveFeature={handleRemoveFeatureFromCircuit}
                onUpdateFeatureInterpretation={handleUpdateFeatureInterpretation}
                onDeleteCircuit={handleDeleteCircuit}
              />
            ))}
          </div>
        </div>
      )}

      {/* 加载状态 */}
      {loadingAllCircuits && showAllCircuits && allCircuits.length === 0 && (
        <div className="text-center py-8 text-gray-500">加载中...</div>
      )}

      {/* 空状态 - 只在非加载状态且没有数据时显示 */}
      {!loadingAllCircuits && !loading && displayCircuits.length === 0 && (
        <div className="text-center py-8 text-gray-500">
          {showAllCircuits 
            ? '没有找到任何circuit标注'
            : '该feature还没有属于任何circuit标注'}
        </div>
      )}
    </div>
  );
};

// Circuit项组件
interface CircuitItemProps {
  circuit: CircuitAnnotation;
  node: {
    nodeId: string;
    layer: number;
    feature: number;
    feature_type: string;
  };
  currentSaeName: string;
  featureType: 'transcoder' | 'lorsa';
  saeSeries: string;
  editingCircuitId: string | null;
  editingInterpretation: string;
  editingFeatureCircuitId: string | null;
  editingFeatureIndex: number | null;
  editingFeatureInterpretation: string;
  loading: boolean;
  isCollapsed: boolean;
  onToggleCollapse: (circuitId: string) => void;
  onSetEditingCircuitId: (id: string | null) => void;
  onSetEditingInterpretation: (text: string) => void;
  onSetEditingFeatureCircuitId: (id: string | null) => void;
  onSetEditingFeatureIndex: (idx: number | null) => void;
  onSetEditingFeatureInterpretation: (text: string) => void;
  onUpdateCircuitInterpretation: (circuitId: string) => void;
  onAddFeature?: (circuitId: string) => void;
  onRemoveFeature: (
    circuitId: string,
    saeName: string,
    layer: number,
    featureIndex: number,
    featureType: 'transcoder' | 'lorsa'
  ) => void;
  onUpdateFeatureInterpretation: (
    circuitId: string,
    saeName: string,
    layer: number,
    featureIndex: number,
    featureType: 'transcoder' | 'lorsa'
  ) => void;
  onDeleteCircuit: (circuitId: string) => void;
}

const CircuitItem: React.FC<CircuitItemProps> = ({
  circuit,
  node,
  currentSaeName,
  featureType,
  saeSeries: _saeSeries,
  editingCircuitId,
  editingInterpretation,
  editingFeatureCircuitId,
  editingFeatureIndex,
  editingFeatureInterpretation,
  loading,
  isCollapsed,
  onToggleCollapse,
  onSetEditingCircuitId,
  onSetEditingInterpretation,
  onSetEditingFeatureCircuitId,
  onSetEditingFeatureIndex,
  onSetEditingFeatureInterpretation,
  onUpdateCircuitInterpretation,
  onAddFeature,
  onRemoveFeature,
  onUpdateFeatureInterpretation,
  onDeleteCircuit,
}) => {
  const isCurrentFeatureInCircuit = circuit.features.some(
    (f) =>
      f.sae_name === currentSaeName &&
      f.layer === node.layer &&
      f.feature_index === node.feature &&
      f.feature_type === featureType
  );

  return (
    <div className="border border-gray-300 rounded p-4">
      {/* Circuit头部 */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          {editingCircuitId === circuit.circuit_id ? (
            <div>
              <textarea
                value={editingInterpretation}
                onChange={(e) => onSetEditingInterpretation(e.target.value)}
                className="w-full p-2 border border-gray-300 rounded mb-2 text-sm"
                rows={2}
              />
              <div className="flex gap-2">
                <button
                  onClick={() => onUpdateCircuitInterpretation(circuit.circuit_id)}
                  className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700"
                  disabled={loading}
                >
                  保存
                </button>
                <button
                  onClick={() => {
                    onSetEditingCircuitId(null);
                    onSetEditingInterpretation('');
                  }}
                  className="px-3 py-1 bg-gray-300 text-gray-700 rounded text-sm hover:bg-gray-400"
                >
                  取消
                </button>
              </div>
            </div>
          ) : (
            <div>
              <p className="text-gray-700 mb-1 text-sm">
                <span className="font-semibold">Circuit解释:</span>{' '}
                {circuit.circuit_interpretation || '(无)'}
              </p>
              <p className="text-xs text-gray-500">
                Circuit ID: {circuit.circuit_id}
              </p>
            </div>
          )}
        </div>
        <div className="flex gap-2 ml-4">
          {editingCircuitId !== circuit.circuit_id && (
            <>
              <button
                onClick={() => onToggleCollapse(circuit.circuit_id)}
                className="px-3 py-1 bg-gray-500 text-white rounded text-xs hover:bg-gray-600"
                disabled={loading}
                title={isCollapsed ? '展开' : '折叠'}
              >
                {isCollapsed ? (
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                ) : (
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                  </svg>
                )}
              </button>
              <button
                onClick={() => {
                  onSetEditingCircuitId(circuit.circuit_id);
                  onSetEditingInterpretation(circuit.circuit_interpretation);
                }}
                className="px-3 py-1 bg-yellow-600 text-white rounded text-xs hover:bg-yellow-700"
                disabled={loading}
              >
                编辑
              </button>
              {!isCurrentFeatureInCircuit && onAddFeature && (
                <button
                  onClick={() => onAddFeature(circuit.circuit_id)}
                  className="px-3 py-1 bg-green-600 text-white rounded text-xs hover:bg-green-700"
                  disabled={loading}
                >
                  添加当前feature
                </button>
              )}
              <button
                onClick={() => onDeleteCircuit(circuit.circuit_id)}
                className="px-3 py-1 bg-red-600 text-white rounded text-xs hover:bg-red-700"
                disabled={loading}
              >
                删除
              </button>
            </>
          )}
        </div>
      </div>

      {/* Features列表 */}
      {!isCollapsed && (
        <div className="mt-3 border-t border-gray-200 pt-3">
          <h4 className="font-semibold mb-2 text-xs">Features ({circuit.features.length}):</h4>
          <div className="space-y-2">
          {circuit.features.map((feature, idx) => {
            const isCurrentFeature =
              feature.sae_name === currentSaeName &&
              feature.layer === node.layer &&
              feature.feature_index === node.feature &&
              feature.feature_type === featureType;

            return (
              <div
                key={idx}
                className={`p-2 rounded text-xs ${
                  isCurrentFeature ? 'bg-blue-50 border border-blue-300' : 'bg-gray-50'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="font-medium">
                      {feature.feature_type === 'lorsa' ? 'LoRSA' : 'Transcoder'} L{feature.layer} #{feature.feature_index}
                    </div>
                    <div className="text-xs text-gray-600 mt-1">
                      SAE: {feature.sae_name}
                    </div>
                    {editingFeatureCircuitId === circuit.circuit_id &&
                    editingFeatureIndex === idx ? (
                      <div className="mt-2">
                        <textarea
                          value={editingFeatureInterpretation}
                          onChange={(e) => onSetEditingFeatureInterpretation(e.target.value)}
                          placeholder="输入feature解释..."
                          className="w-full p-2 border border-gray-300 rounded text-xs"
                          rows={2}
                        />
                        <div className="flex gap-2 mt-1">
                          <button
                            onClick={() =>
                              onUpdateFeatureInterpretation(
                                circuit.circuit_id,
                                feature.sae_name,
                                feature.layer,
                                feature.feature_index,
                                feature.feature_type
                              )
                            }
                            className="px-2 py-1 bg-blue-600 text-white rounded text-xs hover:bg-blue-700"
                            disabled={loading}
                          >
                            保存
                          </button>
                          <button
                            onClick={() => {
                              onSetEditingFeatureCircuitId(null);
                              onSetEditingFeatureIndex(null);
                              onSetEditingFeatureInterpretation('');
                            }}
                            className="px-2 py-1 bg-gray-300 text-gray-700 rounded text-xs hover:bg-gray-400"
                          >
                            取消
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="text-gray-700 mt-1 text-xs">
                        <span className="font-medium">解释:</span>{' '}
                        {feature.interpretation || '(无)'}
                      </div>
                    )}
                  </div>
                  <div className="flex gap-1 ml-2">
                    {editingFeatureCircuitId !== circuit.circuit_id && (
                      <>
                        <button
                          onClick={() => {
                            onSetEditingFeatureCircuitId(circuit.circuit_id);
                            onSetEditingFeatureIndex(idx);
                            onSetEditingFeatureInterpretation(feature.interpretation || '');
                          }}
                          className="px-2 py-1 bg-yellow-600 text-white rounded text-xs hover:bg-yellow-700"
                          disabled={loading}
                        >
                          编辑
                        </button>
                        <button
                          onClick={() =>
                            onRemoveFeature(
                              circuit.circuit_id,
                              feature.sae_name,
                              feature.layer,
                              feature.feature_index,
                              feature.feature_type
                            )
                          }
                          className="px-2 py-1 bg-red-600 text-white rounded text-xs hover:bg-red-700"
                          disabled={loading}
                        >
                          删除
                        </button>
                      </>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
          </div>
        </div>
      )}
      {/* 折叠时显示feature数量 */}
      {isCollapsed && (
        <div className="mt-3 border-t border-gray-200 pt-3">
          <p className="text-xs text-gray-600">
            Features: {circuit.features.length} 个
          </p>
        </div>
      )}
    </div>
  );
};

