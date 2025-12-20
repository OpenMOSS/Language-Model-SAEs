import React, { useState, useCallback, useEffect } from 'react';
import {
  CircuitAnnotation,
  CircuitFeature,
  getCircuitsByFeature,
  createCircuitAnnotation,
  updateCircuitInterpretation,
  addFeatureToCircuit,
  removeFeatureFromCircuit,
  updateFeatureInterpretationInCircuit,
  deleteCircuitAnnotation,
} from '@/utils/api';

interface CircuitInterpretationProps {
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
  /** 是否显示 */
  visible: boolean;
  /** 关闭回调 */
  onClose: () => void;
}

export const CircuitInterpretation: React.FC<CircuitInterpretationProps> = ({
  node,
  saeComboId,
  saeSeries,
  getSaeName,
  visible,
  onClose,
}) => {
  console.log('[CircuitInterpretation] Component initialized', {
    node,
    saeComboId,
    saeSeries,
    visible,
    hasGetSaeName: !!getSaeName,
  });

  const [circuits, setCircuits] = useState<CircuitAnnotation[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [creatingNew, setCreatingNew] = useState(false);
  const [newCircuitInterpretation, setNewCircuitInterpretation] = useState('');
  const [editingCircuitId, setEditingCircuitId] = useState<string | null>(null);
  const [editingInterpretation, setEditingInterpretation] = useState('');
  const [editingFeatureCircuitId, setEditingFeatureCircuitId] = useState<string | null>(null);
  const [editingFeatureIndex, setEditingFeatureIndex] = useState<number | null>(null);
  const [editingFeatureInterpretation, setEditingFeatureInterpretation] = useState('');

  // 加载circuits
  const loadCircuits = useCallback(async () => {
    console.log('[CircuitInterpretation] loadCircuits called', { node });
    if (!node) {
      console.log('[CircuitInterpretation] loadCircuits: node is null, returning');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      console.log('[CircuitInterpretation] loadCircuits: calling getSaeName...');
      const isLorsa = node.feature_type?.toLowerCase() === 'lorsa';
      console.log('[CircuitInterpretation] loadCircuits: isLorsa =', isLorsa);
      const saeName = getSaeName(node.layer, isLorsa);
      console.log('[CircuitInterpretation] loadCircuits: saeName =', saeName);
      const featureType = isLorsa ? 'lorsa' : 'transcoder';
      console.log('[CircuitInterpretation] loadCircuits: featureType =', featureType);

      const result = await getCircuitsByFeature(
        saeName,
        saeSeries,
        node.layer,
        node.feature,
        featureType as 'transcoder' | 'lorsa'
      );

      setCircuits(result.circuits);
    } catch (err) {
      setError(err instanceof Error ? err.message : '加载circuit标注失败');
      console.error('Failed to load circuits:', err);
    } finally {
      setLoading(false);
    }
  }, [node, saeSeries, getSaeName]);

  // 当节点变化时重新加载
  useEffect(() => {
    if (visible && node) {
      loadCircuits();
    }
  }, [visible, node, loadCircuits]);

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

      console.log('[CircuitInterpretation] Creating circuit with:', {
        saeName,
        saeSeries,
        layer: node.layer,
        feature_index: node.feature,
        feature_type: featureType,
        saeComboId,
      });

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
      
      console.log('[CircuitInterpretation] Circuit created successfully');

      setNewCircuitInterpretation('');
      setCreatingNew(false);
      await loadCircuits();
    } catch (err) {
      setError(err instanceof Error ? err.message : '创建circuit失败');
      console.error('Failed to create circuit:', err);
    } finally {
      setLoading(false);
    }
  }, [node, newCircuitInterpretation, saeComboId, saeSeries, getSaeName, loadCircuits]);

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
      await loadCircuits();
    } catch (err) {
      setError(err instanceof Error ? err.message : '更新circuit解释失败');
      console.error('Failed to update circuit interpretation:', err);
    } finally {
      setLoading(false);
    }
  }, [editingInterpretation, loadCircuits]);

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
      await loadCircuits();
    } catch (err) {
      setError(err instanceof Error ? err.message : '添加feature到circuit失败');
      console.error('Failed to add feature to circuit:', err);
    } finally {
      setLoading(false);
    }
  }, [node, saeSeries, getSaeName, loadCircuits]);

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
      await loadCircuits();
    } catch (err) {
      setError(err instanceof Error ? err.message : '从circuit删除feature失败');
      console.error('Failed to remove feature from circuit:', err);
    } finally {
      setLoading(false);
    }
  }, [saeSeries, loadCircuits]);

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
      await loadCircuits();
    } catch (err) {
      setError(err instanceof Error ? err.message : '更新feature解释失败');
      console.error('Failed to update feature interpretation:', err);
    } finally {
      setLoading(false);
    }
  }, [saeSeries, editingFeatureInterpretation, loadCircuits]);

  // 删除circuit
  const handleDeleteCircuit = useCallback(async (circuitId: string) => {
    if (!confirm('确定要删除这个circuit标注吗？')) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      await deleteCircuitAnnotation(circuitId);
      await loadCircuits();
    } catch (err) {
      setError(err instanceof Error ? err.message : '删除circuit失败');
      console.error('Failed to delete circuit:', err);
    } finally {
      setLoading(false);
    }
  }, [loadCircuits]);

  console.log('[CircuitInterpretation] Render check', { visible, node });
  
  if (!visible || !node) {
    console.log('[CircuitInterpretation] Not rendering (visible=false or node=null)');
    return null;
  }

  console.log('[CircuitInterpretation] Rendering component');

  const isLorsa = node.feature_type?.toLowerCase() === 'lorsa';
  const currentSaeName = getSaeName(node.layer, isLorsa);
  const featureType = isLorsa ? 'lorsa' : 'transcoder';

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
          <h2 className="text-xl font-semibold text-gray-800">
            Circuit 标注 - {featureType === 'lorsa' ? 'LoRSA' : 'Transcoder'} L{node.layer} #{node.feature}
          </h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 text-2xl font-bold"
          >
            ×
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
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
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
                disabled={loading}
              >
                + 创建新 Circuit
              </button>
            ) : (
              <div className="border border-gray-300 rounded p-4">
                <h3 className="font-semibold mb-2">创建新 Circuit</h3>
                <textarea
                  value={newCircuitInterpretation}
                  onChange={(e) => setNewCircuitInterpretation(e.target.value)}
                  placeholder="输入circuit的整体解释..."
                  className="w-full p-2 border border-gray-300 rounded mb-2"
                  rows={3}
                />
                <div className="flex gap-2">
                  <button
                    onClick={handleCreateCircuit}
                    className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
                    disabled={loading || !newCircuitInterpretation.trim()}
                  >
                    创建
                  </button>
                  <button
                    onClick={() => {
                      setCreatingNew(false);
                      setNewCircuitInterpretation('');
                    }}
                    className="px-4 py-2 bg-gray-300 text-gray-700 rounded hover:bg-gray-400"
                  >
                    取消
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Circuits列表 */}
          {loading && circuits.length === 0 ? (
            <div className="text-center py-8 text-gray-500">加载中...</div>
          ) : circuits.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              该feature还没有属于任何circuit标注
            </div>
          ) : (
            <div className="space-y-4">
              {circuits.map((circuit) => (
                <div
                  key={circuit.circuit_id}
                  className="border border-gray-300 rounded p-4"
                >
                  {/* Circuit头部 */}
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      {editingCircuitId === circuit.circuit_id ? (
                        <div>
                          <textarea
                            value={editingInterpretation}
                            onChange={(e) => setEditingInterpretation(e.target.value)}
                            className="w-full p-2 border border-gray-300 rounded mb-2"
                            rows={2}
                          />
                          <div className="flex gap-2">
                            <button
                              onClick={() => handleUpdateCircuitInterpretation(circuit.circuit_id)}
                              className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700"
                              disabled={loading}
                            >
                              保存
                            </button>
                            <button
                              onClick={() => {
                                setEditingCircuitId(null);
                                setEditingInterpretation('');
                              }}
                              className="px-3 py-1 bg-gray-300 text-gray-700 rounded text-sm hover:bg-gray-400"
                            >
                              取消
                            </button>
                          </div>
                        </div>
                      ) : (
                        <div>
                          <p className="text-gray-700 mb-1">
                            <span className="font-semibold">Circuit解释:</span>{' '}
                            {circuit.circuit_interpretation || '(无)'}
                          </p>
                          <p className="text-sm text-gray-500">
                            Circuit ID: {circuit.circuit_id}
                          </p>
                        </div>
                      )}
                    </div>
                    <div className="flex gap-2 ml-4">
                      {editingCircuitId !== circuit.circuit_id && (
                        <>
                          <button
                            onClick={() => {
                              setEditingCircuitId(circuit.circuit_id);
                              setEditingInterpretation(circuit.circuit_interpretation);
                            }}
                            className="px-3 py-1 bg-yellow-600 text-white rounded text-sm hover:bg-yellow-700"
                            disabled={loading}
                          >
                            编辑
                          </button>
                          {!circuit.features.some(
                            (f) =>
                              f.sae_name === currentSaeName &&
                              f.layer === node.layer &&
                              f.feature_index === node.feature &&
                              f.feature_type === featureType
                          ) && (
                            <button
                              onClick={() => handleAddFeatureToCircuit(circuit.circuit_id)}
                              className="px-3 py-1 bg-green-600 text-white rounded text-sm hover:bg-green-700"
                              disabled={loading}
                            >
                              添加当前feature
                            </button>
                          )}
                          <button
                            onClick={() => handleDeleteCircuit(circuit.circuit_id)}
                            className="px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700"
                            disabled={loading}
                          >
                            删除
                          </button>
                        </>
                      )}
                    </div>
                  </div>

                  {/* Features列表 */}
                  <div className="mt-3 border-t border-gray-200 pt-3">
                    <h4 className="font-semibold mb-2 text-sm">Features ({circuit.features.length}):</h4>
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
                            className={`p-2 rounded text-sm ${
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
                                      onChange={(e) => setEditingFeatureInterpretation(e.target.value)}
                                      placeholder="输入feature解释..."
                                      className="w-full p-2 border border-gray-300 rounded text-xs"
                                      rows={2}
                                    />
                                    <div className="flex gap-2 mt-1">
                                      <button
                                        onClick={() =>
                                          handleUpdateFeatureInterpretation(
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
                                          setEditingFeatureCircuitId(null);
                                          setEditingFeatureIndex(null);
                                          setEditingFeatureInterpretation('');
                                        }}
                                        className="px-2 py-1 bg-gray-300 text-gray-700 rounded text-xs hover:bg-gray-400"
                                      >
                                        取消
                                      </button>
                                    </div>
                                  </div>
                                ) : (
                                  <div className="text-gray-700 mt-1">
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
                                        setEditingFeatureCircuitId(circuit.circuit_id);
                                        setEditingFeatureIndex(idx);
                                        setEditingFeatureInterpretation(feature.interpretation || '');
                                      }}
                                      className="px-2 py-1 bg-yellow-600 text-white rounded text-xs hover:bg-yellow-700"
                                      disabled={loading}
                                    >
                                      编辑
                                    </button>
                                    <button
                                      onClick={() =>
                                        handleRemoveFeatureFromCircuit(
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
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
