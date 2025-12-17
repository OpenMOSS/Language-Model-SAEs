import React, { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { AppNavbar } from '@/components/app/navbar';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Loader2, ArrowLeft, ExternalLink } from 'lucide-react';
import { api } from '@/utils/api';

interface GlobalWeightFeature {
  name: string;
  weight: number;
}

interface GlobalWeightResult {
  feature_type: string;
  layer_idx: number;
  feature_idx: number;
  feature_name: string;
  features_in: GlobalWeightFeature[];
  features_out: GlobalWeightFeature[];
}

export const GlobalWeightPage: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  
  // 从URL参数获取初始值
  const initialFeatureType = searchParams.get('feature_type') || 'tc';
  const initialLayerIdx = parseInt(searchParams.get('layer_idx') || '0');
  const initialFeatureIdx = parseInt(searchParams.get('feature_idx') || '0');
  const initialSaeComboId = searchParams.get('sae_combo_id') || undefined;
  
  const [featureType, setFeatureType] = useState<'tc' | 'lorsa'>(initialFeatureType as 'tc' | 'lorsa');
  const [layerIdx, setLayerIdx] = useState<number>(initialLayerIdx);
  const [featureIdx, setFeatureIdx] = useState<number>(initialFeatureIdx);
  const [saeComboId, setSaeComboId] = useState<string | undefined>(initialSaeComboId);
  const [k, setK] = useState<number>(100);
  
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<GlobalWeightResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // 当URL参数变化时更新状态
  useEffect(() => {
    const featureTypeParam = searchParams.get('feature_type');
    const layerIdxParam = searchParams.get('layer_idx');
    const featureIdxParam = searchParams.get('feature_idx');
    const saeComboIdParam = searchParams.get('sae_combo_id');
    
    if (featureTypeParam) setFeatureType(featureTypeParam as 'tc' | 'lorsa');
    if (layerIdxParam) setLayerIdx(parseInt(layerIdxParam));
    if (featureIdxParam) setFeatureIdx(parseInt(featureIdxParam));
    if (saeComboIdParam) setSaeComboId(saeComboIdParam);
  }, [searchParams]);

  const fetchGlobalWeight = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const params = new URLSearchParams({
        model_name: 'lc0/BT4-1024x15x32h',
        feature_type: featureType,
        layer_idx: layerIdx.toString(),
        feature_idx: featureIdx.toString(),
        k: k.toString(),
      });
      
      if (saeComboId) {
        params.append('sae_combo_id', saeComboId);
      }
      
      const response = await api.get<GlobalWeightResult>(`/global_weight?${params.toString()}`);
      setResult(response);
      
      // 更新URL参数
      const newParams = new URLSearchParams({
        feature_type: featureType,
        layer_idx: layerIdx.toString(),
        feature_idx: featureIdx.toString(),
      });
      if (saeComboId) {
        newParams.append('sae_combo_id', saeComboId);
      }
      setSearchParams(newParams);
    } catch (err: any) {
      setError(err.message || '获取全局权重失败');
      console.error('Error fetching global weight:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // 如果URL中有参数，自动加载
    if (searchParams.get('layer_idx') && searchParams.get('feature_idx')) {
      fetchGlobalWeight();
    }
  }, []); // 只在组件挂载时执行一次

  const handleFeatureClick = (featureName: string) => {
    // 解析特征名称，格式: BT4_tc_L0M_k30_e16#123 或 BT4_lorsa_L0A_k30_e16#123
    const match = featureName.match(/BT4_(tc|lorsa)_L(\d+)(M|A)_k30_e16#(\d+)/);
    if (match) {
      const [, type, layer, , idx] = match;
      const newParams = new URLSearchParams({
        feature_type: type,
        layer_idx: layer,
        feature_idx: idx,
      });
      if (saeComboId) {
        newParams.append('sae_combo_id', saeComboId);
      }
      navigate(`/global-weight?${newParams.toString()}`);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <AppNavbar />
      <div className="container mx-auto p-6 space-y-6">
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
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
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

            <div className="mt-4">
              <Button onClick={fetchGlobalWeight} disabled={loading}>
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    计算中...
                  </>
                ) : (
                  '计算全局权重'
                )}
              </Button>
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
            <Card>
              <CardHeader>
                <CardTitle>当前特征: {result.feature_name}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <p><strong>类型:</strong> {result.feature_type === 'tc' ? 'TC (Transcoder)' : 'LoRSA'}</p>
                  <p><strong>层:</strong> {result.layer_idx}</p>
                  <p><strong>特征索引:</strong> {result.feature_idx}</p>
                </div>
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
                  <div className="space-y-2 max-h-[600px] overflow-y-auto">
                    {result.features_in.map((feature, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between p-2 rounded border hover:bg-muted cursor-pointer transition-colors"
                        onClick={() => handleFeatureClick(feature.name)}
                        title="点击查看该特征的全局权重"
                      >
                        <span className="font-mono text-sm flex-1 break-all">{feature.name}</span>
                        <span className="text-right font-semibold ml-4 whitespace-nowrap">
                          {feature.weight.toFixed(4)}
                        </span>
                      </div>
                    ))}
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
                  <div className="space-y-2 max-h-[600px] overflow-y-auto">
                    {result.features_out.map((feature, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between p-2 rounded border hover:bg-muted cursor-pointer transition-colors"
                        onClick={() => handleFeatureClick(feature.name)}
                        title="点击查看该特征的全局权重"
                      >
                        <span className="font-mono text-sm flex-1 break-all">{feature.name}</span>
                        <span className="text-right font-semibold ml-4 whitespace-nowrap">
                          {feature.weight.toFixed(4)}
                        </span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
