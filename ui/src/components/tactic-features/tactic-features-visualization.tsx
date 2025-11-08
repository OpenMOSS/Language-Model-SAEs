import React, { useState, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Loader2, Upload, FileText, ExternalLink } from 'lucide-react';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface FeatureResult {
  layer: number;
  feature: number;
  diff: number;
  p_random: number;
  p_tactic: number;
  kind: string;
}

interface AnalysisResult {
  valid_tactic_fens: number;
  invalid_tactic_fens: number;
  random_fens: number;
  tactic_fens: number;
  top_lorsa_features: FeatureResult[];
  top_tc_features: FeatureResult[];
  invalid_fens_sample: string[];
  specific_layer_lorsa?: FeatureResult[];
  specific_layer_tc?: FeatureResult[];
  specific_layer?: number;
}

export const TacticFeaturesVisualization: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState('lc0/BT4-1024x15x32h');
  const [availableModels, setAvailableModels] = useState([
    { name: 'lc0/T82-768x15x24h', display_name: 'T82-768x15x24h' },
    { name: 'lc0/BT4-1024x15x32h', display_name: 'BT4-1024x15x32h' },
  ]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [topKLorsa, setTopKLorsa] = useState<number>(10);
  const [topKTC, setTopKTC] = useState<number>(10);
  const [nFens, setNFens] = useState<number>(200);
  const [specificLayer, setSpecificLayer] = useState<string>('');
  const [specificLayerTopK, setSpecificLayerTopK] = useState<number>(20);

  // 获取可用模型列表
  const fetchAvailableModels = useCallback(async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/models`);
      if (response.ok) {
        const data = await response.json();
        setAvailableModels(data.models);
      }
    } catch (error) {
      console.error('获取模型列表失败:', error);
    }
  }, []);

  React.useEffect(() => {
    fetchAvailableModels();
  }, [fetchAvailableModels]);

  // 构建dictionary名称（参考circuit-visualization.tsx）
  const buildDictionaryName = useCallback((layer: number, kind: string): string => {
    const isBT4 = selectedModel.includes('BT4');
    
    if (kind === 'LoRSA') {
      if (isBT4) {
        return `BT4_lorsa_L${layer}A`;
      } else {
        return `lc0-lorsa-L${layer}`;
      }
    } else { // TC
      if (isBT4) {
        return `BT4_tc_L${layer}M`;
      } else {
        return `lc0_L${layer}M_16x_k30_lr2e-03_auxk_sparseadam`;
      }
    }
  }, [selectedModel]);

  // 处理文件选择
  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.type === 'text/plain' || file.name.endsWith('.txt')) {
        setSelectedFile(file);
        setError(null);
      } else {
        setError('请上传.txt文件');
        setSelectedFile(null);
      }
    }
  }, []);

  // 运行分析
  const runAnalysis = useCallback(async () => {
    if (!selectedFile) {
      setError('请先选择文件');
      return;
    }

    setIsLoading(true);
    setError(null);
    setAnalysisResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('model_name', selectedModel);
      formData.append('n_random', nFens.toString());
      formData.append('n_fens', nFens.toString());
      formData.append('top_k_lorsa', topKLorsa.toString());
      formData.append('top_k_tc', topKTC.toString());
      
      // 添加指定层参数
      if (specificLayer && !isNaN(parseInt(specificLayer))) {
        formData.append('specific_layer', specificLayer);
        formData.append('specific_layer_top_k', specificLayerTopK.toString());
      }
      
      console.log('🔍 发送分析请求:', {
        model_name: selectedModel,
        n_fens: nFens,
        top_k_lorsa: topKLorsa,
        top_k_tc: topKTC,
        specific_layer: specificLayer,
        specific_layer_top_k: specificLayerTopK
      });

      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/tactic_features/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        console.log('✅ 收到分析结果:', data);
        console.log('🔍 指定层数据检查:', {
          specific_layer: data.specific_layer,
          has_specific_layer_lorsa: !!data.specific_layer_lorsa,
          specific_layer_lorsa_length: data.specific_layer_lorsa?.length || 0,
          has_specific_layer_tc: !!data.specific_layer_tc,
          specific_layer_tc_length: data.specific_layer_tc?.length || 0,
        });
        setAnalysisResult(data);
      } else {
        const errorText = await response.text();
        setError(`分析失败: ${errorText}`);
      }
    } catch (error) {
      console.error('运行分析失败:', error);
      setError('运行分析失败，请检查后端服务');
    } finally {
      setIsLoading(false);
    }
  }, [selectedFile, selectedModel, nFens, topKLorsa, topKTC, specificLayer, specificLayerTopK]);

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <FileText className="w-8 h-8" />
          战术特征分析
        </h1>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 左侧：配置 */}
        <div className="space-y-4">
          {/* 模型选择 */}
          <Card>
            <CardHeader>
              <CardTitle>模型选择</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium">选择模型</label>
                <Select value={selectedModel} onValueChange={setSelectedModel}>
                  <SelectTrigger className="mt-1">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {availableModels.map((model) => (
                      <SelectItem key={model.name} value={model.name}>
                        {model.display_name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          {/* 文件上传 */}
          <Card>
            <CardHeader>
              <CardTitle>上传FEN文件</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium">选择文件 (.txt)</label>
                <div className="mt-2 flex items-center gap-2">
                  <Input
                    type="file"
                    accept=".txt"
                    onChange={handleFileChange}
                    className="cursor-pointer"
                  />
                </div>
                {selectedFile && (
                  <div className="mt-2 text-sm text-gray-600">
                    已选择: {selectedFile.name}
                  </div>
                )}
                {error && (
                  <div className="mt-2 text-sm text-red-600">{error}</div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* 参数配置 */}
          <Card>
            <CardHeader>
              <CardTitle>分析参数</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium">FEN数量</label>
                <Input
                  type="number"
                  min="1"
                  max="1000"
                  value={nFens}
                  onChange={(e) => setNFens(parseInt(e.target.value) || 200)}
                  className="mt-1"
                />
                <div className="text-xs text-gray-500 mt-1">
                  从txt文件和随机FEN中各取这么多条（如果文件中FEN少于此数量则全部使用）
                </div>
              </div>
              <div>
                <label className="text-sm font-medium">显示Top K LoRSA特征</label>
                <Input
                  type="number"
                  min="1"
                  max="100"
                  value={topKLorsa}
                  onChange={(e) => setTopKLorsa(parseInt(e.target.value) || 10)}
                  className="mt-1"
                />
              </div>
              <div>
                <label className="text-sm font-medium">显示Top K TC特征</label>
                <Input
                  type="number"
                  min="1"
                  max="100"
                  value={topKTC}
                  onChange={(e) => setTopKTC(parseInt(e.target.value) || 10)}
                  className="mt-1"
                />
              </div>
              <div className="border-t pt-4">
                <label className="text-sm font-medium">指定层分析（可选）</label>
                <Input
                  type="number"
                  min="0"
                  max="14"
                  value={specificLayer}
                  onChange={(e) => setSpecificLayer(e.target.value)}
                  placeholder="留空则不分析特定层"
                  className="mt-1"
                />
                <div className="text-xs text-gray-500 mt-1">
                  输入层号（0-14）以获取该层的详细特征
                </div>
              </div>
              <div>
                <label className="text-sm font-medium">指定层Top K特征数</label>
                <Input
                  type="number"
                  min="1"
                  max="100"
                  value={specificLayerTopK}
                  onChange={(e) => setSpecificLayerTopK(parseInt(e.target.value) || 20)}
                  className="mt-1"
                />
              </div>
            </CardContent>
          </Card>

          {/* 运行按钮 */}
          <Button
            onClick={runAnalysis}
            disabled={isLoading || !selectedFile}
            className="w-full"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                分析中...
              </>
            ) : (
              <>
                <Upload className="w-4 h-4 mr-2" />
                开始分析
              </>
            )}
          </Button>
        </div>

        {/* 右侧：结果展示 */}
        <div className="lg:col-span-2 space-y-4">
          {analysisResult ? (
            <>
              {/* 统计信息 */}
              <Card>
                <CardHeader>
                  <CardTitle>分析统计</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-sm text-gray-600">有效战术FEN</div>
                      <div className="text-2xl font-bold">{analysisResult.valid_tactic_fens}</div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-600">无效FEN</div>
                      <div className="text-2xl font-bold text-red-600">{analysisResult.invalid_tactic_fens}</div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-600">随机FEN</div>
                      <div className="text-2xl font-bold">{analysisResult.random_fens}</div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-600">处理的战术FEN</div>
                      <div className="text-2xl font-bold">{analysisResult.tactic_fens}</div>
                    </div>
                  </div>
                  {analysisResult.invalid_fens_sample.length > 0 && (
                    <div className="mt-4">
                      <div className="text-sm text-gray-600">无效FEN示例:</div>
                      <div className="text-xs font-mono bg-gray-100 p-2 rounded mt-1">
                        {analysisResult.invalid_fens_sample.slice(0, 5).join(', ')}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* LoRSA特征 */}
              <Card>
                <CardHeader>
                  <CardTitle>Top {topKLorsa} LoRSA特征 (差异最大)</CardTitle>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>排名</TableHead>
                        <TableHead>层级</TableHead>
                        <TableHead>特征索引</TableHead>
                        <TableHead>差异 (p_tactic - p_random)</TableHead>
                        <TableHead>p_random</TableHead>
                        <TableHead>p_tactic</TableHead>
                        <TableHead>操作</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {analysisResult.top_lorsa_features.map((feat, idx) => {
                        const dictionary = buildDictionaryName(feat.layer, 'LoRSA');
                        const featureUrl = `/features?dictionary=${encodeURIComponent(dictionary)}&featureIndex=${feat.feature}`;
                        return (
                          <TableRow key={idx}>
                            <TableCell className="font-medium">#{idx + 1}</TableCell>
                            <TableCell>Layer {feat.layer}</TableCell>
                            <TableCell>
                              <Badge variant="outline">Feature {feat.feature}</Badge>
                            </TableCell>
                            <TableCell className="font-bold text-green-600">
                              {feat.diff.toFixed(6)}
                            </TableCell>
                            <TableCell>{feat.p_random.toFixed(6)}</TableCell>
                            <TableCell>{feat.p_tactic.toFixed(6)}</TableCell>
                            <TableCell>
                              <Link
                                to={featureUrl}
                                className="inline-flex items-center px-2 py-1 bg-blue-500 text-white text-xs font-medium rounded hover:bg-blue-600 transition-colors"
                                title={`查看Layer ${feat.layer} LoRSA Feature #${feat.feature}`}
                              >
                                <ExternalLink className="w-3 h-3 mr-1" />
                                查看
                              </Link>
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>

              {/* TC特征 */}
              <Card>
                <CardHeader>
                  <CardTitle>Top {topKTC} TC特征 (差异最大)</CardTitle>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>排名</TableHead>
                        <TableHead>层级</TableHead>
                        <TableHead>特征索引</TableHead>
                        <TableHead>差异 (p_tactic - p_random)</TableHead>
                        <TableHead>p_random</TableHead>
                        <TableHead>p_tactic</TableHead>
                        <TableHead>操作</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {analysisResult.top_tc_features.map((feat, idx) => {
                        const dictionary = buildDictionaryName(feat.layer, 'TC');
                        const featureUrl = `/features?dictionary=${encodeURIComponent(dictionary)}&featureIndex=${feat.feature}`;
                        return (
                          <TableRow key={idx}>
                            <TableCell className="font-medium">#{idx + 1}</TableCell>
                            <TableCell>Layer {feat.layer}</TableCell>
                            <TableCell>
                              <Badge variant="outline">Feature {feat.feature}</Badge>
                            </TableCell>
                            <TableCell className="font-bold text-green-600">
                              {feat.diff.toFixed(6)}
                            </TableCell>
                            <TableCell>{feat.p_random.toFixed(6)}</TableCell>
                            <TableCell>{feat.p_tactic.toFixed(6)}</TableCell>
                            <TableCell>
                              <Link
                                to={featureUrl}
                                className="inline-flex items-center px-2 py-1 bg-blue-500 text-white text-xs font-medium rounded hover:bg-blue-600 transition-colors"
                                title={`查看Layer ${feat.layer} TC Feature #${feat.feature}`}
                              >
                                <ExternalLink className="w-3 h-3 mr-1" />
                                查看
                              </Link>
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>

              {/* 指定层的LoRSA特征 */}
              {analysisResult.specific_layer !== undefined && analysisResult.specific_layer !== null && (
                <Card className="border-2 border-purple-200">
                  <CardHeader className="bg-purple-50">
                    <CardTitle>Layer {analysisResult.specific_layer} - Top {specificLayerTopK} LoRSA特征</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {analysisResult.specific_layer_lorsa && analysisResult.specific_layer_lorsa.length > 0 ? (
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>排名</TableHead>
                            <TableHead>特征索引</TableHead>
                            <TableHead>差异 (p_tactic - p_random)</TableHead>
                            <TableHead>p_random</TableHead>
                            <TableHead>p_tactic</TableHead>
                            <TableHead>操作</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {analysisResult.specific_layer_lorsa.map((feat, idx) => {
                          const dictionary = buildDictionaryName(feat.layer, 'LoRSA');
                          const featureUrl = `/features?dictionary=${encodeURIComponent(dictionary)}&featureIndex=${feat.feature}`;
                          return (
                            <TableRow key={idx}>
                              <TableCell className="font-medium">#{idx + 1}</TableCell>
                              <TableCell>
                                <Badge variant="outline">Feature {feat.feature}</Badge>
                              </TableCell>
                              <TableCell className="font-bold text-purple-600">
                                {feat.diff.toFixed(6)}
                              </TableCell>
                              <TableCell>{feat.p_random.toFixed(6)}</TableCell>
                              <TableCell>{feat.p_tactic.toFixed(6)}</TableCell>
                              <TableCell>
                                <Link
                                  to={featureUrl}
                                  className="inline-flex items-center px-2 py-1 bg-blue-500 text-white text-xs font-medium rounded hover:bg-blue-600 transition-colors"
                                  title={`查看Layer ${feat.layer} LoRSA Feature #${feat.feature}`}
                                >
                                  <ExternalLink className="w-3 h-3 mr-1" />
                                  查看
                                </Link>
                              </TableCell>
                            </TableRow>
                          );
                        })}
                        </TableBody>
                      </Table>
                    ) : (
                      <div className="text-center py-8 text-gray-500">
                        <p>Layer {analysisResult.specific_layer} 没有找到 LoRSA 特征</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}

              {/* 指定层的TC特征 */}
              {analysisResult.specific_layer !== undefined && analysisResult.specific_layer !== null && (
                <Card className="border-2 border-purple-200">
                  <CardHeader className="bg-purple-50">
                    <CardTitle>Layer {analysisResult.specific_layer} - Top {specificLayerTopK} TC特征</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {analysisResult.specific_layer_tc && analysisResult.specific_layer_tc.length > 0 ? (
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>排名</TableHead>
                            <TableHead>特征索引</TableHead>
                            <TableHead>差异 (p_tactic - p_random)</TableHead>
                            <TableHead>p_random</TableHead>
                            <TableHead>p_tactic</TableHead>
                            <TableHead>操作</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {analysisResult.specific_layer_tc.map((feat, idx) => {
                          const dictionary = buildDictionaryName(feat.layer, 'TC');
                          const featureUrl = `/features?dictionary=${encodeURIComponent(dictionary)}&featureIndex=${feat.feature}`;
                          return (
                            <TableRow key={idx}>
                              <TableCell className="font-medium">#{idx + 1}</TableCell>
                              <TableCell>
                                <Badge variant="outline">Feature {feat.feature}</Badge>
                              </TableCell>
                              <TableCell className="font-bold text-purple-600">
                                {feat.diff.toFixed(6)}
                              </TableCell>
                              <TableCell>{feat.p_random.toFixed(6)}</TableCell>
                              <TableCell>{feat.p_tactic.toFixed(6)}</TableCell>
                              <TableCell>
                                <Link
                                  to={featureUrl}
                                  className="inline-flex items-center px-2 py-1 bg-blue-500 text-white text-xs font-medium rounded hover:bg-blue-600 transition-colors"
                                  title={`查看Layer ${feat.layer} TC Feature #${feat.feature}`}
                                >
                                  <ExternalLink className="w-3 h-3 mr-1" />
                                  查看
                                </Link>
                              </TableCell>
                            </TableRow>
                          );
                        })}
                        </TableBody>
                      </Table>
                    ) : (
                      <div className="text-center py-8 text-gray-500">
                        <p>Layer {analysisResult.specific_layer} 没有找到 TC 特征</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}
            </>
          ) : (
            <Card>
              <CardContent className="py-12">
                <div className="text-center text-gray-500">
                  <FileText className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>上传FEN文件并点击"开始分析"开始战术特征分析</p>
                  <p className="text-xs mt-2">文件应为.txt格式，每行一个FEN字符串</p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

