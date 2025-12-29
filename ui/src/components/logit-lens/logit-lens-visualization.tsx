import React, { useState, useCallback } from 'react';
import { ChessBoard } from '@/components/chess/chess-board';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Loader2, Play, Layers, Eraser } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { SaeComboLoader } from '@/components/common/SaeComboLoader';

interface LayerMoveData {
  idx: number;
  uci: string;
  score: number;
  prob: number;
}

interface MoveRanking {
  move: string;
  layer_score: number;
  final_rank: number | null;
  final_score: number | null;
  final_prob?: number | null;
  rank_change: number | null;
}

interface TargetInfo {
  uci: string;
  rank: number | null;
  score: number | null;
  prob?: number | null;
  error?: string;
}

interface LayerAnalysis {
  top_legal_moves: LayerMoveData[];
  move_rankings: MoveRanking[];
  target: TargetInfo | null;
  final_top_move?: {
    uci: string;
    rank: number | null;
    score: number | null;
    prob?: number | null;
  } | null;
  logit_entropy?: number | null;
}

interface KLDivergence {
  from_layer: number;
  to_layer: number;
  kl_forward: number | null;
  kl_backward: number | null;
  jsd: number | null;  // Jensen-Shannon Divergence (真正的度量)
}

interface LayerToFinalKL {
  layer: number;
  kl_to_final: number | null;
  kl_from_final: number | null;
  jsd: number | null;  // Jensen-Shannon Divergence
}

interface LogitLensResult {
  fen: string;
  final_layer_predictions: LayerMoveData[];
  layer_analysis: Record<string, LayerAnalysis>;
  target_move: string | null;
  num_layers: number;
  model_used: string;
  final_top_move_uci?: string | null;
  layer_kl_divergences?: KLDivergence[];
  layer_to_final_kl?: LayerToFinalKL[];
}

interface AblationData {
  hook_point: string;
  layer: number;
  hook_type: string;
  top_legal_moves: LayerMoveData[];
  logit_diff_stats: {
    mean: number;
    std: number;
    max: number;
    min: number;
    l2_norm: number;
  };
  target: TargetInfo | null;
  target_diff?: {
    uci: string;
    original_rank: number | null;
    ablated_rank: number | null;
    delta_rank: number | null;
    original_score: number | null;
    ablated_score: number | null;
    delta_score: number | null;
    original_prob: number | null;
    ablated_prob: number | null;
    delta_prob: number | null;
  } | null;
  original_top_move: {
    uci: string;
    rank: number | null;
    score: number | null;
    prob: number | null;
  } | null;
  logit_entropy: number | null;
  n_tokens_in_mean: number | null;
  error?: string;
}

interface MeanAblationResult {
  fen: string;
  original_top_legal_moves: LayerMoveData[];
  original_top_move_uci: string | null;
  ablation_results: Record<string, AblationData>;
  target_move: string | null;
  original_target?: TargetInfo | null;
  num_layers: number;
  hook_types: string[];
  model_used: string;
}

export const LogitLensVisualization: React.FC = () => {
  const [fen, setFen] = useState('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  const [targetMove, setTargetMove] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<LogitLensResult | null>(null);
  const [selectedLayer, setSelectedLayer] = useState<number>(0);
  
  // Mean Ablation 相关状态
  const [isLoadingAblation, setIsLoadingAblation] = useState(false);
  const [ablationResult, setAblationResult] = useState<MeanAblationResult | null>(null);
  const [selectedHookType, setSelectedHookType] = useState<string>('attn_out');
  const [activeTab, setActiveTab] = useState<string>('logit-lens');

  // 运行Logit Lens分析（固定使用BT4模型）
  const runAnalysis = useCallback(async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/logit_lens/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          fen,
          target_move: targetMove || null,
          topk_vocab: 2000,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setAnalysisResult(data);
        setSelectedLayer(data.num_layers - 1); // 默认选择最后一层
      } else {
        const errorText = await response.text();
        console.error('分析失败:', errorText);
        alert(`分析失败: ${errorText}`);
      }
    } catch (error) {
      console.error('运行分析失败:', error);
      alert('运行分析失败，请检查后端服务');
    } finally {
      setIsLoading(false);
    }
  }, [fen, targetMove]);

  // 运行Mean Ablation分析（固定使用BT4模型）
  const runMeanAblation = useCallback(async () => {
    setIsLoadingAblation(true);
    try {
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/logit_lens/mean_ablation`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          fen,
          hook_types: ['attn_out', 'mlp_out'],
          target_move: targetMove || null,
          topk_vocab: 2000,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setAblationResult(data);
        setActiveTab('mean-ablation');
      } else {
        const errorText = await response.text();
        console.error('Mean Ablation分析失败:', errorText);
        alert(`Mean Ablation分析失败: ${errorText}`);
      }
    } catch (error) {
      console.error('运行Mean Ablation分析失败:', error);
      alert('运行Mean Ablation分析失败，请检查后端服务');
    } finally {
      setIsLoadingAblation(false);
    }
  }, [fen, targetMove]);

  // 获取当前选中层的分析数据
  const getLayerData = useCallback((layer: number): LayerAnalysis | null => {
    if (!analysisResult) return null;
    return analysisResult.layer_analysis[`layer_${layer}`] || null;
  }, [analysisResult]);

  const currentLayerData = getLayerData(selectedLayer);
  // 直接使用后端返回的“all-legal softmax 概率”，不要对 TopN 再做 softmax
  const currentLayerProbs = currentLayerData?.top_legal_moves.map((m) => m.prob) ?? [];
  const finalLayerProbs = currentLayerData?.move_rankings.map((r) => r.final_prob ?? null) ?? [];

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* 全局 BT4 SAE 组合选择（LoRSA / Transcoder） */}
      <SaeComboLoader />

      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <Layers className="w-8 h-8" />
          Logit Lens分析
        </h1>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 左侧：输入和控制 */}
        <div className="space-y-4">

          {/* FEN输入 */}
          <Card>
            <CardHeader>
              <CardTitle>局面设置</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium">FEN字符串</label>
                <Textarea
                  placeholder="输入FEN字符串..."
                  value={fen}
                  onChange={(e) => setFen(e.target.value)}
                  rows={3}
                  className="mt-1"
                />
              </div>
              <div>
                <label className="text-sm font-medium">目标移动 (可选)</label>
                <Input
                  placeholder="例如: e2e4"
                  value={targetMove}
                  onChange={(e) => setTargetMove(e.target.value)}
                  className="mt-1"
                />
                <div className="text-xs text-gray-500 mt-1">
                  输入UCI格式的移动来追踪其在各层的排名
                </div>
              </div>
              <div className="space-y-2">
                <Button
                  onClick={runAnalysis}
                  disabled={isLoading || !fen.trim()}
                  className="w-full"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      分析中...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 mr-2" />
                      运行Logit Lens分析
                    </>
                  )}
                </Button>
                <Button
                  onClick={runMeanAblation}
                  disabled={isLoadingAblation || !fen.trim()}
                  className="w-full"
                  variant="outline"
                >
                  {isLoadingAblation ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Mean Ablation分析中...
                    </>
                  ) : (
                    <>
                      <Eraser className="w-4 h-4 mr-2" />
                      运行Mean Ablation分析
                    </>
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* 棋盘显示 */}
          <Card>
            <CardHeader>
              <CardTitle>当前局面</CardTitle>
            </CardHeader>
            <CardContent>
              <ChessBoard
                fen={fen}
                size="medium"
                showCoordinates={true}
                isInteractive={false}
                analysisName="Logit Lens"
              />
            </CardContent>
          </Card>
        </div>

        {/* 右侧：分析结果 */}
        <div className="lg:col-span-2 space-y-4">
          {(analysisResult || ablationResult) ? (
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="logit-lens" disabled={!analysisResult}>
                  Logit Lens
                </TabsTrigger>
                <TabsTrigger value="mean-ablation" disabled={!ablationResult}>
                  Mean Ablation
                </TabsTrigger>
              </TabsList>
              
              {/* Logit Lens标签内容 */}
              <TabsContent value="logit-lens" className="space-y-4">
              {analysisResult && (
            <>
              {/* 层选择器 */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span>层级分析</span>
                    <Badge variant="outline">
                      使用模型: {analysisResult.model_used.split('/')[1]}
                    </Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <label className="text-sm font-medium">
                        选择层级: Layer {selectedLayer}
                      </label>
                      {analysisResult.target_move && currentLayerData?.target && (
                        <Badge variant={currentLayerData.target.rank && currentLayerData.target.rank <= 3 ? "default" : "secondary"}>
                          目标移动 {analysisResult.target_move}: 
                          {currentLayerData.target.rank ? ` Rank #${currentLayerData.target.rank}` : ' 不在前列'}
                        </Badge>
                      )}
                    </div>
                    <input
                      type="range"
                      min="0"
                      max={analysisResult.num_layers - 1}
                      value={selectedLayer}
                      onChange={(e) => setSelectedLayer(parseInt(e.target.value))}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                      <span>Layer 0</span>
                      <span>Layer {analysisResult.num_layers - 1}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* 当前层Top移动 */}
              {currentLayerData && (
                <Card>
                  <CardHeader>
                    <CardTitle>Layer {selectedLayer} - Top 10合法移动</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {typeof currentLayerData.logit_entropy === 'number' && (
                      <div className="mb-3 text-sm space-x-2">
                        <Badge 
                          variant={currentLayerData.logit_entropy < 2.0 ? "default" : currentLayerData.logit_entropy < 3.5 ? "secondary" : "outline"}
                          className={currentLayerData.logit_entropy < 2.0 ? "bg-green-600" : currentLayerData.logit_entropy < 3.5 ? "bg-yellow-600" : ""}
                        >
                          Logit 熵: {currentLayerData.logit_entropy.toFixed(4)}
                        </Badge>
                      </div>
                    )}
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="w-16">排名</TableHead>
                          <TableHead>移动</TableHead>
                          <TableHead>当前层分数</TableHead>
                          <TableHead>当前层概率</TableHead>
                          <TableHead>最终层排名</TableHead>
                          <TableHead>最终层分数</TableHead>
                          <TableHead>最终层概率</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {currentLayerData.top_legal_moves.map((move, idx) => {
                          const ranking = currentLayerData.move_rankings.find(r => r.move === move.uci);
                          const isTargetMove = analysisResult.target_move === move.uci;
                          
                          return (
                            <TableRow key={idx} className={isTargetMove ? 'bg-yellow-50' : ''}>
                              <TableCell className="font-medium">#{idx + 1}</TableCell>
                              <TableCell>
                                <span className={`font-mono ${isTargetMove ? 'font-bold text-yellow-700' : ''}`}>
                                  {move.uci}
                                </span>
                              </TableCell>
                              <TableCell>{move.score.toFixed(4)}</TableCell>
                              <TableCell>
                                <span className="text-blue-600 font-medium">
                                  {(currentLayerProbs[idx] * 100).toFixed(2)}%
                                </span>
                              </TableCell>
                              <TableCell>
                                {ranking?.final_rank ? (
                                  <Badge variant={ranking.final_rank <= 3 ? 'default' : 'secondary'}>
                                    #{ranking.final_rank}
                                  </Badge>
                                ) : 'N/A'}
                              </TableCell>
                              <TableCell>
                                {ranking?.final_score !== null && ranking?.final_score !== undefined
                                  ? ranking.final_score.toFixed(4)
                                  : 'N/A'}
                              </TableCell>
                              <TableCell>
                                {typeof finalLayerProbs[idx] === "number" ? (
                                  <span className="text-green-600 font-medium">
                                    {(finalLayerProbs[idx] * 100).toFixed(2)}%
                                  </span>
                                ) : 'N/A'}
                              </TableCell>
                            </TableRow>
                          );
                        })}
                      </TableBody>
                    </Table>
                  </CardContent>
                </Card>
              )}

              {/* 每层Top 3移动 + 最终层Top在各层的表现 */}
              <Card>
                <CardHeader>
                  <CardTitle>每层Top 3 及“最终层Top”在各层的排名/概率/分数</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="overflow-x-auto">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="w-24">层级</TableHead>
                          <TableHead>Logit 熵<br/></TableHead>
                          <TableHead>Top 1</TableHead>
                          <TableHead>Top 2</TableHead>
                          <TableHead>Top 3</TableHead>
                          <TableHead>最终Top@本层<br/>rank / prob / score</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {Array.from({ length: analysisResult!.num_layers }, (_, i) => {
                          const layerData = getLayerData(i);
                          const top3 = layerData?.top_legal_moves.slice(0, 3) || [];
                          const entropy = layerData?.logit_entropy;
                          return (
                            <TableRow key={i} className={i === selectedLayer ? 'bg-blue-50' : ''}>
                              <TableCell className="text-sm">
                                <strong>{`Layer ${i}`}</strong>
                              </TableCell>
                              <TableCell>
                                {typeof entropy === 'number' ? (
                                  <Badge 
                                    variant={entropy < 2.0 ? "default" : entropy < 3.5 ? "secondary" : "outline"}
                                    className={entropy < 2.0 ? "bg-green-600 text-white" : entropy < 3.5 ? "bg-yellow-600 text-white" : ""}
                                  >
                                    {entropy.toFixed(4)}
                                  </Badge>
                                ) : 'N/A'}
                              </TableCell>
                              {[0, 1, 2].map((rank) => {
                                const move = top3[rank];
                                const prob = move?.prob;
                                const isTargetMove = move && analysisResult?.target_move === move.uci;
                                return (
                                  <TableCell key={rank}>
                                    {move ? (
                                      <div className={`space-between ${isTargetMove ? 'bg-yellow-100 p-2 rounded' : ''}`}>
                                        <div className={`font-mono text-sm ${isTargetMove ? 'font-bold text-yellow-700' : ''}`}>{move.uci}</div>
                                        <div className="text-xs text-gray-600">分数: {move.score.toFixed(3)}</div>
                                        {typeof prob === "number" ? (
                                          <div className="text-xs text-gray-600">概率: {(prob * 100).toFixed(1)}%</div>
                                        ) : null}
                                      </div>
                                    ) : (
                                      <span className="text-xs text-gray-400">N/A</span>
                                    )}
                                  </TableCell>
                                );
                              })}
                              <TableCell>
                                {layerData?.final_top_move ? (
                                  <div className="text-xs">
                                    <div>rank: {layerData.final_top_move.rank ?? 'N/A'}</div>
                                    <div>prob: {layerData.final_top_move.prob !== undefined && layerData.final_top_move.prob !== null ? `${(layerData.final_top_move.prob * 100).toFixed(1)}%` : 'N/A'}</div>
                                    <div>score: {layerData.final_top_move.score !== undefined && layerData.final_top_move.score !== null ? layerData.final_top_move.score.toFixed(3) : 'N/A'}</div>
                                  </div>
                                ) : (
                                  <span className="text-xs text-gray-400">N/A</span>
                                )}
                              </TableCell>
                            </TableRow>
                          );
                        })}
                      </TableBody>
                    </Table>
                  </div>
                </CardContent>
              </Card>

              {/* JSD分析 - 检测相变 */}
              {analysisResult.layer_kl_divergences && analysisResult.layer_kl_divergences.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>层间KL和JSD</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {/* 相邻层之间的KL散度和JSD */}
                    <div>
                      <h4 className="text-sm font-semibold mb-2">相邻层之间的分布差异</h4>
                      <div className="overflow-x-auto">
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>从层</TableHead>
                              <TableHead>到层</TableHead>
                              <TableHead>JSD<br/></TableHead>
                              <TableHead>KL(当前→前一层)</TableHead>
                              <TableHead>KL(前一层→当前)</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {analysisResult.layer_kl_divergences.map((kl, idx) => {
                              const jsd = kl.jsd;
                              // 计算所有JSD的平均值和标准差，用于判断突变
                              const allJSDs = analysisResult.layer_kl_divergences!
                                .map(k => k.jsd)
                                .filter((k): k is number => k !== null);
                              const meanJSD = allJSDs.length > 0 
                                ? allJSDs.reduce((a, b) => a + b, 0) / allJSDs.length 
                                : 0;
                              const stdJSD = allJSDs.length > 0
                                ? Math.sqrt(allJSDs.reduce((sum, k) => sum + Math.pow(k - meanJSD, 2), 0) / allJSDs.length)
                                : 0;
                              const isPhaseTransition = jsd !== null && jsd > meanJSD + 1.5 * stdJSD;
                              
                              return (
                                <TableRow 
                                  key={idx}
                                  className={isPhaseTransition ? 'bg-red-50 border-red-300' : ''}
                                >
                                  <TableCell className="font-medium">Layer {kl.from_layer}</TableCell>
                                  <TableCell className="font-medium">Layer {kl.to_layer}</TableCell>
                                  <TableCell>
                                    {jsd !== null ? (
                                      <Badge 
                                        variant={isPhaseTransition ? "destructive" : jsd > meanJSD ? "secondary" : "outline"}
                                        className={isPhaseTransition ? "bg-red-600 text-white" : ""}
                                      >
                                        {jsd.toFixed(4)}
                                        {isPhaseTransition && <span className="ml-1">⚠️ 相变</span>}
                                      </Badge>
                                    ) : 'N/A'}
                                  </TableCell>
                                  <TableCell>
                                    {kl.kl_forward !== null ? kl.kl_forward.toFixed(4) : 'N/A'}
                                  </TableCell>
                                  <TableCell>
                                    {kl.kl_backward !== null ? kl.kl_backward.toFixed(4) : 'N/A'}
                                  </TableCell>
                                </TableRow>
                              );
                            })}
                          </TableBody>
                        </Table>
                      </div>
                    </div>

                    {/* 每层相对于最终层的KL散度和JSD */}
                    {analysisResult.layer_to_final_kl && analysisResult.layer_to_final_kl.length > 0 && (
                      <div>
                        <h4 className="text-sm font-semibold mb-2">每层相对于最终层的分布差异</h4>
                        <div className="overflow-x-auto">
                          <Table>
                            <TableHeader>
                              <TableRow>
                                <TableHead>层级</TableHead>
                                <TableHead>JSD<br/><span className="text-xs font-normal text-gray-500">(距离最终层)</span></TableHead>
                                <TableHead>KL(最终→当前)</TableHead>
                                <TableHead>KL(当前→最终)</TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {analysisResult.layer_to_final_kl.map((kl) => {
                                const jsd = kl.jsd;
                                return (
                                  <TableRow 
                                    key={kl.layer}
                                    className={kl.layer === selectedLayer ? 'bg-blue-50' : ''}
                                  >
                                    <TableCell className="font-medium">Layer {kl.layer}</TableCell>
                                    <TableCell>
                                      {jsd !== null ? (
                                        <Badge variant="outline">
                                          {jsd.toFixed(4)}
                                        </Badge>
                                      ) : 'N/A'}
                                    </TableCell>
                                    <TableCell>
                                      {kl.kl_to_final !== null ? kl.kl_to_final.toFixed(4) : 'N/A'}
                                    </TableCell>
                                    <TableCell>
                                      {kl.kl_from_final !== null ? kl.kl_from_final.toFixed(4) : 'N/A'}
                                    </TableCell>
                                  </TableRow>
                                );
                              })}
                            </TableBody>
                          </Table>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}
              
              {/* 目标移动追踪 */}
              {analysisResult.target_move && (
                <Card>
                  <CardHeader>
                    <CardTitle>目标移动追踪: {analysisResult.target_move}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>层级</TableHead>
                          <TableHead>排名</TableHead>
                          <TableHead>分数</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {Array.from({ length: analysisResult.num_layers }, (_, i) => {
                          const layerData = getLayerData(i);
                          const target = layerData?.target;
                          
                          return (
                            <TableRow key={i} className={i === selectedLayer ? 'bg-blue-50' : ''}>
                              <TableCell className="font-medium">Layer {i}</TableCell>
                              <TableCell>
                                {target?.rank ? (
                                  <Badge variant={target.rank <= 3 ? 'default' : 'secondary'}>
                                    #{target.rank}
                                  </Badge>
                                ) : (
                                  <span className="text-gray-400">{target?.error || 'N/A'}</span>
                                )}
                              </TableCell>
                              <TableCell>
                                {target?.score !== null && target?.score !== undefined
                                  ? target.score.toFixed(4)
                                  : 'N/A'}
                              </TableCell>
                            </TableRow>
                          );
                        })}
                      </TableBody>
                    </Table>
                  </CardContent>
                </Card>
              )}
            </>
          )}
              </TabsContent>
              
              {/* Mean Ablation标签内容 */}
              <TabsContent value="mean-ablation" className="space-y-4">
              {ablationResult && (
                <>
                  {/* Hook类型选择器 */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center justify-between">
                        <span>Mean Ablation分析</span>
                        <Badge variant="outline">
                          使用模型: {ablationResult.model_used.split('/')[1]}
                        </Badge>
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div>
                        <label className="text-sm font-medium">选择Hook类型</label>
                        <Select value={selectedHookType} onValueChange={setSelectedHookType}>
                          <SelectTrigger className="mt-1">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="attn_out">Attention Out</SelectItem>
                            <SelectItem value="mlp_out">MLP Out</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="text-sm">
                        <Badge variant="outline">原始Top移动: {ablationResult.original_top_move_uci || 'N/A'}</Badge>
                      </div>
                    </CardContent>
                  </Card>

                  {/* 每层Mean Ablation结果汇总 */}
                  <Card>
                    <CardHeader>
                      <CardTitle>各层Mean Ablation影响 ({selectedHookType})</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="overflow-x-auto">
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>层级</TableHead>
                              <TableHead>L2 Norm<br/><span className="text-xs font-normal text-gray-500">(影响程度)</span></TableHead>
                              <TableHead>熵<br/><span className="text-xs font-normal text-gray-500">(越小越集中)</span></TableHead>
                              <TableHead>Top 3移动</TableHead>
                              {ablationResult.target_move && (
                                <TableHead>目标移动排名</TableHead>
                              )}
                              {ablationResult.target_move && (
                                <TableHead>目标移动Δlogit/Δprob</TableHead>
                              )}
                              {ablationResult.original_top_move_uci && (
                                <TableHead>原Top移动排名</TableHead>
                              )}
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {Array.from({ length: ablationResult.num_layers }, (_, i) => {
                              const hookPoint = `blocks.${i}.hook_${selectedHookType}`;
                              const data = ablationResult.ablation_results[hookPoint];
                              
                              if (!data || data.error) {
                                return (
                                  <TableRow key={i}>
                                    <TableCell className="font-medium">Layer {i}</TableCell>
                                    <TableCell colSpan={5} className="text-red-500">
                                      错误: {data?.error || '数据不可用'}
                                    </TableCell>
                                  </TableRow>
                                );
                              }

                              const top3 = data.top_legal_moves.slice(0, 3);
                              const entropy = data.logit_entropy;
                              
                              return (
                                <TableRow key={i}>
                                  <TableCell className="font-medium">Layer {i}</TableCell>
                                  <TableCell>
                                    <Badge variant={data.logit_diff_stats.l2_norm > 10 ? 'destructive' : 'secondary'}>
                                      {data.logit_diff_stats.l2_norm.toFixed(2)}
                                    </Badge>
                                  </TableCell>
                                  <TableCell>
                                    {typeof entropy === 'number' ? (
                                      <Badge 
                                        variant={entropy < 2.0 ? "default" : entropy < 3.5 ? "secondary" : "outline"}
                                        className={entropy < 2.0 ? "bg-green-600 text-white" : entropy < 3.5 ? "bg-yellow-600 text-white" : ""}
                                      >
                                        {entropy.toFixed(4)}
                                      </Badge>
                                    ) : 'N/A'}
                                  </TableCell>
                                  <TableCell>
                                    <div className="space-y-1">
                                      {top3.map((move, idx) => {
                                        const isTarget = ablationResult.target_move === move.uci;
                                        return (
                                          <div key={idx} className={`text-xs ${isTarget ? 'font-bold text-yellow-700' : ''}`}>
                                            {idx + 1}. {move.uci} ({move.score.toFixed(3)})
                                          </div>
                                        );
                                      })}
                                    </div>
                                  </TableCell>
                                  {ablationResult.target_move && (
                                    <TableCell>
                                      {data.target ? (
                                        <Badge variant={data.target.rank && data.target.rank <= 3 ? 'default' : 'secondary'}>
                                          #{data.target.rank || 'N/A'}
                                        </Badge>
                                      ) : 'N/A'}
                                    </TableCell>
                                  )}
                                  {ablationResult.target_move && (
                                    <TableCell>
                                      {data.target_diff ? (
                                        <div className="text-xs space-y-1">
                                          <div className="font-mono">{data.target_diff.uci}</div>
                                          <div>
                                            Δlogit:{" "}
                                            {typeof data.target_diff.delta_score === "number"
                                              ? data.target_diff.delta_score.toFixed(4)
                                              : "N/A"}
                                          </div>
                                          <div>
                                            Δprob:{" "}
                                            {typeof data.target_diff.delta_prob === "number"
                                              ? data.target_diff.delta_prob.toFixed(6)
                                              : "N/A"}
                                          </div>
                                          <div className="text-gray-500">
                                            orig {typeof data.target_diff.original_score === "number" ? data.target_diff.original_score.toFixed(3) : "N/A"} /{" "}
                                            {typeof data.target_diff.original_prob === "number" ? (data.target_diff.original_prob * 100).toFixed(2) + "%" : "N/A"}
                                          </div>
                                          <div className="text-gray-500">
                                            ablt {typeof data.target_diff.ablated_score === "number" ? data.target_diff.ablated_score.toFixed(3) : "N/A"} /{" "}
                                            {typeof data.target_diff.ablated_prob === "number" ? (data.target_diff.ablated_prob * 100).toFixed(2) + "%" : "N/A"}
                                          </div>
                                        </div>
                                      ) : (
                                        "N/A"
                                      )}
                                    </TableCell>
                                  )}
                                  {ablationResult.original_top_move_uci && (
                                    <TableCell>
                                      {data.original_top_move ? (
                                        <Badge variant={data.original_top_move.rank && data.original_top_move.rank <= 3 ? 'default' : 'secondary'}>
                                          #{data.original_top_move.rank || 'N/A'}
                                        </Badge>
                                      ) : 'N/A'}
                                    </TableCell>
                                  )}
                                </TableRow>
                              );
                            })}
                          </TableBody>
                        </Table>
                      </div>
                    </CardContent>
                  </Card>

                  {/* 目标移动追踪 - Mean Ablation版本 */}
                  {ablationResult.target_move && (
                    <Card>
                      <CardHeader>
                        <CardTitle>目标移动追踪 (Mean Ablation): {ablationResult.target_move}</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>层级</TableHead>
                              <TableHead>Hook类型</TableHead>
                              <TableHead>Ablation后排名</TableHead>
                              <TableHead>Ablation后分数</TableHead>
                              <TableHead>Logit差异(L2)</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {ablationResult.hook_types.flatMap((hookType) =>
                              Array.from({ length: ablationResult.num_layers }, (_, i) => {
                                const hookPoint = `blocks.${i}.hook_${hookType}`;
                                const data = ablationResult.ablation_results[hookPoint];
                                
                                if (!data || data.error) return null;
                                
                                return (
                                  <TableRow key={`${i}-${hookType}`}>
                                    <TableCell className="font-medium">Layer {i}</TableCell>
                                    <TableCell>{hookType}</TableCell>
                                    <TableCell>
                                      {data.target && data.target.rank ? (
                                        <Badge variant={data.target.rank <= 3 ? 'default' : 'secondary'}>
                                          #{data.target.rank}
                                        </Badge>
                                      ) : 'N/A'}
                                    </TableCell>
                                    <TableCell>
                                      {data.target && data.target.score !== null && data.target.score !== undefined
                                        ? data.target.score.toFixed(4)
                                        : 'N/A'}
                                    </TableCell>
                                    <TableCell>
                                      <Badge variant="outline">
                                        {data.logit_diff_stats.l2_norm.toFixed(2)}
                                      </Badge>
                                    </TableCell>
                                  </TableRow>
                                );
                              }).filter(Boolean)
                            )}
                          </TableBody>
                        </Table>
                      </CardContent>
                    </Card>
                  )}
                </>
              )}
              </TabsContent>
            </Tabs>
          ) : (
            <Card>
              <CardContent className="py-12">
                <div className="text-center text-gray-500">
                  <Layers className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>输入FEN并点击"运行分析"开始Logit Lens或Mean Ablation分析</p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};
