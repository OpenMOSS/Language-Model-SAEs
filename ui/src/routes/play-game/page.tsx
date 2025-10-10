import React, { useState, useCallback, useEffect } from 'react';
import { Chess } from 'chess.js';
import { ChessBoard } from '@/components/chess/chess-board';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Loader2, RotateCcw, Play, Square, Move, Undo2, ExternalLink } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import { useCircuitState } from '@/contexts/AppStateContext';
import { transformCircuitData } from '@/components/circuits/link-graph/utils';
// 新增导入
import { LinkGraphContainer } from '@/components/circuits/link-graph-container';
import { NodeConnections } from '@/components/circuits/node-connections';
import { FeatureCard } from '@/components/feature/feature-card';
import { Feature } from '@/types/feature';

interface GameState {
  fen: string;
  moves: string[];
  isGameOver: boolean;
  winner: string | null;
  isPlayerTurn: boolean;
}

interface StockfishAnalysis {
  bestMove: string;
  evaluation: number;
  depth: number;
  wdl?: {
    winProb: number;
    drawProb: number;
    lossProb: number;
  };
}

const OPENING_POSITIONS = [
  { name: "起始局面", fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" },
  { name: "意大利开局", fen: "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3" },
  { name: "西班牙开局", fen: "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3" },
  { name: "西西里防御", fen: "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2" },
  { name: "法兰西防御", fen: "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2" },
  { name: "卡罗-卡恩防御", fen: "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2" },
  { name: "古印度防御", fen: "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2" },
  { name: "英式开局", fen: "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1" },
];

export const PlayGamePage = () => {
  const navigate = useNavigate();
  const { setCircuitData } = useCircuitState();
  const [game, setGame] = useState<Chess>(new Chess());
  const [gameState, setGameState] = useState<GameState>({
    fen: game.fen(),
    moves: [],
    isGameOver: false,
    winner: null,
    isPlayerTurn: true,
  });
  const [selectedOpening, setSelectedOpening] = useState(OPENING_POSITIONS[0]);
  const [customFen, setCustomFen] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [analysis, setAnalysis] = useState<StockfishAnalysis | null>(null);
  const [gameMode, setGameMode] = useState<'player-vs-model' | 'analysis'>('player-vs-model');
  const [isAutoPlay, setIsAutoPlay] = useState(false);
  const [autoPlayInterval, setAutoPlayInterval] = useState<NodeJS.Timeout | null>(null);
  
  // 翻转棋盘设置
  const [autoFlipWhenBlack, setAutoFlipWhenBlack] = useState<boolean>(false);
  
  // 新增：对战模式与人类方
  const [matchMode, setMatchMode] = useState<'human-human' | 'human-model' | 'model-model'>('human-model');
  const [humanPlays, setHumanPlays] = useState<'w' | 'b'>('w');

  // 新增：手动输入走法的状态
  const [manualMove, setManualMove] = useState('');
  const [moveError, setMoveError] = useState('');

  // 新增：用于强制更新组件（由于 Chess 实例是可变的）
  const [dummy, setDummy] = useState(0);

  // 新增：Circuit Trace状态
  const [isTracing, setIsTracing] = useState(false);
  const [circuitTraceResult, setCircuitTraceResult] = useState<any>(null);

  // 新增：Circuit可视化相关状态
  const [circuitVisualizationData, setCircuitVisualizationData] = useState<any>(null);
  const [clickedNodeId, setClickedNodeId] = useState<string | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [pinnedNodeIds, setPinnedNodeIds] = useState<string[]>([]);
  const [hiddenNodeIds, setHiddenNodeIds] = useState<string[]>([]);
  const [selectedFeature, setSelectedFeature] = useState<Feature | null>(null);
  const [connectedFeatures, setConnectedFeatures] = useState<Feature[]>([]);
  const [isLoadingConnectedFeatures, setIsLoadingConnectedFeatures] = useState(false);

  // 当使用自定义FEN开始且轮到模型走子时，自动触发一次模型走棋
  const [justStartedFromCustomFen, setJustStartedFromCustomFen] = useState(false);

  // 根据模式与当前局面导出当前是否人类/模型回合
  const isWhiteToMove = game.turn() === 'w';
  const isHumanTurn = (
    (matchMode === 'human-human') ||
    (matchMode === 'human-model' && ((humanPlays === 'w' && isWhiteToMove) || (humanPlays === 'b' && !isWhiteToMove)))
  ) && !gameState.isGameOver;
  const isModelTurn = (
    (matchMode === 'model-model') ||
    (matchMode === 'human-model' && !isHumanTurn)
  ) && !gameState.isGameOver;

  // 更新游戏状态
  const updateGameState = useCallback((newGame: Chess) => {
    const historyVerbose = newGame.history({ verbose: true });
    const moves = historyVerbose.map(m => m.from + m.to + (m.promotion ? m.promotion : ''));
    const isGameOver = newGame.isGameOver();
    let winner = null;
    
    if (isGameOver) {
      if (newGame.isCheckmate()) {
        winner = newGame.turn() === 'w' ? 'Black' : 'White';
      } else if (newGame.isDraw()) {
        winner = 'Draw';
      }
    }

    setGameState({
      fen: newGame.fen(),
      moves,
      isGameOver,
      winner,
      isPlayerTurn: newGame.turn() === 'w', // 假设玩家执白
    });
  }, []);

  // 开始新游戏
  const startNewGame = useCallback((fen?: string) => {
    const newGame = new Chess(fen || selectedOpening.fen);
    setGame(newGame);
    updateGameState(newGame);
    setAnalysis(null);
    setIsAutoPlay(false);
    if (autoPlayInterval) {
      clearInterval(autoPlayInterval);
      setAutoPlayInterval(null);
    }
  }, [selectedOpening.fen, updateGameState, autoPlayInterval]);

  // 使用自定义FEN开始游戏
  const startWithCustomFen = useCallback(() => {
    if (customFen.trim()) {
      try {
        const testGame = new Chess(customFen.trim());
        startNewGame(customFen.trim());
        setJustStartedFromCustomFen(true);
      } catch (error) {
        alert('无效的FEN字符串');
      }
    }
  }, [customFen, startNewGame]);

  // 获取Stockfish分析
  const getStockfishAnalysis = useCallback(async (fen: string) => {
    try {
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/analyze/stockfish`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ fen }),
      });

      if (response.ok) {
        const data = await response.json();
        setAnalysis({
          bestMove: data.bestMove,
          evaluation: data.evaluation,
          depth: data.depth,
          wdl: data.wdl,
        });
        return data;
      } else {
        console.error('Stockfish分析失败:', response.status, response.statusText);
        const errorText = await response.text();
        console.error('错误详情:', errorText);
      }
    } catch (error) {
      console.error('获取Stockfish分析失败:', error);
    }
    return null;
  }, []);

  // 获取模型建议的移动
  const getModelMove = useCallback(async (fen: string) => {
    try {
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/play_game`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ fen }),
      });

      if (response.ok) {
        const data = await response.json();
        return data.move;
      } else {
        console.error('API调用失败:', response.status, response.statusText);
        const errorText = await response.text();
        console.error('错误详情:', errorText);
      }
    } catch (error) {
      console.error('获取模型移动失败:', error);
    }
    return null;
  }, []);

  // 将UCI字符串转换为chess.js可接受的move对象
  const toChessJsMove = useCallback((move: string) => {
    const m = (move || '').trim();
    const uciMatch = m.match(/^([a-h][1-8])([a-h][1-8])([qrbn])?$/i);
    if (uciMatch) {
      return {
        from: uciMatch[1],
        to: uciMatch[2],
        promotion: uciMatch[3] ? uciMatch[3].toLowerCase() : undefined,
      } as any;
    }
    return m; // 兼容SAN等其他格式
  }, []);

  // 当轮到黑方走且棋盘点击以白方视角产生UCI时，镜像rank（e2e3 -> e7e6）
  const mirrorUciRanks = useCallback((uci: string) => {
    const m = (uci || '').trim();
    const match = m.match(/^([a-h])([1-8])([a-h])([1-8])([qrbn])?$/i);
    if (!match) return uci;
    const fromFile = match[1];
    const fromRank = 9 - parseInt(match[2], 10);
    const toFile = match[3];
    const toRank = 9 - parseInt(match[4], 10);
    const promo = match[5] ? match[5].toLowerCase() : '';
    return `${fromFile}${fromRank}${toFile}${toRank}${promo}`;
  }, []);

  // 执行移动
  const makeMove = useCallback((move: any) => {
    try {
      const parsed0 = typeof move === 'string' ? toChessJsMove(move) : move;
      const parsed = typeof parsed0 === 'string' ? parsed0 : {
        from: String(parsed0.from || '').toLowerCase(),
        to: String(parsed0.to || '').toLowerCase(),
        promotion: parsed0.promotion ? String(parsed0.promotion).toLowerCase() : undefined,
      } as any;

      // 若是字符串（如SAN），直接尝试；否则先用合法走法校验
      if (typeof parsed === 'string') {
        const result = game.move(parsed as any);
        if (!result) return false;
      } else {
        const legal = game.moves({ verbose: true }) as any[];
        const isLegal = legal.some(m => m.from === parsed.from && m.to === parsed.to && (parsed.promotion ? m.promotion === parsed.promotion : true));
        if (!isLegal) return false;
        const result = game.move(parsed as any);
        if (!result) return false;
      }

      // 在替换实例前，先从当前game提取完整历史与状态
      const historyVerbose = game.history({ verbose: true }) as any[];
      const movesUci = historyVerbose.map(m => m.from + m.to + (m.promotion ? m.promotion : ''));
      const nextFen = game.fen();
      const isGameOver = game.isGameOver();
      let winner: string | null = null;
      if (isGameOver) {
        if (game.isCheckmate()) {
          winner = game.turn() === 'w' ? 'Black' : 'White';
        } else if (game.isDraw()) {
          winner = 'Draw';
        }
      }
      const isPlayerTurn = game.turn() === 'w';

      // 用PGN重建以保留历史，避免FEN丢失undo栈
      const pgn = game.pgn();
      const replaced = new Chess();
      // 新版chess.js使用 loadPgn，旧版为 load_pgn，这里两者都尝试
      if (typeof (replaced as any).loadPgn === 'function') {
        (replaced as any).loadPgn(pgn);
      } else if (typeof (replaced as any).load_pgn === 'function') {
        (replaced as any).load_pgn(pgn);
      }
      setGame(replaced);

      // 直接设置游戏状态，保留完整历史
      setGameState({
        fen: nextFen,
        moves: movesUci,
        isGameOver,
        winner,
        isPlayerTurn,
      });

      setDummy(prev => prev + 1); // 强制触发刷新
      return true;
    } catch (error) {
      console.error('移动失败:', error);
    }
    return false;
  }, [game, toChessJsMove]);

  // 处理玩家移动：仅在人类回合允许
  const handlePlayerMove = useCallback((move: string) => {
    if (!isHumanTurn) return false;

    // 若当前轮到黑方走子，则将UCI按rank镜像（棋盘点击以白视角）
    const uci = game.turn() === 'b' ? mirrorUciRanks(move) : move;

    const ok = makeMove(uci);
    if (ok) {
      if (gameMode === 'analysis') {
        setTimeout(() => {
          getStockfishAnalysis(game.fen());
        }, 100);
      }
      // 走棋成功后清除Circuit Trace结果
      setCircuitTraceResult(null);
      return true;
    }
    return false;
  }, [isHumanTurn, makeMove, gameMode, getStockfishAnalysis, game, mirrorUciRanks]);

  // 处理模型移动：仅在模型回合触发
  const handleModelMove = useCallback(async () => {
    if (isModelTurn && !isLoading && !isTracing) {
      setIsLoading(true);
      try {
        const modelMove = await getModelMove(game.fen());
        if (!modelMove) {
          console.warn('模型未返回走法或返回为空');
          alert('模型未返回走法，请检查后端或当前局面');
          return;
        }
        if (modelMove && makeMove(modelMove)) {
          // 获取新局面的分析
          if (gameMode === 'analysis') {
            setTimeout(() => {
              getStockfishAnalysis(game.fen());
            }, 100);
          }
          setCircuitTraceResult(null);
        } else {
          alert('模型给出的走法无效或不合法');
        }
      } finally {
        setIsLoading(false);
      }
    }
  }, [isModelTurn, isLoading, isTracing, getModelMove, game, makeMove, gameMode, getStockfishAnalysis]);

  // 自动对局：
  // - human-human: 自动无意义，保持现状（只控制随机玩家步原逻辑保留）
  // - human-model: 人类回合不自动；模型回合自动调用模型
  // - model-model: 双方都用模型
  const toggleAutoPlay = useCallback(() => {
    if (isAutoPlay) {
      if (autoPlayInterval) {
        clearInterval(autoPlayInterval);
        setAutoPlayInterval(null);
      }
      setIsAutoPlay(false);
    } else {
      setIsAutoPlay(true);
      const interval = setInterval(() => {
        if (!gameState.isGameOver) {
          if (matchMode === 'model-model') {
            handleModelMove();
          } else if (matchMode === 'human-model') {
            if (isModelTurn) handleModelMove();
          } else {
            // human-human: 不自动
          }
        } else {
          setIsAutoPlay(false);
          if (autoPlayInterval) {
            clearInterval(autoPlayInterval);
            setAutoPlayInterval(null);
          }
        }
      }, 2000);
      setAutoPlayInterval(interval);
    }
  }, [isAutoPlay, autoPlayInterval, gameState.isGameOver, matchMode, isModelTurn, handleModelMove]);

  // 清理定时器
  useEffect(() => {
    return () => {
      if (autoPlayInterval) {
        clearInterval(autoPlayInterval);
      }
    };
  }, [autoPlayInterval]);

  // 获取当前局面的分析
  const getCurrentAnalysis = useCallback(async () => {
    await getStockfishAnalysis(game.fen());
  }, [game, getStockfishAnalysis]);

  // 退回上一步，直接在现有 game 实例上撤销移动（保留历史）
  const undoLastMove = useCallback(() => {
    try {
      const last = game.undo();
      if (!last) return; // 无可撤销

      const historyVerbose = game.history({ verbose: true }) as any[];
      const movesUci = historyVerbose.map(m => m.from + m.to + (m.promotion ? m.promotion : ''));
      const nextFen = game.fen();
      const isGameOver = game.isGameOver();
      let winner: string | null = null;
      if (isGameOver) {
        if (game.isCheckmate()) {
          winner = game.turn() === 'w' ? 'Black' : 'White';
        } else if (game.isDraw()) {
          winner = 'Draw';
        }
      }
      const isPlayerTurn = game.turn() === 'w';

      // 用PGN重建以保留历史
      const pgn = game.pgn();
      const replaced = new Chess();
      if (typeof (replaced as any).loadPgn === 'function') {
        (replaced as any).loadPgn(pgn);
      } else if (typeof (replaced as any).load_pgn === 'function') {
        (replaced as any).load_pgn(pgn);
      }
      setGame(replaced);

      setGameState({
        fen: nextFen,
        moves: movesUci,
        isGameOver,
        winner,
        isPlayerTurn,
      });

      setDummy(prev => prev + 1);
      if (gameMode === 'analysis') {
        setTimeout(() => { getStockfishAnalysis(nextFen); }, 100);
      }
    } catch (error) {
      console.error('撤销失败:', error);
    }
  }, [game, gameMode, getStockfishAnalysis]);

  // 修改手动输入为兼容UCI
  const handleManualMove = useCallback(() => {
    if (!manualMove.trim()) {
      setMoveError('请输入走法');
      return;
    }
    if (!isHumanTurn) {
      setMoveError('现在不是人类回合');
      return;
    }
    if (gameState.isGameOver) {
      setMoveError('游戏已结束');
      return;
    }
    try {
      const newGame = new Chess(game.fen());
      const parsed = toChessJsMove(manualMove.trim());
      const result = newGame.move(parsed as any);
      
      if (result) {
        setGame(newGame);
        updateGameState(newGame);
        setManualMove('');
        setMoveError('');
        
        if (gameMode === 'analysis') {
          setTimeout(() => {
            getStockfishAnalysis(newGame.fen());
          }, 100);
        }
        setCircuitTraceResult(null);
      } else {
        setMoveError('无效的走法');
      }
    } catch (error) {
      setMoveError('无效的走法格式');
    }
  }, [manualMove, isHumanTurn, gameState.isGameOver, game, updateGameState, gameMode, getStockfishAnalysis, toChessJsMove]);

  // 新增：获取合法走法列表
  const getLegalMoves = useCallback(() => {
    return game.moves();
  }, [game]);

  // 新增：验证走法格式
  const validateMove = useCallback((move: string) => {
    const legalMoves = game.moves();
    return legalMoves.includes(move.trim());
  }, [game]);

  // 新增：handleCircuitTrace函数
  const handleCircuitTraceResult = useCallback((result: any) => {
    if (result && result.nodes) {
      try {
        const transformedData = transformCircuitData(result);
        setCircuitVisualizationData(transformedData);
        setCircuitTraceResult(result);
      } catch (error) {
        console.error('Circuit数据转换失败:', error);
        alert('Circuit数据转换失败: ' + (error instanceof Error ? error.message : '未知错误'));
      }
    }
  }, []);

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

  // 修改handleCircuitTrace函数
  const handleCircuitTrace = useCallback(async () => {
    const historyVerbose = game.history({ verbose: true });
    if (historyVerbose.length === 0) {
      alert('请先走一步棋，再进行Circuit Trace');
      return;
    }
    const lastVerbose = historyVerbose[historyVerbose.length - 1];
    const moveUci = lastVerbose.from + lastVerbose.to + (lastVerbose.promotion ? lastVerbose.promotion : '');
    setIsTracing(true);
    try {
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/circuit_trace`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ fen: game.fen(), move_uci: moveUci }),
      });
      if (response.ok) {
        const data = await response.json();
        handleCircuitTraceResult(data);
      } else {
        const errorText = await response.text();
        console.error('Circuit trace API调用失败:', response.status, response.statusText, errorText);
        alert('Circuit trace失败: ' + errorText);
      }
    } catch (error) {
      console.error('Circuit trace出错:', error);
      alert('Circuit trace出错');
    } finally {
      setIsTracing(false);
    }
  }, [game, handleCircuitTraceResult]);

  // 新增：处理跳转到circuits页面
  const handleViewCircuitGraph = useCallback(() => {
    if (!circuitTraceResult) return;
    
    try {
      console.log('🔍 转换Circuit Trace数据:', circuitTraceResult);
      const transformedData = transformCircuitData(circuitTraceResult);
      setCircuitData(transformedData);
      navigate('/circuits');
    } catch (error: any) {
      console.error('❌ 数据转换失败:', error);
      alert('无法加载Circuit可视化，数据格式不匹配: ' + error.message);
    }
  }, [circuitTraceResult, setCircuitData, navigate]);

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
      const fenParts = (gameState?.fen || '').split(' ');
      const fullmove = fenParts.length >= 6 && !Number.isNaN(parseInt(fenParts[5]))
        ? parseInt(fenParts[5])
        : Math.max(1, Math.ceil((gameState?.moves?.length || 0) / 2));
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
  }, [circuitTraceResult, circuitVisualizationData, gameState.fen, gameState.moves.length]);

  useEffect(() => {
    if (justStartedFromCustomFen) {
      // 仅在从自定义FEN启动后的首次渲染尝试触发一次模型走棋
      if (isModelTurn && !isLoading && !isTracing) {
        handleModelMove();
      }
      setJustStartedFromCustomFen(false);
    }
  }, [justStartedFromCustomFen, isModelTurn, isLoading, isTracing, handleModelMove]);

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">与模型对局</h1>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-sm">
            <span>黑方回合自动翻转</span>
            <Switch checked={autoFlipWhenBlack} onCheckedChange={setAutoFlipWhenBlack} />
          </div>
        <div className="flex gap-2">
          <Button
            onClick={() => startNewGame()}
            variant="outline"
            size="sm"
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            新游戏
          </Button>
            <Button
              onClick={undoLastMove}
              variant="outline"
              size="sm"
              disabled={game.history().length === 0}
            >
              <Undo2 className="w-4 h-4 mr-2" />
              退回一步
          </Button>
          <Button
            onClick={toggleAutoPlay}
            variant={isAutoPlay ? "destructive" : "default"}
            size="sm"
          >
            {isAutoPlay ? (
              <>
                <Square className="w-4 h-4 mr-2" />
                停止自动
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                自动对局
              </>
            )}
          </Button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 棋盘区域 */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>棋盘</span>
                <div className="flex gap-2">
                  <Badge variant={isHumanTurn ? "default" : "secondary"}>
                    {isHumanTurn ? "人类回合" : "模型回合"}
                  </Badge>
                  {gameState.isGameOver && (
                    <Badge variant="destructive">
                      {gameState.winner ? `游戏结束: ${gameState.winner} 获胜` : "平局"}
                    </Badge>
                  )}
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex justify-center">
                <ChessBoard
                  fen={gameState.fen}
                  size="large"
                  showCoordinates={true}
                  onMove={(uci) => handlePlayerMove(uci)}
                  onSquareClick={(square) => {
                    console.log('点击格子:', square);
                  }}
                  // 禁用交互如果不是人类回合或者正在trace
                  isInteractive={isHumanTurn && !isTracing}
                  autoFlipWhenBlack={autoFlipWhenBlack}
                  analysisName="对局棋盘"
                  showSelfPlay={true}  // 启用自对弈功能
                />
              </div>
              
              {/* 移动历史 */}
              <div className="mt-4">
                <h3 className="text-sm font-medium mb-2">移动历史</h3>
                <div className="bg-gray-50 p-3 rounded max-h-32 overflow-y-auto">
                  <div className="text-sm font-mono">
                    {gameState.moves.length > 0 ? gameState.moves.join(' ') : '暂无移动'}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* 控制面板 */}
        <div className="space-y-4">
          {/* 模式与人类方选择 */}
          <Card>
            <CardHeader>
              <CardTitle>对战设置</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 gap-3">
                <div>
                  <label className="text-sm font-medium">对战模式</label>
                  <Select value={matchMode} onValueChange={(v: 'human-human' | 'human-model' | 'model-model') => setMatchMode(v)}>
                    <SelectTrigger className="mt-1">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="human-human">人类 vs 人类</SelectItem>
                      <SelectItem value="human-model">人类 vs 模型</SelectItem>
                      <SelectItem value="model-model">模型 vs 模型</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {matchMode === 'human-model' && (
                  <div>
                    <label className="text-sm font-medium">人类执子</label>
                    <Select value={humanPlays} onValueChange={(v: 'w' | 'b') => setHumanPlays(v)}>
                      <SelectTrigger className="mt-1">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="w">白方</SelectItem>
                        <SelectItem value="b">黑方</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* 新增：手动输入走法 */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Move className="w-4 h-4" />
                手动走棋
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Input
                  placeholder="输入走法 (如: e2e4, Nf3, O-O)"
                  value={manualMove}
                  onChange={(e) => {
                    setManualMove(e.target.value);
                    setMoveError('');
                  }}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      handleManualMove();
                    }
                  }}
                  disabled={!isHumanTurn}
                />
                {moveError && (
                  <div className="text-sm text-red-600">{moveError}</div>
                )}
                <Button
                  onClick={handleManualMove}
                  disabled={!isHumanTurn || !manualMove.trim()}
                  className="w-full"
                >
                  执行走法
                </Button>
              </div>
              
              {/* 显示合法走法 */}
              <div className="space-y-2">
                <div className="text-sm font-medium">合法走法:</div>
                <div className="bg-gray-50 p-2 rounded max-h-24 overflow-y-auto">
                  <div className="text-xs font-mono">
                    {game.moves().slice(0, 20).join(', ')}
                    {game.moves().length > 20 && '...'}
                  </div>
                </div>
                <div className="text-xs text-gray-500">
                  共 {game.moves().length} 种合法走法
                </div>
              </div>
            </CardContent>
          </Card>

          {/* 游戏模式（保留原分析模式切换，与上方对战设置不同维度） */}
          <Card>
            <CardHeader>
              <CardTitle>游戏模式</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Select
                value={gameMode}
                onValueChange={(value: 'player-vs-model' | 'analysis') => setGameMode(value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="player-vs-model">玩家 vs 模型</SelectItem>
                  <SelectItem value="analysis">分析模式</SelectItem>
                </SelectContent>
              </Select>
            </CardContent>
          </Card>

          {/* 开局选择 */}
          <Card>
            <CardHeader>
              <CardTitle>选择开局</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Select
                value={selectedOpening.name}
                onValueChange={(value) => {
                  const opening = OPENING_POSITIONS.find(o => o.name === value);
                  if (opening) setSelectedOpening(opening);
                }}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {OPENING_POSITIONS.map((opening) => (
                    <SelectItem key={opening.name} value={opening.name}>
                      {opening.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              
              <Button
                onClick={() => startNewGame()}
                className="w-full"
                variant="outline"
              >
                使用此开局
              </Button>
            </CardContent>
          </Card>

          {/* 自定义FEN */}
          <Card>
            <CardHeader>
              <CardTitle>自定义局面</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Textarea
                placeholder="输入FEN字符串..."
                value={customFen}
                onChange={(e) => setCustomFen(e.target.value)}
                rows={3}
              />
              <Button
                onClick={startWithCustomFen}
                className="w-full"
                variant="outline"
                disabled={!customFen.trim()}
              >
                使用自定义FEN
              </Button>
            </CardContent>
          </Card>

          {/* 模型移动控制：仅在模型回合可用 */}
          <Card>
            <CardHeader>
              <CardTitle>模型控制</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button
                onClick={handleModelMove}
                disabled={!isModelTurn || isLoading || isTracing}
                className="w-full"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    思考中...
                  </>
                ) : (
                  '让模型走棋'
                )}
              </Button>
              <Button
                onClick={handleCircuitTrace}
                disabled={isTracing || gameState.isGameOver}
                variant={isTracing ? 'destructive' : 'default'}
                size="sm"
                className="w-full"
              >
                {isTracing ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Circuit Trace中...
                  </>
                ) : (
                  'Circuit Trace'
                )}
              </Button>
              
              {gameMode === 'analysis' && (
                <Button
                  onClick={getCurrentAnalysis}
                  variant="outline"
                  className="w-full"
                >
                  获取分析
                </Button>
              )}
            </CardContent>
          </Card>

          {/* 分析结果 */}
          {analysis && (
            <Card>
              <CardHeader>
                <CardTitle>分析结果</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="text-sm">
                  <strong>最佳移动:</strong> {analysis.bestMove}
                </div>
                <div className="text-sm">
                  <strong>评估:</strong> {analysis.evaluation ? analysis.evaluation.toFixed(2) : 'N/A'}
                </div>
                <div className="text-sm">
                  <strong>深度:</strong> {analysis.depth || 'N/A'}
                </div>
                {analysis.wdl && (
                  <div className="text-sm">
                    <strong>胜率:</strong> 
                    <div className="mt-1 space-y-1">
                      <div className="flex justify-between">
                        <span>白胜:</span>
                        <span>{(analysis.wdl.winProb * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>平局:</span>
                        <span>{(analysis.wdl.drawProb * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>黑胜:</span>
                        <span>{(analysis.wdl.lossProb * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* 游戏状态 */}
          <Card>
            <CardHeader>
              <CardTitle>游戏状态</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="text-sm">
                <strong>FEN:</strong>
                <div className="font-mono text-xs bg-gray-100 p-2 rounded mt-1 break-all">
                  {gameState.fen}
                </div>
              </div>
              <div className="text-sm">
                <strong>移动数:</strong> {gameState.moves.length}
              </div>
              <div className="text-sm">
                <strong>当前回合:</strong> {game.turn() === 'w' ? '白方' : '黑方'}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* 新增：Circuit可视化区域 */}
      {circuitVisualizationData && (
        <div className="mt-6 space-y-6">
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

                {/* Bottom Row: Feature Card */}
                {clickedNodeId && (() => {
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
                  const dictionary = isLorsa 
                    ? `lc0-lorsa-L${layerIdx}`
                    : `lc0_L${layerIdx}M_16x_k30_lr2e-03_auxk_sparseadam`;
                  
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
        </div>
      )}

      {/* 保留原有的简单circuitTraceResult显示，但移除跳转按钮 */}
      {circuitTraceResult && !circuitVisualizationData && (
        <Card className="mt-6">
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
    </div>
  );
};
