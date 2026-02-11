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
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Loader2, RotateCcw, Play, Square, Move, Undo2, Download } from 'lucide-react';
import { SaeComboLoader } from '@/components/common/SaeComboLoader';

interface GameState {
  fen: string;
  moves: string[];
  isGameOver: boolean;
  winner: string | null;
  isPlayerTurn: boolean;
  gameEndReason: 'checkmate' | 'resignation' | 'draw' | null;
}

interface TimerState {
  whiteTime: number; // 剩余时间（秒）
  blackTime: number;
  whiteIncrement: number; // 每步增加时间（秒）
  blackIncrement: number;
  isRunning: boolean;
  currentPlayer: 'w' | 'b';
  lastMoveTime: number;
}

interface PgnHeaders {
  Event: string;
  Date: string;
  Round: string;
  White: string;
  Black: string;
  Result: string;
  TimeControl: string;
  WhiteFideId: string;
  BlackFideId: string;
  WhiteTitle: string;
  BlackTitle: string;
  WhiteElo: string;
  BlackElo: string;
  WhiteTeam: string;
  BlackTeam: string;
  WhiteClock: string;
  BlackClock: string;
  Variant: string;
}

interface MoveWithTime {
  move: string;
  time: number;
  clock: string;
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

interface ModelMoveResponse {
  move: string;
  model_used?: string;
  search_used?: boolean;
  search_info?: {
    total_playouts: number;
    max_depth_reached: number;
    max_depth_limit: number;
  };
  trace_file_path?: string;
  trace_filename?: string;
}

interface GameVisualizationProps {
  onCircuitTrace?: (data: any) => void;
  onCircuitTraceStart?: () => void;
  onCircuitTraceEnd?: () => void;
  onGameStateUpdate?: (fen: string, moves: string[]) => void;
}

const OPENING_POSITIONS = [
  { name: "起始局面", fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" },
  { name: "意大利开局", fen: "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4" },
  { name: "西班牙开局", fen: "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3" },
  { name: "西西里防御", fen: "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2" },
  { name: "法兰西防御", fen: "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2" },
  { name: "斯堪的纳维亚防御", fen: "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2" },
  { name: "卡罗-卡恩防御", fen: "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2" },
  { name: "古印度防御", fen: "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3" },
  { name: "英式开局", fen: "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1" },
];

export const GameVisualization: React.FC<GameVisualizationProps> = ({
  onGameStateUpdate,
}) => {
  const [game, setGame] = useState<Chess>(new Chess());
  const [gameState, setGameState] = useState<GameState>({
    fen: game.fen(),
    moves: [],
    isGameOver: false,
    winner: null,
    isPlayerTurn: true,
    gameEndReason: null,
  });

  // 计时器状态
  const [timer, setTimer] = useState<TimerState>({
    whiteTime: 180 * 60, // 3小时 = 180分钟 = 180*60秒
    blackTime: 180 * 60,
    whiteIncrement: 60, // 1分钟
    blackIncrement: 60,
    isRunning: false,
    currentPlayer: 'w',
    lastMoveTime: Date.now(),
  });

  // 移动历史（包含时间信息）
  const [moveHistory, setMoveHistory] = useState<MoveWithTime[]>([]);

  // PGN保存对话框状态
  const [showPgnDialog, setShowPgnDialog] = useState(false);
  const [pgnHeaders, setPgnHeaders] = useState<PgnHeaders>({
    Event: new Date().toLocaleString('en-US', { 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }),
    Date: new Date().toISOString().split('T')[0].replace(/-/g, '.'),
    Round: '1',
    White: 'P1',
    Black: 'P2',
    Result: '*',
    TimeControl: '180+1',
    WhiteFideId: '12345678',
    BlackFideId: '87654321',
    WhiteTitle: 'CM',
    BlackTitle: 'CM',
    WhiteElo: '2000',
    BlackElo: '2000',
    WhiteTeam: 'CHN',
    BlackTeam: 'CHN',
    WhiteClock: '03:00:00',
    BlackClock: '03:00:00',
    Variant: 'Standard',
  });

  // 认输/和棋对话框状态（已移除，现在直接调用 endGame）
  const [selectedOpening, setSelectedOpening] = useState(OPENING_POSITIONS[0]);
  const [customFen, setCustomFen] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [analysis, setAnalysis] = useState<StockfishAnalysis | null>(null);
  const [lastFenBeforeMove, setLastFenBeforeMove] = useState<string | null>(null);
  const [lastMoveEval, setLastMoveEval] = useState<any | null>(null);
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
  
  // 已移除：加载日志现在由 SaeComboLoader 组件统一管理

  // 新增：用于强制更新组件（由于 Chess 实例是可变的）
  const [, setDummy] = useState(0);

  // 新增：Circuit Trace状态
  const [isTracing] = useState(false);

  // 新增：MCTS 搜索设置
  const [useSearch, setUseSearch] = useState(false);
  const [searchParams, setSearchParams] = useState({
    max_playouts: 100,
    target_minibatch_size: 8,
    cpuct: 3.0,
    max_depth: 10,
    // 低Q值探索增强参数（用于发现弃后连杀等隐藏走法）
    low_q_exploration_enabled: false,
    low_q_threshold: 0.3,
    low_q_exploration_bonus: 0.1,
    low_q_visit_threshold: 5,
  });
  const [lastSearchInfo, setLastSearchInfo] = useState<{
    total_playouts: number;
    max_depth_reached: number;
    max_depth_limit: number;
  } | null>(null);
  const [saveMctsTrace, setSaveMctsTrace] = useState(false);

  // 移除：不在初始化或任何自动时机触发模型走棋，改为仅按按钮触发

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

  // 格式化时间为MM:SS或HH:MM:SS格式
  const formatTime = (seconds: number): string => {
    const totalSeconds = Math.floor(seconds); // 只取整数秒
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const secs = totalSeconds % 60;
    
    if (hours > 0) {
      return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    } else {
      return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
  };


  // 计时器更新
  useEffect(() => {
    if (!timer.isRunning || gameState.isGameOver) {
      return;
    }

    const interval = setInterval(() => {
      setTimer(prev => {
        const now = Date.now();
        const elapsed = (now - prev.lastMoveTime) / 1000;
        
        if (prev.currentPlayer === 'w') {
          const newWhiteTime = Math.max(0, prev.whiteTime - elapsed);
          if (newWhiteTime <= 0) {
            // 白方超时
            endGame('resignation', 'Black');
            return { ...prev, whiteTime: 0, isRunning: false };
          }
          return { ...prev, whiteTime: newWhiteTime, lastMoveTime: now };
        } else {
          const newBlackTime = Math.max(0, prev.blackTime - elapsed);
          if (newBlackTime <= 0) {
            // 黑方超时
            endGame('resignation', 'White');
            return { ...prev, blackTime: 0, isRunning: false };
          }
          return { ...prev, blackTime: newBlackTime, lastMoveTime: now };
        }
      });
    }, 100);

    return () => clearInterval(interval);
  }, [timer.isRunning, gameState.isGameOver]);

  // 结束游戏
  const endGame = useCallback((reason: 'checkmate' | 'resignation' | 'draw', winner?: string | null) => {
    setGameState(prev => ({
      ...prev,
      isGameOver: true,
      winner: winner || null,
      gameEndReason: reason,
    }));
    
    setTimer(prev => ({ ...prev, isRunning: false }));
    
    // 更新PGN头信息
    let result = '*';
    if (reason === 'checkmate') {
      result = winner === 'White' ? '1-0' : '0-1';
    } else if (reason === 'resignation') {
      result = winner === 'White' ? '1-0' : '0-1';
    } else if (reason === 'draw') {
      result = '1/2-1/2';
    }
    
    setPgnHeaders(prev => ({
      ...prev,
      Result: result,
      WhiteClock: formatTime(timer.whiteTime),
      BlackClock: formatTime(timer.blackTime),
    }));
    
    // 显示PGN保存对话框
    setTimeout(() => {
      setShowPgnDialog(true);
    }, 1000);
  }, [timer.whiteTime, timer.blackTime, formatTime]);

  // 更新游戏状态
  const updateGameState = useCallback((newGame: Chess, moveNotation?: string) => {
    const historyVerbose = newGame.history({ verbose: true });
    const moves = historyVerbose.map(m => m.from + m.to + (m.promotion ? m.promotion : ''));
    const isGameOver = newGame.isGameOver();
    let winner = null;
    let gameEndReason: 'checkmate' | 'resignation' | 'draw' | null = null;
    
    if (isGameOver) {
      if (newGame.isCheckmate()) {
        winner = newGame.turn() === 'w' ? 'Black' : 'White';
        gameEndReason = 'checkmate';
      } else if (newGame.isDraw()) {
        winner = 'Draw';
        gameEndReason = 'draw';
      }
    }

    // 更新计时器
    setTimer(prev => {
      const now = Date.now();
      const elapsed = (now - prev.lastMoveTime) / 1000;
      
      let newWhiteTime = prev.whiteTime;
      let newBlackTime = prev.blackTime;
      
      if (prev.currentPlayer === 'w') {
        newWhiteTime = Math.max(0, prev.whiteTime - elapsed);
        if (newWhiteTime > 0) {
          newWhiteTime += prev.whiteIncrement;
        }
      } else {
        newBlackTime = Math.max(0, prev.blackTime - elapsed);
        if (newBlackTime > 0) {
          newBlackTime += prev.blackIncrement;
        }
      }
      
      return {
        ...prev,
        whiteTime: newWhiteTime,
        blackTime: newBlackTime,
        currentPlayer: newGame.turn(),
        lastMoveTime: now,
      };
    });

    // 记录移动历史
    if (moveNotation && !isGameOver) {
      const now = Date.now();
      const timeUsed = timer.currentPlayer === 'w' ? 
        timer.whiteTime - Math.max(0, timer.whiteTime - (now - timer.lastMoveTime) / 1000) :
        timer.blackTime - Math.max(0, timer.blackTime - (now - timer.lastMoveTime) / 1000);
      
      const clockFormat = formatTime(timer.currentPlayer === 'w' ? timer.whiteTime : timer.blackTime);
      
      setMoveHistory(prev => [...prev, {
        move: moveNotation,
        time: timeUsed,
        clock: clockFormat,
      }]);
    }

    setGameState({
      fen: newGame.fen(),
      moves,
      isGameOver,
      winner,
      isPlayerTurn: newGame.turn() === 'w',
      gameEndReason,
    });

    // 如果游戏结束，调用endGame
    if (isGameOver && gameEndReason) {
      endGame(gameEndReason, winner);
    }

    // 通知父组件游戏状态更新
    onGameStateUpdate?.(newGame.fen(), moves);
  }, [onGameStateUpdate, timer, formatTime, endGame]);

  // 开始新游戏
  const startNewGame = useCallback((fen?: string) => {
    const newGame = new Chess(fen || selectedOpening.fen);
    setGame(newGame);
    
    // 重置计时器
    setTimer(prev => ({
      ...prev,
      whiteTime: 180 * 60,
      blackTime: 180 * 60,
      isRunning: false,
      currentPlayer: 'w',
      lastMoveTime: Date.now(),
    }));
    
    // 重置移动历史
    setMoveHistory([]);
    
    updateGameState(newGame);
    setAnalysis(null);
    setIsAutoPlay(false);
    setShowPgnDialog(false);
    
    if (autoPlayInterval) {
      clearInterval(autoPlayInterval);
      setAutoPlayInterval(null);
    }
  }, [selectedOpening.fen, updateGameState, autoPlayInterval]);

  // 使用自定义FEN开始游戏
  const startWithCustomFen = useCallback(() => {
    if (customFen.trim()) {
    try {
      // 直接用自定义fen新建Chess实例
      const newGame = new Chess(customFen.trim());
      setGame(newGame);
      // 直接同步设置当前游戏state
      setGameState({
        fen: newGame.fen(),
        moves: [],
        isGameOver: newGame.isGameOver(),
        winner: null,
        isPlayerTurn: newGame.turn() === 'w',
        gameEndReason: null,
      });
      // 同步通知外部（如 Circuit Tracing）当前FEN与空历史，用于立刻更新“分析FEN(移动前)”与“当前FEN”
      onGameStateUpdate?.(newGame.fen(), []);
      // 额外：写入本地缓存，供 Circuit Tracing 直接读取
      try {
        localStorage.setItem('circuit_game_fen', newGame.fen());
      } catch {}
      // 重置计时/分析状态
      setTimer(prev => ({ ...prev, whiteTime: 180 * 60, blackTime: 180 * 60, isRunning: false, currentPlayer: 'w', lastMoveTime: Date.now() }));
      setMoveHistory([]);
      setAnalysis(null); // 清空旧分析
      setIsAutoPlay(false);
      setShowPgnDialog(false);
      if (autoPlayInterval) { clearInterval(autoPlayInterval); setAutoPlayInterval(null); }
      // 刷新分析（如有需要，可注释掉）
      // getStockfishAnalysis(newGame.fen()); // 如果你希望用“分析模式”自动刷新分析结果，请取消注释
    } catch (error) {
      alert('无效的FEN字符串');
    }
    }
  }, [customFen, autoPlayInterval]);

  // 获取Stockfish分析（固定使用BT4模型）
  const getStockfishAnalysis = useCallback(async (fen: string) => {
    try {
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/analyze/board`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ fen }),
      });

      if (response.ok) {
        const data = await response.json();
        setAnalysis({
          bestMove: 'N/A', // 模型分析不提供最佳移动
          evaluation: data.evaluation[0] - data.evaluation[2], // 胜率差值
          depth: 0, // 模型分析没有深度概念
          wdl: {
            winProb: data.evaluation[0],
            drawProb: data.evaluation[1],
            lossProb: data.evaluation[2],
          },
        });
        return data;
      } else {
        console.error('模型分析失败:', response.status, response.statusText);
        const errorText = await response.text();
        console.error('错误详情:', errorText);
      }
    } catch (error) {
      console.error('获取模型分析失败:', error);
    }
    return null;
  }, []);

  // 获取模型建议的移动（固定使用BT4模型，支持可选搜索）
  const getModelMove = useCallback(async (fen: string): Promise<ModelMoveResponse | null> => {
    try {
      // 根据是否启用搜索选择不同的 API 端点
      const endpoint = useSearch 
        ? `${import.meta.env.VITE_BACKEND_URL}/play_game_with_search`
        : `${import.meta.env.VITE_BACKEND_URL}/play_game`;
      
      const requestBody = useSearch 
        ? { 
            fen, 
            ...searchParams, 
            save_trace: saveMctsTrace,
            trace_max_edges: saveMctsTrace ? 0 : 1000  // 0 表示保存完整搜索树，不限制边数
          }
        : { fen };
      
      console.log(`🎯 请求模型移动: ${useSearch ? 'MCTS搜索' : '直接推理'}, playouts=${searchParams.max_playouts}`);
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (response.ok) {
        const data = await response.json();
        
        // 如果使用了搜索，保存搜索信息
        if (useSearch && data.search_info) {
          setLastSearchInfo({
            total_playouts: data.search_info.total_playouts,
            max_depth_reached: data.search_info.max_depth_reached,
            max_depth_limit: data.search_info.max_depth_limit,
          });
          console.log(`✅ MCTS搜索完成: playouts=${data.search_info.total_playouts}, depth=${data.search_info.max_depth_reached}`);
        } else {
          setLastSearchInfo(null);
        }
        
        return data as ModelMoveResponse;
      } else {
        console.error('API调用失败:', response.status, response.statusText);
        const errorText = await response.text();
        console.error('错误详情:', errorText);
      }
    } catch (error) {
      console.error('获取模型移动失败:', error);
    }
    return null;
  }, [useSearch, searchParams, saveMctsTrace]);

  const downloadTraceFile = useCallback(async (filename: string) => {
    try {
      const url = `${import.meta.env.VITE_BACKEND_URL}/search_trace/files/${encodeURIComponent(filename)}`;
      const response = await fetch(url);
      if (!response.ok) {
        const errorText = await response.text();
        console.error('下载MCTS搜索文件失败:', errorText);
        return;
      }
      const blob = await response.blob();
      const downloadUrl = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = downloadUrl;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      document.body.removeChild(anchor);
      URL.revokeObjectURL(downloadUrl);
    } catch (error) {
      console.error('下载MCTS搜索文件失败:', error);
    }
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

  // 执行移动
  const makeMove = useCallback((move: any) => {
    try {
      const prevFenBefore = game.fen();
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

      // 记录本步前的FEN，供评测使用
      setLastFenBeforeMove(prevFenBefore);

      // 使用updateGameState来更新状态（包含计时器更新）
      updateGameState(replaced, movesUci[movesUci.length - 1]);

      setDummy(prev => prev + 1); // 强制触发刷新
      return true;
    } catch (error) {
      console.error('移动失败:', error);
    }
    return false;
  }, [game, toChessJsMove, updateGameState]);

  // 处理玩家移动：仅在人类回合允许
  const handlePlayerMove = useCallback((move: string) => {
    if (!isHumanTurn) {
      return false;
    }

    // ChessBoard组件现在已经正确处理了翻转逻辑，直接使用返回的UCI
    const uci = move;

    const ok = makeMove(uci);
    if (ok) {
      if (gameMode === 'analysis') {
        setTimeout(() => {
          getStockfishAnalysis(game.fen());
        }, 100);
      }
      return true;
    }
    return false;
  }, [isHumanTurn, makeMove, gameMode, getStockfishAnalysis, game]);

  // 处理模型移动：仅在模型回合触发
  const handleModelMove = useCallback(async () => {
    if (isModelTurn && !isLoading && !isTracing) {
      setIsLoading(true);
      try {
        const moveResponse = await getModelMove(game.fen());
        const modelMove = moveResponse?.move;
        if (!modelMove) {
          console.warn('模型未返回走法或返回为空');
          alert('模型未返回走法，请检查后端或当前局面');
          return;
        }
        if (modelMove && makeMove(modelMove)) {
          if (useSearch && saveMctsTrace && moveResponse?.trace_filename) {
            await downloadTraceFile(moveResponse.trace_filename);
          }
          // 获取新局面的分析
          if (gameMode === 'analysis') {
            setTimeout(() => {
              getStockfishAnalysis(game.fen());
            }, 100);
          }
        } else {
          alert('模型给出的走法无效或不合法');
        }
      } finally {
        setIsLoading(false);
      }
    }
  }, [isModelTurn, isLoading, isTracing, getModelMove, game, makeMove, gameMode, getStockfishAnalysis, useSearch, saveMctsTrace, downloadTraceFile]);

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
        gameEndReason: null,
      });

      // 通知父组件游戏状态更新
      onGameStateUpdate?.(nextFen, movesUci);

      setDummy(prev => prev + 1);
      if (gameMode === 'analysis') {
        setTimeout(() => { getStockfishAnalysis(nextFen); }, 100);
      }
    } catch (error) {
      console.error('撤销失败:', error);
    }
  }, [game, gameMode, getStockfishAnalysis, onGameStateUpdate]);

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
      const prevFenBefore = game.fen();
      const newGame = new Chess(game.fen());
      const parsed = toChessJsMove(manualMove.trim());
      const result = newGame.move(parsed as any);
      
      if (result) {
        setGame(newGame);
        updateGameState(newGame, manualMove.trim());
        setLastFenBeforeMove(prevFenBefore);
        setManualMove('');
        setMoveError('');
        
        if (gameMode === 'analysis') {
          setTimeout(() => {
            getStockfishAnalysis(newGame.fen());
          }, 100);
        }
      } else {
        setMoveError('无效的走法');
      }
    } catch (error) {
      setMoveError('无效的走法格式');
    }
  }, [manualMove, isHumanTurn, gameState.isGameOver, game, updateGameState, gameMode, getStockfishAnalysis, toChessJsMove]);

  // 调用后端进行走法评测（基于上一步之前的FEN与上一步UCI）
  const evaluateLastMove = useCallback(async () => {
    try {
      const lastMove = gameState.moves[gameState.moves.length - 1];
      if (!lastMove || !lastFenBeforeMove) {
        alert('没有可评测的上一步或缺少前一FEN');
        return;
      }
      const resp = await fetch(`${import.meta.env.VITE_BACKEND_URL}/evaluate_move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen: lastFenBeforeMove, move: lastMove, time_limit: 0.2 }),
      });
      if (!resp.ok) {
        const t = await resp.text();
        throw new Error(t || '评测失败');
      }
      const data = await resp.json();
      setLastMoveEval(data);
    } catch (e) {
      console.error('评测失败:', e);
      alert('评测失败，请查看控制台日志');
    }
  }, [gameState.moves, lastFenBeforeMove]);

  // 生成PGN字符串
  const generatePgn = useCallback(() => {
    let pgn = '';
    
    // 添加头部信息
    Object.entries(pgnHeaders).forEach(([key, value]) => {
      pgn += `[${key} "${value}"]\n`;
    });
    
    pgn += '\n';
    
    // 添加移动历史
    const moves = game.history();
    for (let i = 0; i < moves.length; i++) {
      const moveNum = Math.floor(i / 2) + 1;
      const isWhiteMove = i % 2 === 0;
      
      if (isWhiteMove) {
        pgn += `${moveNum}. `;
      }
      
      pgn += `${moves[i]} `;
      
      // 添加时间信息
      if (moveHistory[i]) {
        const timeUsed = Math.round(moveHistory[i].time);
        const minutes = Math.floor(timeUsed / 60);
        const seconds = timeUsed % 60;
        pgn += `{[%clk 0:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}]} `;
      }
      
      if (i % 2 === 1) {
        pgn += '\n';
      }
    }
    
    pgn += ` ${pgnHeaders.Result}`;
    
    return pgn;
  }, [pgnHeaders, game, moveHistory]);

  // 保存PGN文件
  const savePgnFile = useCallback(() => {
    const pgn = generatePgn();
    const blob = new Blob([pgn], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${pgnHeaders.Event.replace(/[^a-zA-Z0-9]/g, '_')}_${pgnHeaders.Date.replace(/\./g, '-')}.pgn`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [generatePgn, pgnHeaders]);

  // 处理认输
  const handleResign = useCallback((player: 'White' | 'Black') => {
    endGame('resignation', player === 'White' ? 'Black' : 'White');
  }, [endGame]);

  // 处理和棋
  const handleDraw = useCallback(() => {
    endGame('draw');
  }, [endGame]);

  // 开始计时器
  const startTimer = useCallback(() => {
    setTimer(prev => ({ ...prev, isRunning: true, lastMoveTime: Date.now() }));
  }, []);


  // 已移除：加载日志和预加载逻辑现在由 SaeComboLoader 组件统一管理

  // 已移除自动加载逻辑：现在通过页面顶部的 SaeComboLoader 组件手动加载

  // 不自动触发模型走棋，用户需点击"让模型走棋"按钮

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* 全局 BT4 SAE 组合选择（LoRSA / Transcoder） */}
      <SaeComboLoader />

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
              {/* 计时器显示 */}
              <div className="mb-4 flex justify-center gap-8">
                <div className={`text-center p-3 rounded-lg ${game.turn() === 'w' ? 'bg-blue-100 border-2 border-blue-500' : 'bg-gray-100'}`}>
                  <div className="text-sm font-medium text-gray-600">白方</div>
                  <div className={`text-2xl font-mono font-bold ${timer.whiteTime < 60 ? 'text-red-600' : 'text-gray-900'}`}>
                    {formatTime(timer.whiteTime)}
                  </div>
                </div>
                <div className="flex items-center">
                  <Button
                    onClick={timer.isRunning ? () => setTimer(prev => ({ ...prev, isRunning: false })) : startTimer}
                    variant={timer.isRunning ? "destructive" : "default"}
                    size="sm"
                  >
                    {timer.isRunning ? '暂停' : '开始'}
                  </Button>
                </div>
                <div className={`text-center p-3 rounded-lg ${game.turn() === 'b' ? 'bg-blue-100 border-2 border-blue-500' : 'bg-gray-100'}`}>
                  <div className="text-sm font-medium text-gray-600">黑方</div>
                  <div className={`text-2xl font-mono font-bold ${timer.blackTime < 60 ? 'text-red-600' : 'text-gray-900'}`}>
                    {formatTime(timer.blackTime)}
                  </div>
                </div>
              </div>

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
                  showSelfPlay={false}
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
              
              {/* 加载日志已迁移到页面顶部的 SaeComboLoader 组件 */}
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

          {/* MCTS 搜索设置 */}
          <Card>
            <CardHeader>
              <CardTitle>搜索设置</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">启用 MCTS 搜索</label>
                <Switch checked={useSearch} onCheckedChange={setUseSearch} />
              </div>
              
              {useSearch && (
                <div className="space-y-3 pt-2 border-t">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium">保存MCTS搜索JSON</label>
                    <Switch checked={saveMctsTrace} onCheckedChange={setSaveMctsTrace} />
                  </div>
                  <p className="text-xs text-gray-500">
                    启用后，每次搜索完成并落子时会自动下载对应FEN的搜索trace（默认文件名包含当前FEN）。
                  </p>
                  <div>
                    <label className="text-sm font-medium">最大模拟次数</label>
                    <Input
                      type="number"
                      value={searchParams.max_playouts}
                      onChange={(e) => setSearchParams(prev => ({
                        ...prev,
                        max_playouts: parseInt(e.target.value) || 100,
                      }))}
                      min={10}
                      max={10000}
                      className="mt-1"
                    />
                  </div>
                  
                  <div>
                    <label className="text-sm font-medium">最大搜索深度</label>
                    <Input
                      type="number"
                      value={searchParams.max_depth}
                      onChange={(e) => setSearchParams(prev => ({
                        ...prev,
                        max_depth: parseInt(e.target.value) || 10,
                      }))}
                      min={1}
                      max={50}
                      className="mt-1"
                    />
                  </div>
                  
                  <div>
                    <label className="text-sm font-medium">UCT 探索系数 (cpuct)</label>
                    <Input
                      type="number"
                      step="0.1"
                      value={searchParams.cpuct}
                      onChange={(e) => setSearchParams(prev => ({
                        ...prev,
                        cpuct: parseFloat(e.target.value) || 3.0,
                      }))}
                      min={0.1}
                      max={10}
                      className="mt-1"
                    />
                  </div>
                  
                  <div>
                    <label className="text-sm font-medium">批处理大小</label>
                    <Input
                      type="number"
                      value={searchParams.target_minibatch_size}
                      onChange={(e) => setSearchParams(prev => ({
                        ...prev,
                        target_minibatch_size: parseInt(e.target.value) || 8,
                      }))}
                      min={1}
                      max={64}
                      className="mt-1"
                    />
                  </div>
                  
                  {/* 低Q值探索增强参数 */}
                  <div className="border-t pt-3 mt-3">
                    <div className="flex items-center gap-2 mb-2">
                      <input
                        type="checkbox"
                        id="low_q_exploration_enabled"
                        checked={searchParams.low_q_exploration_enabled}
                        onChange={(e) => setSearchParams(prev => ({
                          ...prev,
                          low_q_exploration_enabled: e.target.checked,
                        }))}
                        className="w-4 h-4"
                      />
                      <label htmlFor="low_q_exploration_enabled" className="text-sm font-medium">
                        启用低Q值探索增强（用于发现弃后连杀等隐藏走法）
                      </label>
                    </div>
                    
                    {searchParams.low_q_exploration_enabled && (
                      <div className="space-y-3 ml-6 mt-3 bg-blue-50 p-3 rounded">
                        <p className="text-xs text-gray-600 mb-2">
                          对Q值低于阈值且访问次数较少的走法给予额外探索奖励，有助于发现模型先验评估不高但实际可能是好走法的情况（如弃后连杀）。
                        </p>
                        <div className="space-y-2">
                          <div>
                            <label className="text-xs font-medium text-gray-700">Q值阈值</label>
                            <Input
                              type="number"
                              step="0.1"
                              value={searchParams.low_q_threshold}
                              onChange={(e) => setSearchParams(prev => ({
                                ...prev,
                                low_q_threshold: parseFloat(e.target.value) || 0.3,
                              }))}
                              min={-1}
                              max={1}
                              className="mt-1 text-xs"
                            />
                            <p className="text-xs text-gray-500 mt-1">
                              低于此Q值的走法会被增强探索（默认0.3，可为负数）
                            </p>
                          </div>
                          <div>
                            <label className="text-xs font-medium text-gray-700">探索奖励</label>
                            <Input
                              type="number"
                              step="0.01"
                              value={searchParams.low_q_exploration_bonus}
                              onChange={(e) => setSearchParams(prev => ({
                                ...prev,
                                low_q_exploration_bonus: parseFloat(e.target.value) || 0.1,
                              }))}
                              min={0}
                              max={1}
                              className="mt-1 text-xs"
                            />
                            <p className="text-xs text-gray-500 mt-1">
                              奖励的基础值，越大则对低Q值走法的探索越积极（默认0.1）
                            </p>
                          </div>
                          <div>
                            <label className="text-xs font-medium text-gray-700">访问次数阈值</label>
                            <Input
                              type="number"
                              value={searchParams.low_q_visit_threshold}
                              onChange={(e) => setSearchParams(prev => ({
                                ...prev,
                                low_q_visit_threshold: parseInt(e.target.value) || 5,
                              }))}
                              min={1}
                              max={50}
                              className="mt-1 text-xs"
                            />
                            <p className="text-xs text-gray-500 mt-1">
                              访问次数低于此值的走法才会获得奖励（默认5）
                            </p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                  
                  {/* 显示上次搜索信息 */}
                  {lastSearchInfo && (
                    <div className="bg-gray-50 p-2 rounded text-xs space-y-1">
                      <div><strong>上次搜索:</strong></div>
                      <div>总模拟次数: {lastSearchInfo.total_playouts}</div>
                      <div>达到深度: {lastSearchInfo.max_depth_reached}</div>
                    </div>
                  )}
                </div>
              )}
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

              {/* 模型移动控制：移动到手动走棋下方 */}
              <div className="border-t pt-4">
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
                
                {gameMode === 'analysis' && (
                  <Button
                    onClick={getCurrentAnalysis}
                    variant="outline"
                    className="w-full mt-2"
                  >
                    获取分析
                  </Button>
                )}

                {/* 评测上一步 */}
                <Button
                  onClick={evaluateLastMove}
                  variant="outline"
                  className="w-full mt-2"
                  disabled={!lastFenBeforeMove || gameState.moves.length === 0}
                >
                  评测上一步
                </Button>
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

          {/* 上一步评测结果展示 */}
          {lastMoveEval && (
            <Card>
              <CardHeader>
                <CardTitle>上一步评测结果</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <div><strong>我的走法:</strong> {lastMoveEval.my_move}</div>
                <div><strong>最佳走法:</strong> {lastMoveEval.best_move || 'N/A'}</div>
                <div><strong>评分(0-100):</strong> {typeof lastMoveEval.score_100 === 'number' ? lastMoveEval.score_100.toFixed(1) : 'N/A'}</div>
                <div><strong>CP损失:</strong> {lastMoveEval.cp_loss !== null && lastMoveEval.cp_loss !== undefined ? lastMoveEval.cp_loss.toFixed(1) : 'N/A'}</div>
                <div><strong>根局面CP:</strong> {lastMoveEval.root_cp !== null && lastMoveEval.root_cp !== undefined ? lastMoveEval.root_cp.toFixed(1) : 'N/A'}</div>
                <div><strong>最佳后CP:</strong> {lastMoveEval.best_cp !== null && lastMoveEval.best_cp !== undefined ? lastMoveEval.best_cp.toFixed(1) : 'N/A'}</div>
                <div><strong>我的后CP:</strong> {lastMoveEval.my_cp !== null && lastMoveEval.my_cp !== undefined ? lastMoveEval.my_cp.toFixed(1) : 'N/A'}</div>
                {lastMoveEval.root_wdl && (
                  <div>
                    <strong>根局面WDL:</strong> 白胜 {(lastMoveEval.root_wdl[0]*100).toFixed(1)}% / 和棋 {(lastMoveEval.root_wdl[1]*100).toFixed(1)}% / 黑胜 {(lastMoveEval.root_wdl[2]*100).toFixed(1)}%
                  </div>
                )}
                {lastMoveEval.best_wdl && (
                  <div>
                    <strong>最佳后WDL:</strong> 白胜 {(lastMoveEval.best_wdl[0]*100).toFixed(1)}% / 和棋 {(lastMoveEval.best_wdl[1]*100).toFixed(1)}% / 黑胜 {(lastMoveEval.best_wdl[2]*100).toFixed(1)}%
                  </div>
                )}
                {lastMoveEval.my_wdl && (
                  <div>
                    <strong>我的后WDL:</strong> 白胜 {(lastMoveEval.my_wdl[0]*100).toFixed(1)}% / 和棋 {(lastMoveEval.my_wdl[1]*100).toFixed(1)}% / 黑胜 {(lastMoveEval.my_wdl[2]*100).toFixed(1)}%
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* 游戏结束控制 */}
          {!gameState.isGameOver && (
            <Card>
              <CardHeader>
                <CardTitle>结束游戏</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-2">
                  <Button
                    onClick={() => handleResign('White')}
                    variant="outline"
                    size="sm"
                    className="text-red-600 hover:text-red-700"
                  >
                    白方认输
                  </Button>
                  <Button
                    onClick={() => handleResign('Black')}
                    variant="outline"
                    size="sm"
                    className="text-red-600 hover:text-red-700"
                  >
                    黑方认输
                  </Button>
                </div>
                <Button
                  onClick={handleDraw}
                  variant="outline"
                  size="sm"
                  className="w-full text-blue-600 hover:text-blue-700"
                >
                  提议和棋
                </Button>
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
              {lastFenBeforeMove && (
                <div className="text-sm">
                  <strong>上一步前FEN:</strong>
                  <div className="font-mono text-xs bg-gray-100 p-2 rounded mt-1 break-all">
                    {lastFenBeforeMove}
                  </div>
                </div>
              )}
              <div className="text-sm">
                <strong>移动数:</strong> {gameState.moves.length}
              </div>
              <div className="text-sm">
                <strong>当前回合:</strong> {game.turn() === 'w' ? '白方' : '黑方'}
              </div>
              {gameState.isGameOver && (
                <div className="text-sm">
                  <strong>游戏结束原因:</strong> {
                    gameState.gameEndReason === 'checkmate' ? '将死' :
                    gameState.gameEndReason === 'resignation' ? '认输' :
                    gameState.gameEndReason === 'draw' ? '和棋' : '未知'
                  }
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* PGN保存对话框 */}
      <Dialog open={showPgnDialog} onOpenChange={setShowPgnDialog}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Download className="w-5 h-5" />
              保存对局为PGN文件
            </DialogTitle>
          </DialogHeader>
          
          <div className="space-y-4">
            <div className="text-sm text-gray-600">
              游戏已结束！请确认以下信息后保存对局。
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="event">Event</Label>
                <Input
                  id="event"
                  value={pgnHeaders.Event}
                  onChange={(e) => setPgnHeaders(prev => ({ ...prev, Event: e.target.value }))}
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="date">Date</Label>
                <Input
                  id="date"
                  value={pgnHeaders.Date}
                  onChange={(e) => setPgnHeaders(prev => ({ ...prev, Date: e.target.value }))}
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="round">Round</Label>
                <Input
                  id="round"
                  value={pgnHeaders.Round}
                  onChange={(e) => setPgnHeaders(prev => ({ ...prev, Round: e.target.value }))}
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="white">White</Label>
                <Input
                  id="white"
                  value={pgnHeaders.White}
                  onChange={(e) => setPgnHeaders(prev => ({ ...prev, White: e.target.value }))}
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="black">Black</Label>
                <Input
                  id="black"
                  value={pgnHeaders.Black}
                  onChange={(e) => setPgnHeaders(prev => ({ ...prev, Black: e.target.value }))}
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="result">Result</Label>
                <Input
                  id="result"
                  value={pgnHeaders.Result}
                  readOnly
                  className="bg-gray-100"
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="timeControl">TimeControl</Label>
                <Input
                  id="timeControl"
                  value={pgnHeaders.TimeControl}
                  readOnly
                  className="bg-gray-100"
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="whiteElo">WhiteElo</Label>
                <Input
                  id="whiteElo"
                  value={pgnHeaders.WhiteElo}
                  onChange={(e) => setPgnHeaders(prev => ({ ...prev, WhiteElo: e.target.value }))}
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="blackElo">BlackElo</Label>
                <Input
                  id="blackElo"
                  value={pgnHeaders.BlackElo}
                  onChange={(e) => setPgnHeaders(prev => ({ ...prev, BlackElo: e.target.value }))}
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="whiteTitle">WhiteTitle</Label>
                <Input
                  id="whiteTitle"
                  value={pgnHeaders.WhiteTitle}
                  onChange={(e) => setPgnHeaders(prev => ({ ...prev, WhiteTitle: e.target.value }))}
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="blackTitle">BlackTitle</Label>
                <Input
                  id="blackTitle"
                  value={pgnHeaders.BlackTitle}
                  onChange={(e) => setPgnHeaders(prev => ({ ...prev, BlackTitle: e.target.value }))}
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="whiteTeam">WhiteTeam</Label>
                <Input
                  id="whiteTeam"
                  value={pgnHeaders.WhiteTeam}
                  onChange={(e) => setPgnHeaders(prev => ({ ...prev, WhiteTeam: e.target.value }))}
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="blackTeam">BlackTeam</Label>
                <Input
                  id="blackTeam"
                  value={pgnHeaders.BlackTeam}
                  onChange={(e) => setPgnHeaders(prev => ({ ...prev, BlackTeam: e.target.value }))}
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="whiteClock">WhiteClock</Label>
                <Input
                  id="whiteClock"
                  value={pgnHeaders.WhiteClock}
                  readOnly
                  className="bg-gray-100"
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="blackClock">BlackClock</Label>
                <Input
                  id="blackClock"
                  value={pgnHeaders.BlackClock}
                  readOnly
                  className="bg-gray-100"
                />
              </div>
            </div>
            
            {/* 显示PGN预览 */}
            <div className="space-y-2">
              <Label>PGN预览</Label>
              <Textarea
                value={generatePgn()}
                readOnly
                rows={8}
                className="font-mono text-xs"
              />
            </div>
          </div>
          
          <DialogFooter className="flex gap-2">
            <Button
              variant="outline"
              onClick={() => setShowPgnDialog(false)}
            >
              取消
            </Button>
            <Button
              onClick={() => {
                savePgnFile();
                setShowPgnDialog(false);
              }}
              className="flex items-center gap-2"
            >
              <Download className="w-4 h-4" />
              保存PGN文件
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      
    </div>
  );
};