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
  whiteTime: number; // remaining time (seconds)
  blackTime: number;
  whiteIncrement: number; // per-move increment (seconds)
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
  { name: "Starting Position", fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" },
  { name: "Italian Opening", fen: "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4" },
  { name: "Spanish Opening", fen: "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3" },
  { name: "Sicilian Defense", fen: "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2" },
  { name: "French Defense", fen: "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2" },
  { name: "Scandinavian Defense", fen: "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2" },
  { name: "Caro-Kann Defense", fen: "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2" },
  { name: "Indian Defense", fen: "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3" },
  { name: "English Opening", fen: "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1" },
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

  const [timer, setTimer] = useState<TimerState>({
    whiteTime: 180 * 60, // 3h = 180min = 180*60s
    blackTime: 180 * 60,
    whiteIncrement: 60, // 1min = 60s
    blackIncrement: 60,
    isRunning: false,
    currentPlayer: 'w',
    lastMoveTime: Date.now(),
  });

  const [moveHistory, setMoveHistory] = useState<MoveWithTime[]>([]);

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

  const [selectedOpening, setSelectedOpening] = useState(OPENING_POSITIONS[0]);
  const [customFen, setCustomFen] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [analysis, setAnalysis] = useState<StockfishAnalysis | null>(null);
  const [lastFenBeforeMove, setLastFenBeforeMove] = useState<string | null>(null);
  const [lastMoveEval, setLastMoveEval] = useState<any | null>(null);
  const [gameMode, setGameMode] = useState<'player-vs-model' | 'analysis'>('player-vs-model');
  const [isAutoPlay, setIsAutoPlay] = useState(false);
  const [autoPlayInterval, setAutoPlayInterval] = useState<NodeJS.Timeout | null>(null);
  
  const [autoFlipWhenBlack, setAutoFlipWhenBlack] = useState<boolean>(false);
  
  const [matchMode, setMatchMode] = useState<'human-human' | 'human-model' | 'model-model'>('human-model');
  const [humanPlays, setHumanPlays] = useState<'w' | 'b'>('w');

  const [manualMove, setManualMove] = useState('');
  const [moveError, setMoveError] = useState('');
  
  const [, setDummy] = useState(0);

  const [isTracing] = useState(false);

  const [useSearch, setUseSearch] = useState(false);
  const [searchParams, setSearchParams] = useState({
    max_playouts: 100,
    target_minibatch_size: 8,
    cpuct: 3.0,
    max_depth: 10,
    // a lower q value exploration bonus is used to find hidden good moves
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

  const isWhiteToMove = game.turn() === 'w';
  const isHumanTurn = (
    (matchMode === 'human-human') ||
    (matchMode === 'human-model' && ((humanPlays === 'w' && isWhiteToMove) || (humanPlays === 'b' && !isWhiteToMove)))
  ) && !gameState.isGameOver;
  const isModelTurn = (
    (matchMode === 'model-model') ||
    (matchMode === 'human-model' && !isHumanTurn)
  ) && !gameState.isGameOver;

  const formatTime = (seconds: number): string => {
    const totalSeconds = Math.floor(seconds);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const secs = totalSeconds % 60;
    
    if (hours > 0) {
      return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    } else {
      return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
  };


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
            endGame('resignation', 'Black');
            return { ...prev, whiteTime: 0, isRunning: false };
          }
          return { ...prev, whiteTime: newWhiteTime, lastMoveTime: now };
        } else {
          const newBlackTime = Math.max(0, prev.blackTime - elapsed);
          if (newBlackTime <= 0) {
            endGame('resignation', 'White');
            return { ...prev, blackTime: 0, isRunning: false };
          }
          return { ...prev, blackTime: newBlackTime, lastMoveTime: now };
        }
      });
    }, 100);

    return () => clearInterval(interval);
  }, [timer.isRunning, gameState.isGameOver]);

  const endGame = useCallback((reason: 'checkmate' | 'resignation' | 'draw', winner?: string | null) => {
    setGameState(prev => ({
      ...prev,
      isGameOver: true,
      winner: winner || null,
      gameEndReason: reason,
    }));
    
    setTimer(prev => ({ ...prev, isRunning: false }));
    
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
    
    setTimeout(() => {
      setShowPgnDialog(true);
    }, 1000);
  }, [timer.whiteTime, timer.blackTime, formatTime]);

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

    if (isGameOver && gameEndReason) {
      endGame(gameEndReason, winner);
    }

    onGameStateUpdate?.(newGame.fen(), moves);
  }, [onGameStateUpdate, timer, formatTime, endGame]);

  const startNewGame = useCallback((fen?: string) => {
    const newGame = new Chess(fen || selectedOpening.fen);
    setGame(newGame);
    
    setTimer(prev => ({
      ...prev,
      whiteTime: 180 * 60,
      blackTime: 180 * 60,
      isRunning: false,
      currentPlayer: 'w',
      lastMoveTime: Date.now(),
    }));
    
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

  const startWithCustomFen = useCallback(() => {
    if (customFen.trim()) {
    try {
      const newGame = new Chess(customFen.trim());
      setGame(newGame);
      setGameState({
        fen: newGame.fen(),
        moves: [],
        isGameOver: newGame.isGameOver(),
        winner: null,
        isPlayerTurn: newGame.turn() === 'w',
        gameEndReason: null,
      });
      onGameStateUpdate?.(newGame.fen(), []);
      try {
        localStorage.setItem('circuit_game_fen', newGame.fen());
      } catch {}
      setTimer(prev => ({ ...prev, whiteTime: 180 * 60, blackTime: 180 * 60, isRunning: false, currentPlayer: 'w', lastMoveTime: Date.now() }));
      setMoveHistory([]);
      setAnalysis(null);
      setIsAutoPlay(false);
      setShowPgnDialog(false);
      if (autoPlayInterval) { clearInterval(autoPlayInterval); setAutoPlayInterval(null); }
    } catch (error) {
      alert('Invalid FEN');
    }
    }
  }, [customFen, autoPlayInterval]);

  // get stockfish analysis
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
          bestMove: 'N/A',
          evaluation: data.evaluation[0] - data.evaluation[2],
          depth: 0,
          wdl: {
            winProb: data.evaluation[0],
            drawProb: data.evaluation[1],
            lossProb: data.evaluation[2],
          },
        });
        return data;
      } else {
        console.error('Stockfish analysis failed:', response.status, response.statusText);
        const errorText = await response.text();
        console.error('Error details:', errorText);
      }
    } catch (error) {
      console.error('Get stockfish analysis failed:', error);
    }
    return null;
  }, []);

  // get model move
  const getModelMove = useCallback(async (fen: string): Promise<ModelMoveResponse | null> => {
    try {
      const endpoint = useSearch 
        ? `${import.meta.env.VITE_BACKEND_URL}/play_game_with_search`
        : `${import.meta.env.VITE_BACKEND_URL}/play_game`;
      
      const requestBody = useSearch 
        ? { 
            fen, 
            ...searchParams, 
            save_trace: saveMctsTrace,
            trace_max_edges: saveMctsTrace ? 0 : 1000
          }
        : { fen };
      
      console.log(`🎯 Request model move: ${useSearch ? 'MCTS search' : 'Direct inference'}, playouts=${searchParams.max_playouts}`);
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (response.ok) {
        const data = await response.json();
        
        if (useSearch && data.search_info) {
          setLastSearchInfo({
            total_playouts: data.search_info.total_playouts,
            max_depth_reached: data.search_info.max_depth_reached,
            max_depth_limit: data.search_info.max_depth_limit,
          });
          console.log(`✅ MCTS search completed: playouts=${data.search_info.total_playouts}, depth=${data.search_info.max_depth_reached}`);
        } else {
          setLastSearchInfo(null);
        }
        
        return data as ModelMoveResponse;
      } else {
        console.error('API call failed:', response.status, response.statusText);
        const errorText = await response.text();
        console.error('Error details:', errorText);
      }
    } catch (error) {
      console.error('Get model move failed:', error);
    }
    return null;
  }, [useSearch, searchParams, saveMctsTrace]);

  const downloadTraceFile = useCallback(async (filename: string) => {
    try {
      const url = `${import.meta.env.VITE_BACKEND_URL}/search_trace/files/${encodeURIComponent(filename)}`;
      const response = await fetch(url);
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Download MCTS search file failed:', errorText);
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
      console.error('Download MCTS search file failed:', error);
    }
  }, []);

  // convert UCI string to chess.js move object
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
    return m;
  }, []);

  // make move
  const makeMove = useCallback((move: any) => {
    try {
      const prevFenBefore = game.fen();
      const parsed0 = typeof move === 'string' ? toChessJsMove(move) : move;
      const parsed = typeof parsed0 === 'string' ? parsed0 : {
        from: String(parsed0.from || '').toLowerCase(),
        to: String(parsed0.to || '').toLowerCase(),
        promotion: parsed0.promotion ? String(parsed0.promotion).toLowerCase() : undefined,
      } as any;

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

      const historyVerbose = game.history({ verbose: true }) as any[];
      const movesUci = historyVerbose.map(m => m.from + m.to + (m.promotion ? m.promotion : ''));

      // use PGN to rebuild the game to keep the history, avoid FEN loss of undo stack
      const pgn = game.pgn();
      const replaced = new Chess();
      // new version of chess.js uses loadPgn, old version uses load_pgn, both are tried here
      if (typeof (replaced as any).loadPgn === 'function') {
        (replaced as any).loadPgn(pgn);
      } else if (typeof (replaced as any).load_pgn === 'function') {
        (replaced as any).load_pgn(pgn);
      }
      setGame(replaced);

      // record the FEN before this move, for evaluation
      setLastFenBeforeMove(prevFenBefore);

      // use updateGameState to update the state (including timer update)
      updateGameState(replaced, movesUci[movesUci.length - 1]);

      setDummy(prev => prev + 1); // force refresh
      return true;
    } catch (error) {
      console.error('Make move failed:', error);
    }
    return false;
  }, [game, toChessJsMove, updateGameState]);

  // handle player move: only allowed in human turn
  const handlePlayerMove = useCallback((move: string) => {
    if (!isHumanTurn) {
      return false;
    }

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

  // handle model move: only triggered in model turn
  const handleModelMove = useCallback(async () => {
    if (isModelTurn && !isLoading && !isTracing) {
      setIsLoading(true);
      try {
        const moveResponse = await getModelMove(game.fen());
        const modelMove = moveResponse?.move;
        if (!modelMove) {
          console.warn('Model did not return a move or returned empty');
          alert('Model did not return a move or returned empty');
          return;
        }
        if (modelMove && makeMove(modelMove)) {
          if (useSearch && saveMctsTrace && moveResponse?.trace_filename) {
            await downloadTraceFile(moveResponse.trace_filename);
          }
          // get the analysis of the new position
          if (gameMode === 'analysis') {
            setTimeout(() => {
              getStockfishAnalysis(game.fen());
            }, 100);
          }
        } else {
          alert('The move given by the model is invalid or illegal');
        }
      } finally {
        setIsLoading(false);
      }
    }
  }, [isModelTurn, isLoading, isTracing, getModelMove, game, makeMove, gameMode, getStockfishAnalysis, useSearch, saveMctsTrace, downloadTraceFile]);

  // auto play:
  // - human-human: auto play is meaningless, keep the current state (only keep the original logic of controlling the random player)
  // - human-model: human turn does not auto play; model turn auto call model
  // - model-model: both sides use model
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
            // human-human: do not auto play
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

  // clear timer
  useEffect(() => {
    return () => {
      if (autoPlayInterval) {
        clearInterval(autoPlayInterval);
      }
    };
  }, [autoPlayInterval]);

  // get the analysis of the current position
  const getCurrentAnalysis = useCallback(async () => {
    await getStockfishAnalysis(game.fen());
  }, [game, getStockfishAnalysis]);

  // undo the last move, directly undo the move on the current game instance (keep the history)
  const undoLastMove = useCallback(() => {
    try {
      const last = game.undo();
      if (!last) return; // no more undo

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

      // use PGN to rebuild the game to keep the history
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

      // notify the parent component the game state update
      onGameStateUpdate?.(nextFen, movesUci);

      setDummy(prev => prev + 1);
      if (gameMode === 'analysis') {
        setTimeout(() => { getStockfishAnalysis(nextFen); }, 100);
      }
    } catch (error) {
      console.error('Undo failed:', error);
    }
  }, [game, gameMode, getStockfishAnalysis, onGameStateUpdate]);

  // modify manual input to be compatible with UCI
  const handleManualMove = useCallback(() => {
    if (!manualMove.trim()) {
      setMoveError('Please input a move');
      return;
    }
    if (!isHumanTurn) {
      setMoveError('Now is not human turn');
      return;
    }
    if (gameState.isGameOver) {
      setMoveError('Game is over');
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
        setMoveError('Invalid move');
      }
    } catch (error) {
      setMoveError('Invalid move format');
    }
  }, [manualMove, isHumanTurn, gameState.isGameOver, game, updateGameState, gameMode, getStockfishAnalysis, toChessJsMove]);

  // call the backend to evaluate the move (based on the FEN before the last move and the last move UCI)
  const evaluateLastMove = useCallback(async () => {
    try {
      const lastMove = gameState.moves[gameState.moves.length - 1];
      if (!lastMove || !lastFenBeforeMove) {
        alert('No previous move to evaluate or missing previous FEN');
        return;
      }
      const resp = await fetch(`${import.meta.env.VITE_BACKEND_URL}/evaluate_move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen: lastFenBeforeMove, move: lastMove, time_limit: 0.2 }),
      });
      if (!resp.ok) {
        const t = await resp.text();
        throw new Error(t || 'Evaluation failed');
      }
      const data = await resp.json();
      setLastMoveEval(data);
    } catch (e) {
      console.error('Evaluation failed:', e);
      alert('Evaluation failed, please check the console log');
    }
  }, [gameState.moves, lastFenBeforeMove]);

  // generate PGN string
  const generatePgn = useCallback(() => {
    let pgn = '';
    
    // add header information
    Object.entries(pgnHeaders).forEach(([key, value]) => {
      pgn += `[${key} "${value}"]\n`;
    });
    
    pgn += '\n';
    
    // add move history
    const moves = game.history();
    for (let i = 0; i < moves.length; i++) {
      const moveNum = Math.floor(i / 2) + 1;
      const isWhiteMove = i % 2 === 0;
      
      if (isWhiteMove) {
        pgn += `${moveNum}. `;
      }
      
      pgn += `${moves[i]} `;
      
      // add time information
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

  // save PGN file
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

  // handle resignation
  const handleResign = useCallback((player: 'White' | 'Black') => {
    endGame('resignation', player === 'White' ? 'Black' : 'White');
  }, [endGame]);

  // Handle draw offer / agreement
  const handleDraw = useCallback(() => {
    endGame('draw');
  }, [endGame]);

  // Start the game timer
  const startTimer = useCallback(() => {
    setTimer(prev => ({ ...prev, isRunning: true, lastMoveTime: Date.now() }));
  }, []);

  return (
    <div className="container mx-auto p-6 space-y-6">
      <SaeComboLoader />

      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Play with model</h1>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-sm">
            <span>Black turn auto flip</span>
            <Switch checked={autoFlipWhenBlack} onCheckedChange={setAutoFlipWhenBlack} />
          </div>
        <div className="flex gap-2">
          <Button
            onClick={() => startNewGame()}
            variant="outline"
            size="sm"
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            New game
          </Button>
            <Button
              onClick={undoLastMove}
              variant="outline"
              size="sm"
              disabled={game.history().length === 0}
            >
              <Undo2 className="w-4 h-4 mr-2" />
              Undo last move
          </Button>
          <Button
            onClick={toggleAutoPlay}
            variant={isAutoPlay ? "destructive" : "default"}
            size="sm"
          >
            {isAutoPlay ? (
              <>
                <Square className="w-4 h-4 mr-2" />
                Stop auto play
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                Auto play
              </>
            )}
          </Button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Board area */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>Board</span>
                <div className="flex gap-2">
                  <Badge variant={isHumanTurn ? "default" : "secondary"}>
                    {isHumanTurn ? "Human turn" : "Model turn"}
                  </Badge>
                  {gameState.isGameOver && (
                    <Badge variant="destructive">
                      {gameState.winner ? `Game over: ${gameState.winner} wins` : "Draw"}
                    </Badge>
                  )}
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {/* timer display */}
              <div className="mb-4 flex justify-center gap-8">
                <div className={`text-center p-3 rounded-lg ${game.turn() === 'w' ? 'bg-blue-100 border-2 border-blue-500' : 'bg-gray-100'}`}>
                  <div className="text-sm font-medium text-gray-600">White</div>
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
                    {timer.isRunning ? 'Pause' : 'Start'}
                  </Button>
                </div>
                <div className={`text-center p-3 rounded-lg ${game.turn() === 'b' ? 'bg-blue-100 border-2 border-blue-500' : 'bg-gray-100'}`}>
                  <div className="text-sm font-medium text-gray-600">Black</div>
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
                    console.log('Click square:', square);
                  }}
                  // disable interaction if not human turn or tracing
                  isInteractive={isHumanTurn && !isTracing}
                  autoFlipWhenBlack={autoFlipWhenBlack}
                  analysisName="Game board"
                  showSelfPlay={false}
                />
              </div>
              
              {/* move history */}
              <div className="mt-4">
                <h3 className="text-sm font-medium mb-2">Move history</h3>
                <div className="bg-gray-50 p-3 rounded max-h-32 overflow-y-auto">
                  <div className="text-sm font-mono">
                    {gameState.moves.length > 0 ? gameState.moves.join(' ') : 'No moves yet'}
                  </div>
                </div>
              </div>
              
              {/* load logs has been migrated to the SaeComboLoader component at the top of the page */}
            </CardContent>
          </Card>
        </div>

        {/* control panel */}
        <div className="space-y-4">

          {/* game mode and human side selection */}
          <Card>
            <CardHeader>
              <CardTitle>Game settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 gap-3">
                <div>
                  <label className="text-sm font-medium">Game mode</label>
                  <Select value={matchMode} onValueChange={(v: 'human-human' | 'human-model' | 'model-model') => setMatchMode(v)}>
                    <SelectTrigger className="mt-1">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="human-human">Human vs Human</SelectItem>
                      <SelectItem value="human-model">Human vs Model</SelectItem>
                      <SelectItem value="model-model">Model vs Model</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {matchMode === 'human-model' && (
                  <div>
                    <label className="text-sm font-medium">Human side</label>
                    <Select value={humanPlays} onValueChange={(v: 'w' | 'b') => setHumanPlays(v)}>
                      <SelectTrigger className="mt-1">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="w">White</SelectItem>
                        <SelectItem value="b">Black</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* MCTS search settings */}
          <Card>
            <CardHeader>
              <CardTitle>Search settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Enable MCTS search</label>
                <Switch checked={useSearch} onCheckedChange={setUseSearch} />
              </div>
              
              {useSearch && (
                <div className="space-y-3 pt-2 border-t">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium">Save MCTS search JSON</label>
                    <Switch checked={saveMctsTrace} onCheckedChange={setSaveMctsTrace} />
                  </div>
                  <p className="text-xs text-gray-500">
                    After enabled, the search trace of the corresponding FEN will be automatically downloaded when the search is completed and the move is made (the default file name contains the current FEN).
                  </p>
                  <div>
                    <label className="text-sm font-medium">Maximum playouts</label>
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
                    <label className="text-sm font-medium">Maximum search depth</label>
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
                    <label className="text-sm font-medium">UCT exploration coefficient (cpuct)</label>
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
                    <label className="text-sm font-medium">Batch size</label>
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
                  
                  {/* low Q value exploration enhancement parameters */}
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
                        Enable low Q value exploration enhancement (used to find hidden good moves like dropping pieces)
                      </label>
                    </div>
                    
                    {searchParams.low_q_exploration_enabled && (
                      <div className="space-y-3 ml-6 mt-3 bg-blue-50 p-3 rounded">
                        <p className="text-xs text-gray-600 mb-2">
                          Give extra exploration reward to moves with Q value below the threshold and fewer visits. This helps discover moves that the model initially evaluates poorly but that may actually be strong.
                        </p>
                        <div className="space-y-2">
                          <div>
                            <label className="text-xs font-medium text-gray-700">Q threshold</label>
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
                              Moves with Q value below this threshold will be enhanced exploration (default 0.3, can be negative)
                            </p>
                          </div>
                          <div>
                            <label className="text-xs font-medium text-gray-700">Exploration bonus</label>
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
                              The base value of the reward, the greater the value, the more actively the exploration of low Q value moves will be (default 0.1)
                            </p>
                          </div>
                          <div>
                            <label className="text-xs font-medium text-gray-700">Visit threshold</label>
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
                              Moves with visits below this value will be rewarded (default 5)
                            </p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                  
                  {/* display last search information */}
                  {lastSearchInfo && (
                    <div className="bg-gray-50 p-2 rounded text-xs space-y-1">
                      <div><strong>Last search:</strong></div>
                      <div>Total playouts: {lastSearchInfo.total_playouts}</div>
                      <div>Reached depth: {lastSearchInfo.max_depth_reached}</div>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          {/* new: manually input move */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Move className="w-4 h-4" />
                Manually input move
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Input
                  placeholder="Input move (e.g. e2e4, Nf3, O-O)"
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
                  Execute move
                </Button>
              </div>
              
              {/* display legal moves */}
              <div className="space-y-2">
                <div className="text-sm font-medium">Legal moves:</div>
                <div className="bg-gray-50 p-2 rounded max-h-24 overflow-y-auto">
                  <div className="text-xs font-mono">
                    {game.moves().slice(0, 20).join(', ')}
                    {game.moves().length > 20 && '...'}
                  </div>
                </div>
                <div className="text-xs text-gray-500">
                  Total {game.moves().length} legal moves
                </div>
              </div>

              {/* model move control: move below manually input move */}
              <div className="border-t pt-4">
                <Button
                  onClick={handleModelMove}
                  disabled={!isModelTurn || isLoading || isTracing}
                  className="w-full"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Thinking...
                    </>
                  ) : (
                    'Let the model move'
                  )}
                </Button>
                
                {gameMode === 'analysis' && (
                  <Button
                    onClick={getCurrentAnalysis}
                    variant="outline"
                    className="w-full mt-2"
                  >
                    Get analysis
                  </Button>
                )}

                {/* evaluate last move */}
                <Button
                  onClick={evaluateLastMove}
                  variant="outline"
                  className="w-full mt-2"
                  disabled={!lastFenBeforeMove || gameState.moves.length === 0}
                >
                  Evaluate last move
                </Button>
              </div>
            </CardContent>
          </Card>


          {/* game mode (keep the original analysis mode switch, different from the above game settings) */}
          <Card>
            <CardHeader>
              <CardTitle>Game mode</CardTitle>
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
                  <SelectItem value="player-vs-model">Player vs Model</SelectItem>
                  <SelectItem value="analysis">Analysis mode</SelectItem>
                </SelectContent>
              </Select>
            </CardContent>
          </Card>

          {/* opening selection */}
          <Card>
            <CardHeader>
              <CardTitle>Select opening</CardTitle>
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
                Use this opening
              </Button>
            </CardContent>
          </Card>

          {/* custom FEN */}
          <Card>
            <CardHeader>
              <CardTitle>Custom FEN</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Textarea
                placeholder="Input FEN string..."
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
                Use custom FEN
              </Button>
            </CardContent>
          </Card>

          {/* analysis result */}
          {analysis && (
            <Card>
              <CardHeader>
                <CardTitle>Analysis result</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="text-sm">
                  <strong>Best move:</strong> {analysis.bestMove}
                </div>
                <div className="text-sm">
                  <strong>Evaluation:</strong> {analysis.evaluation ? analysis.evaluation.toFixed(2) : 'N/A'}
                </div>
                <div className="text-sm">
                  <strong>Depth:</strong> {analysis.depth || 'N/A'}
                </div>
                {analysis.wdl && (
                  <div className="text-sm">
                    <strong>Win rate:</strong> 
                    <div className="mt-1 space-y-1">
                      <div className="flex justify-between">
                        <span>White win:</span>
                        <span>{(analysis.wdl.winProb * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Draw:</span>
                        <span>{(analysis.wdl.drawProb * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Black win:</span>
                        <span>{(analysis.wdl.lossProb * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* display last move evaluation result */}
          {lastMoveEval && (
            <Card>
              <CardHeader>
                <CardTitle>Last move evaluation result</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <div><strong>My move:</strong> {lastMoveEval.my_move}</div>
                <div><strong>Best move:</strong> {lastMoveEval.best_move || 'N/A'}</div>
                <div><strong>Score(0-100):</strong> {typeof lastMoveEval.score_100 === 'number' ? lastMoveEval.score_100.toFixed(1) : 'N/A'}</div>
                <div><strong>CP loss:</strong> {lastMoveEval.cp_loss !== null && lastMoveEval.cp_loss !== undefined ? lastMoveEval.cp_loss.toFixed(1) : 'N/A'}</div>
                <div><strong>Root CP:</strong> {lastMoveEval.root_cp !== null && lastMoveEval.root_cp !== undefined ? lastMoveEval.root_cp.toFixed(1) : 'N/A'}</div>
                <div><strong>Best after CP:</strong> {lastMoveEval.best_cp !== null && lastMoveEval.best_cp !== undefined ? lastMoveEval.best_cp.toFixed(1) : 'N/A'}</div>
                <div><strong>My after CP:</strong> {lastMoveEval.my_cp !== null && lastMoveEval.my_cp !== undefined ? lastMoveEval.my_cp.toFixed(1) : 'N/A'}</div>
                {lastMoveEval.root_wdl && (
                  <div>
                    <strong>Root WDL:</strong> White win {(lastMoveEval.root_wdl[0]*100).toFixed(1)}% / Draw {(lastMoveEval.root_wdl[1]*100).toFixed(1)}% / Black win {(lastMoveEval.root_wdl[2]*100).toFixed(1)}%
                  </div>
                )}
                {lastMoveEval.best_wdl && (
                  <div>
                    <strong>Best after WDL:</strong> White win {(lastMoveEval.best_wdl[0]*100).toFixed(1)}% / Draw {(lastMoveEval.best_wdl[1]*100).toFixed(1)}% / Black win {(lastMoveEval.best_wdl[2]*100).toFixed(1)}%
                  </div>
                )}
                {lastMoveEval.my_wdl && (
                  <div>
                    <strong>My after WDL:</strong> White win {(lastMoveEval.my_wdl[0]*100).toFixed(1)}% / Draw {(lastMoveEval.my_wdl[1]*100).toFixed(1)}% / Black win {(lastMoveEval.my_wdl[2]*100).toFixed(1)}%
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* game end control */}
          {!gameState.isGameOver && (
            <Card>
              <CardHeader>
                <CardTitle>End game</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-2">
                  <Button
                    onClick={() => handleResign('White')}
                    variant="outline"
                    size="sm"
                    className="text-red-600 hover:text-red-700"
                  >
                    White resign
                  </Button>
                  <Button
                    onClick={() => handleResign('Black')}
                    variant="outline"
                    size="sm"
                    className="text-red-600 hover:text-red-700"
                  >
                    Black resign
                  </Button>
                </div>
                <Button
                  onClick={handleDraw}
                  variant="outline"
                  size="sm"
                  className="w-full text-blue-600 hover:text-blue-700"
                >
                  Propose draw
                </Button>
              </CardContent>
            </Card>
          )}

          {/* game state */}
          <Card>
            <CardHeader>
              <CardTitle>Game state</CardTitle>
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
                  <strong>Last FEN before move:</strong>
                  <div className="font-mono text-xs bg-gray-100 p-2 rounded mt-1 break-all">
                    {lastFenBeforeMove}
                  </div>
                </div>
              )}
              <div className="text-sm">
                <strong>Move count:</strong> {gameState.moves.length}
              </div>
              <div className="text-sm">
                <strong>Current turn:</strong> {game.turn() === 'w' ? 'White' : 'Black'}
              </div>
              {gameState.isGameOver && (
                <div className="text-sm">
                  <strong>Game end reason:</strong> {
                    gameState.gameEndReason === 'checkmate' ? 'Checkmate' :
                    gameState.gameEndReason === 'resignation' ? 'Resignation' :
                    gameState.gameEndReason === 'draw' ? 'Draw' : 'Unknown'
                  }
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* PGN save dialog */}
      <Dialog open={showPgnDialog} onOpenChange={setShowPgnDialog}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Download className="w-5 h-5" />
              Save game as PGN file
            </DialogTitle>
          </DialogHeader>
          
          <div className="space-y-4">
            <div className="text-sm text-gray-600">
              Game is over! Please confirm the following information and save the game.
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
            
            {/* display PGN preview */}
            <div className="space-y-2">
              <Label>PGN preview</Label>
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
              Cancel
            </Button>
            <Button
              onClick={() => {
                savePgnFile();
                setShowPgnDialog(false);
              }}
              className="flex items-center gap-2"
            >
              <Download className="w-4 h-4" />
              Save PGN file
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      
    </div>
  );
};