import { useState, useCallback } from 'react';
import { AppNavbar } from "@/components/app/navbar";
import { GameVisualization } from '@/components/play-game/game_visualization';
import { CircuitTracing } from '@/components/circuits/circuit-tracing';

export const PlayGamePage = () => {
  // Circuit Trace 相关状态
  const [isTracing, setIsTracing] = useState(false);

  // 游戏状态管理（简化版，主要用于Circuit Trace）
  const [gameFen, setGameFen] = useState('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  const [previousFen, setPreviousFen] = useState<string | null>(null); // 新增：上一个FEN状态
  const [gameHistory, setGameHistory] = useState<string[]>([]);
  const [lastMove, setLastMove] = useState<string | null>(null); // 新增：最后一个移动

  // Circuit Trace 回调函数
  const handleCircuitTraceStart = useCallback(() => {
    setIsTracing(true);
  }, []);

  const handleCircuitTraceEnd = useCallback(() => {
    setIsTracing(false);
  }, []);


  // 更新游戏状态的回调函数
  const handleGameStateUpdate = useCallback((fen: string, moves: string[]) => {
    // 保存上一个FEN状态
    setPreviousFen(gameFen);
    
    // 更新当前FEN
    setGameFen(fen);
    
    // 更新移动历史
    setGameHistory(moves);
    
    // 获取最后一个移动
    const lastMoveUci = moves.length > 0 ? moves[moves.length - 1] : null;
    setLastMove(lastMoveUci);
  }, [gameFen]);

  return (
    <div className="min-h-screen bg-background">
      <AppNavbar />
      <div className="container mx-auto p-4 space-y-6">
        {/* 游戏可视化组件 */}
        <GameVisualization
          onCircuitTraceStart={handleCircuitTraceStart}
          onCircuitTraceEnd={handleCircuitTraceEnd}
          onGameStateUpdate={handleGameStateUpdate}
        />

        {/* Circuit Trace 组件 */}
        <CircuitTracing
          gameFen={previousFen || gameFen} // 使用move之前的FEN
          previousFen={previousFen}
          currentFen={gameFen}
          gameHistory={gameHistory}
          lastMove={lastMove}
          onCircuitTraceStart={handleCircuitTraceStart}
          onCircuitTraceEnd={handleCircuitTraceEnd}
          isTracing={isTracing}
        />
      </div>
    </div>
  );
};