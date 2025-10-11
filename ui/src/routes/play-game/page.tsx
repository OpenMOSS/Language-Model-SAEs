import { useState, useCallback } from 'react';
import { AppNavbar } from "@/components/app/navbar";
import { GameVisualization } from '@/components/play-game/game_visualization';
import { CircuitTracing } from '@/components/circuits/circuit-tracing';

export const PlayGamePage = () => {
  // Circuit Trace 相关状态
  const [isTracing, setIsTracing] = useState(false);

  // 游戏状态管理（简化版，主要用于Circuit Trace）
  const [gameFen, setGameFen] = useState('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  const [gameHistory, setGameHistory] = useState<string[]>([]);

  // Circuit Trace 回调函数
  const handleCircuitTraceStart = useCallback(() => {
    setIsTracing(true);
  }, []);

  const handleCircuitTraceEnd = useCallback(() => {
    setIsTracing(false);
  }, []);


  // 更新游戏状态的回调函数
  const handleGameStateUpdate = useCallback((fen: string, moves: string[]) => {
    setGameFen(fen);
    setGameHistory(moves);
  }, []);

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
          gameFen={gameFen}
          gameHistory={gameHistory}
          onCircuitTraceStart={handleCircuitTraceStart}
          onCircuitTraceEnd={handleCircuitTraceEnd}
          isTracing={isTracing}
        />
      </div>
    </div>
  );
};