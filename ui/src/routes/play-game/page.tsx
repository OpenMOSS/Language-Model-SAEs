import { useState, useCallback } from 'react';
import { AppNavbar } from "@/components/app/navbar";
import { GameVisualization } from '@/components/play-game/game_visualization';
import { CircuitTracing } from '@/components/circuits/circuit-tracing';

export const PlayGamePage = () => {
  const [isTracing, setIsTracing] = useState(false);

  // Game state management
  const [gameFen, setGameFen] = useState('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  const [previousFen, setPreviousFen] = useState<string | null>(null); // Previous FEN state
  const [gameHistory, setGameHistory] = useState<string[]>([]);
  const [lastMove, setLastMove] = useState<string | null>(null); // Last move in UCI form

  // Circuit Trace callback functions
  const handleCircuitTraceStart = useCallback(() => {
    setIsTracing(true);
  }, []);

  const handleCircuitTraceEnd = useCallback(() => {
    setIsTracing(false);
  }, []);


  // Update game state callback function
  const handleGameStateUpdate = useCallback((fen: string, moves: string[]) => {
    // Save previous FEN state
    setPreviousFen(gameFen);
    
    // Update current FEN
    setGameFen(fen);
    
    // Update move history
    setGameHistory(moves);
    
    // Get last move
    const lastMoveUci = moves.length > 0 ? moves[moves.length - 1] : null;
    setLastMove(lastMoveUci);
  }, [gameFen]);

  return (
    <div className="min-h-screen bg-background">
      <AppNavbar />
      <div className="container mx-auto p-4 space-y-6">
        {/* Game visualization component */}
        <GameVisualization
          onCircuitTraceStart={handleCircuitTraceStart}
          onCircuitTraceEnd={handleCircuitTraceEnd}
          onGameStateUpdate={handleGameStateUpdate}
        />

        {/* Circuit Trace component */}
        <CircuitTracing
          gameFen={previousFen || gameFen} // use FEN before move
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