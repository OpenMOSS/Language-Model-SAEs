import React, { useMemo, useState, useEffect } from 'react';

interface ChessBoardProps {
  fen: string;
  activations?: number[];
  zPatternIndices?: number[][];
  zPatternValues?: number[];
  sampleIndex?: number;
  analysisName?: string;
  contextId?: number;
  size?: 'small' | 'medium' | 'large';
  showCoordinates?: boolean;
  move?: string; // move string, like "a2a4"
  orientation?: 'white' | 'black' | 'auto'; // direction cover
  flip_activation?: boolean; // control whether to flip activation value
  onMove?: (move: string) => void; // new: move callback
  onSquareClick?: (square: string) => void; // new: square click callback
  isInteractive?: boolean; // new: whether to allow interaction
  autoFlipWhenBlack?: boolean; // new: automatically flip when black to move
  moveColor?: string; // new: arrow color (same as node color)
}

// chess piece Unicode symbol mapping
const PIECE_SYMBOLS: { [key: string]: string } = {
  'K': '♔', // white king
  'Q': '♕', // white queen
  'R': '♖', // white rook
  'B': '♗', // white bishop
  'N': '♘', // white knight
  'P': '♙', // white pawn
  'k': '♚', // black king
  'q': '♛', // black queen
  'r': '♜', // black rook
  'b': '♝', // black bishop
  'n': '♞', // black knight
  'p': '♟', // black pawn
};

// FEN string parsing function
const parseFEN = (fen: string) => {
  const parts = fen.trim().split(' ');
  if (parts.length < 4) {
    throw new Error('Invalid FEN string');
  }

  const [boardPart, activeColor, castling, enPassant] = parts;
  const rows = boardPart.split('/');

  if (rows.length !== 8) {
    throw new Error('Invalid board configuration in FEN');
  }

  // convert FEN to 8x8 array
  const board: (string | null)[][] = [];
  
  for (let i = 0; i < 8; i++) {
    const row: (string | null)[] = [];
    const rowStr = rows[i];
    
    for (const char of rowStr) {
      if (/\d/.test(char)) {
        // numbers represent the number of empty squares
        const emptySquares = parseInt(char);
        for (let j = 0; j < emptySquares; j++) {
          row.push(null);
        }
      } else {
        // chess piece characters
        row.push(char);
      }
    }
    
    board.push(row);
  }

  return {
    board,
    activeColor,
    castling,
    enPassant,
    isWhiteToMove: activeColor === 'w'
  };
};

// convert row and column coordinates to linear index (0-63)
const getSquareIndex = (row: number, col: number): number => {
  return row * 8 + col;
};

// get the color of activation strength
const getActivationColor = (activation: number): string => {
  if (activation === 0) return 'transparent';
  
  // activation value is represented by red, adjust opacity based on strength
  const intensity = Math.min(Math.abs(activation), 1);
  const opacity = Math.max(0.4, intensity);
  
  return `rgba(239, 68, 68, ${opacity})`; // red represents activation
};

// get the target squares of Z pattern - only display the strongest few connections
const getZPatternTargets = (sourceSquare: number, zPatternIndices?: number[][], zPatternValues?: number[]) => {
  if (!zPatternIndices || !zPatternValues) return [];
  
  const targets: { square: number; strength: number }[] = [];
  
  // check data format - reference feature page logic
  const looksLikePairList = Array.isArray(zPatternIndices[0]) && (zPatternIndices[0] as number[]).length === 2;
  
  if (looksLikePairList) {
    // format: [[source, target], ...] corresponds to [value, ...]
    for (let i = 0; i < zPatternIndices.length; i++) {
      const pair = zPatternIndices[i] as number[];
      const value = zPatternValues[i] || 0;
      const [source, target] = pair;
      
      if (source === sourceSquare) {
        targets.push({ square: target, strength: value });
      }
    }
  } else {
    // format: zPatternIndices[0] is the source position array, zPatternIndices[1] is the target position array
    if (zPatternIndices.length >= 2) {
      const sources = zPatternIndices[0] as number[];
      const targets_array = zPatternIndices[1] as number[];
      
      for (let i = 0; i < Math.min(sources.length, targets_array.length, zPatternValues.length); i++) {
        const source = sources[i];
        const target = targets_array[i];
        const value = zPatternValues[i] || 0;
        
        if (source === sourceSquare) {
          targets.push({ square: target, strength: value });
        }
      }
    }
  }
  
  // only return the top 8 connections by absolute strength (reference feature page logic)    
  return targets
    .sort((a, b) => Math.abs(b.strength) - Math.abs(a.strength))
    .slice(0, 8);
};

// parse chess move string
const parseMove = (move: string) => {
  if (!move || move.length < 4) return null;
  
  const fromSquare = move.substring(0, 2);
  const toSquare = move.substring(2, 4);
  
  const parseSquare = (square: string) => {
    if (square.length !== 2) return null;
    const col = square.charCodeAt(0) - 97; // a=0, b=1, etc.
    const row = 8 - parseInt(square[1]); // 8=0, 7=1, etc.
    if (col < 0 || col > 7 || row < 0 || row > 7) return null;
    return { row, col, index: row * 8 + col };
  };
  
  const from = parseSquare(fromSquare);
  const to = parseSquare(toSquare);
  
  if (!from || !to) return null;
  
  return {
    from,
    to,
    moveString: `${fromSquare}${toSquare}`
  };
};

export const ChessBoard: React.FC<ChessBoardProps> = ({
  fen,
  activations,
  zPatternIndices,
  zPatternValues,
  sampleIndex,
  analysisName,
  contextId,
  size = 'medium',
  showCoordinates = true,
  move,
  orientation = 'auto',
  flip_activation = true,
  onMove,
  onSquareClick,
  isInteractive = false,
  autoFlipWhenBlack = false,
  moveColor,
}) => {
  // a concise log: directly print the data structure for display
  console.log(`[CB#${sampleIndex ?? 'NA'}] activations:`, activations);
  console.log(`[CB#${sampleIndex ?? 'NA'}] zPatternIndices:`, zPatternIndices);
  console.log(`[CB#${sampleIndex ?? 'NA'}] zPatternValues:`, zPatternValues);

  // hover state management
  const [hoveredSquare, setHoveredSquare] = useState<number | null>(null);
  
  // new: click selection state management
  const [selectedSquare, setSelectedSquare] = useState<number | null>(null);
  const [possibleMoves, setPossibleMoves] = useState<string[]>([]);
  const [boardEvaluation, setBoardEvaluation] = useState<number[] | null>(null);

  // modify handleAnalyze function, remove trailing comma in JSON.stringify
  const handleAnalyze = async () => {
    try {
      const res = await fetch("/analyze/board", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fen })
      });
      const data = await res.json();
      setBoardEvaluation(data.evaluation);
    } catch (error) {
      console.error("Failed to analyze position:", error);
    }
  };

  // add useEffect to automatically call handleAnalyze, triggered when fen changes
  useEffect(() => {
    handleAnalyze();
  }, [fen]);

  const parsedBoard = useMemo(() => {
    try {
      return parseFEN(fen);
    } catch (error) {
      console.error('Failed to parse FEN:', error);
      return null;
    }
  }, [fen]);

  if (!parsedBoard) {
    return (
      <div className="p-4 border border-red-200 rounded-lg bg-red-50">
        <p className="text-red-700 text-sm">Invalid FEN format: {fen}</p>
      </div>
    );
  }

  const { board, isWhiteToMove } = parsedBoard;

  // flip board when black to move (controlled by switch)
  const flip = autoFlipWhenBlack ? !isWhiteToMove : false;

  // determine whether to flip board based on flip
  const displayBoard = useMemo(() => {
    return flip ? [...board].reverse() : board;
  }, [board, flip]);

  // parse move information
  const parsedMove = useMemo(() => {
    return move ? parseMove(move) : null;
  }, [move]);

  // calculate actual square index based on display position (considering flip)
  const getActualSquareIndex = (displayRow: number, col: number): number => {
    const actualRow = flip ? (7 - displayRow) : displayRow;
    return getSquareIndex(actualRow, col);
  };

  // activation index mapping
  const getActivationIndex = (displayRow: number, col: number): number => {
    // convert display row back to original board row (if board is flipped)
    const originalRow = flip_activation ? (7 - displayRow) : displayRow;
    // activation value always uses original position index
    return originalRow * 8 + col;
  };

  // calculate display position based on actual square index (for arrow drawing)
  const getDisplayPosition = (actualSquareIndex: number) => {
    const actualRow = Math.floor(actualSquareIndex / 8);
    const col = actualSquareIndex % 8;
    const displayRow = flip_activation ? (7 - actualRow) : actualRow;
    return { row: displayRow, col };
  };

  // new: calculate display position based on board orientation (flip) for board elements/arrows aligning with board squares
  const getBoardDisplayPosition = (actualSquareIndex: number) => {
    const actualRow = Math.floor(actualSquareIndex / 8);
    const col = actualSquareIndex % 8;
    const displayRow = flip ? (7 - actualRow) : actualRow;
    return { row: displayRow, col };
  };

  // Z pattern and activation value display position mapping
  const getDisplayPositionFromActivationIndex = (activationIndex: number) => {
    const originalRow = Math.floor(activationIndex / 8);
    const col = activationIndex % 8;
    
    // determine display row based on whether board is flipped, but activation index itself remains unchanged
    const displayRow = flip_activation ? (7 - originalRow) : originalRow;
    return { row: displayRow, col };
  };
  const getDisplayRowNumber = (displayRowIndex: number) => {
    return 8 - displayRowIndex;
  };

  const getSquareNameFromActivationIndex = (activationIndex: number) => {
    const activationRow = Math.floor(activationIndex / 8);
    const col = activationIndex % 8;
    return `${String.fromCharCode(97 + col)}${8 - activationRow}`;
  };

  const handleSquareClick = (activationIndex: number, displayRow: number, col: number) => {
    if (!isInteractive) return;
    
    const squareName = getSquareNameFromActivationIndex(activationIndex);
    console.log('Click square:', squareName, 'activation index:', activationIndex);
    
    if (selectedSquare === null) {
      // first click: select start square
      setSelectedSquare(activationIndex);
      setPossibleMoves([]);
      onSquareClick?.(squareName);
      console.log('Select start square:', squareName);
    } else if (selectedSquare === activationIndex) {
      // click on the same square: cancel selection
      setSelectedSquare(null);
      setPossibleMoves([]);
      console.log('Cancel selection');
    } else {
      // second click: try to move
      const fromSquare = getSquareNameFromActivationIndex(selectedSquare);
      const moveString = `${fromSquare}${squareName}`;
      
      console.log('Try to move:', moveString);
      
      // call move callback
      onMove?.(moveString);
      
      // clear selection state
      setSelectedSquare(null);
      setPossibleMoves([]);
    }
  };

  // set styles based on size
  const sizeClasses = {
    small: 'w-64 h-64',
    medium: 'w-80 h-80',
    large: 'w-96 h-96'
  };

  const squareSize = {
    small: 'w-8 h-8',
    medium: 'w-10 h-10',
    large: 'w-12 h-12'
  };

  const fontSize = {
    small: 'text-lg',
    medium: 'text-xl',
    large: 'text-2xl'
  };

  const squareSizePxMap = {
    small: 32,   // 256px / 8
    medium: 40,  // 320px / 8
    large: 48,   // 384px / 8
  } as const;
  const boardPx = squareSizePxMap[size] * 8;

  // get Z pattern targets for hovered square
  const zPatternTargets = hoveredSquare !== null ? getZPatternTargets(hoveredSquare, zPatternIndices, zPatternValues) : [];

  return (
    <div className="flex flex-col items-center space-y-2">
      {/* board information */}
      <div className="text-sm text-gray-600 text-center">
        {sampleIndex !== undefined && (
          <div>Sample #{sampleIndex}</div>
        )}
        {analysisName && (
          <div>Analysis: {analysisName}</div>
        )}
        <div className="mt-1">
          <span className={`inline-block w-3 h-3 rounded-full mr-1 ${
            isWhiteToMove ? 'bg-white border-2 border-gray-800' : 'bg-gray-800'
          }`}></span>
          {isWhiteToMove ? 'White to move' : 'Black to move'}
        </div>
        {isInteractive && selectedSquare !== null && (
          <div className="mt-1 text-blue-600 font-medium">
            Selected: {getSquareNameFromActivationIndex(selectedSquare)}
          </div>
        )}
      </div>

      {/* Board container */}
      <div className={`relative ${sizeClasses[size]} border-4 border-gray-800 rounded-lg overflow-hidden shadow-lg`}>
        {/* Board grid */}
        <div className="w-full h-full grid grid-cols-8 grid-rows-8">
          {displayBoard.map((row, displayRowIndex) =>
            row.map((_, colIndex) => {
              const piece = row[colIndex];
              const isLight = (displayRowIndex + colIndex) % 2 === 0;
              const squareIndex = getActualSquareIndex(displayRowIndex, colIndex);
              const activationIndex = getActivationIndex(displayRowIndex, colIndex);
              const activation = activations?.[activationIndex] || 0;
              const activationColor = getActivationColor(activation);
              
              // Check whether this square is a Z-pattern target of the hovered square
              const isZPatternTarget = zPatternTargets.some(target => target.square === activationIndex);
              const targetStrength = zPatternTargets.find(target => target.square === activationIndex)?.strength || 0;
              
              // Check whether this is the move's from/to square
              const isMoveFromSquare = parsedMove && parsedMove.from.index === squareIndex;
              const isMoveToSquare = parsedMove && parsedMove.to.index === squareIndex;
              
              // Check whether this is the selected square
              const isSelectedSquare = selectedSquare === activationIndex;
              
              // Base background color
              const baseColor = isLight ? 'bg-amber-100' : 'bg-amber-800';
              
              // Determine final background color
              let finalBackgroundColor;
              const isSourceSquare = hoveredSquare === activationIndex; // whether this is the source square
              
              if (isSelectedSquare) {
                // Selected square highlighted in blue
                finalBackgroundColor = 'rgba(59, 130, 246, 0.8)'; // blue-500 with opacity
              } else if (isMoveFromSquare) {
                // Move start square uses moveColor
                finalBackgroundColor = (moveColor || 'rgba(34, 197, 94, 0.7)');
              } else if (isMoveToSquare) {
                // Move end square uses a darker shade of the same color
                finalBackgroundColor = (moveColor || 'rgba(22, 163, 74, 0.8)');
              } else if (isZPatternTarget) {
                // Z-pattern target squares use different colors based on strength (mirroring feature page)
                const absStrength = Math.abs(targetStrength);
                const normalizedStrength = Math.min(absStrength / 0.01, 1); // normalize to [0,1]
                const opacity = Math.max(0.3, normalizedStrength * 0.7 + 0.3);
                
                if (targetStrength > 0) {
                  // Positive connections in blue
                  finalBackgroundColor = `rgba(59, 130, 246, ${opacity})`;
                } else {
                  // Negative connections in orange/red
                  finalBackgroundColor = `rgba(249, 115, 22, ${opacity})`;
                }
              } else if (activationColor !== 'transparent' && !isSourceSquare && hoveredSquare === null) {
                // Only show activation highlight when no square is hovered
                finalBackgroundColor = activationColor;
              }
              
              return (
                <div
                  key={`${displayRowIndex}-${colIndex}`}
                  className={`
                    ${squareSize[size]} 
                    relative flex items-center justify-center
                    ${baseColor}
                    transition-all duration-200 hover:brightness-110
                    ${activation !== 0 ? 'cursor-pointer' : ''}
                    ${isInteractive ? 'cursor-pointer' : ''}
                    ${isSelectedSquare ? 'ring-2 ring-blue-500 ring-opacity-75' : ''}
                  `}
                  style={{
                    backgroundColor: finalBackgroundColor,
                  }}
                  onMouseEnter={() => {
                    // Only squares with non-zero activation respond to hover
                    if (activation !== 0) {
                      setHoveredSquare(activationIndex);
                    }
                  }}
                  onMouseLeave={() => {
                    setHoveredSquare(null);
                  }}
                  onClick={() => handleSquareClick(activationIndex, displayRowIndex, colIndex)}
                  title={`${String.fromCharCode(97 + colIndex)}${getDisplayRowNumber(displayRowIndex)}${
                    activation !== 0 ? ` (activation: ${activation.toFixed(3)})` : ''
                  }${
                    isZPatternTarget ? ` [Z-pattern target: ${targetStrength.toFixed(3)}]` : ''
                  }${
                    isMoveFromSquare ? ' [move from]' : ''
                  }${
                    isMoveToSquare ? ' [move to]' : ''
                  }${
                    isSelectedSquare ? ' [selected]' : ''
                  }`}
                >
                  {/* Piece */}
                  {piece && (
                    <span
                      className={`
                        ${fontSize[size]} 
                        ${piece === piece.toLowerCase() ? 'text-black' : 'text-white'}
                        drop-shadow-sm select-none
                      `}
                      style={{
                        textShadow: piece === piece.toLowerCase() ? '1px 1px 1px rgba(255,255,255,0.8)' : '1px 1px 1px rgba(0,0,0,0.8)'
                      }}
                    >
                      {PIECE_SYMBOLS[piece] || piece}
                    </span>
                  )}

                  {/* Activation value badge (hidden on source square) */}
                  {activation !== 0 && !isSourceSquare && hoveredSquare === null && (
                    <div className="absolute top-0 right-0 bg-blue-600 text-white text-xs rounded px-1 leading-3" style={{ fontSize: '10px' }}>
                      {Math.abs(activation).toFixed(2)}
                    </div>
                  )}

                  {/* Z-pattern strength badge (top-left inside square) */}
                  {isZPatternTarget && (
                    <div className={`absolute top-0 left-0 text-white text-xs rounded px-1 leading-3 ${
                      targetStrength > 0 ? 'bg-blue-700' : 'bg-orange-700'
                    }`} style={{ fontSize: '10px' }}>
                      {targetStrength.toFixed(3)}
                    </div>
                  )}

                  {/* Coordinate labels */}
                  {showCoordinates && (
                    <>
                      {colIndex === 0 && (
                        <div className="absolute left-1 top-1 text-xs font-bold opacity-60">
                          {getDisplayRowNumber(displayRowIndex)}
                        </div>
                      )}
                      {displayRowIndex === 7 && (
                        <div className="absolute bottom-1 right-1 text-xs font-bold opacity-60">
                          {String.fromCharCode(97 + colIndex)}
                        </div>
                      )}
                    </>
                  )}
                </div>
              );
            })
          )}
        
        {/* Move arrow overlay */}
        {parsedMove && (
          <svg
            className="absolute inset-0 pointer-events-none"
            width={boardPx}
            height={boardPx}
            viewBox={`0 0 ${boardPx} ${boardPx}`}
            style={{ zIndex: 10 }}
          >
            <defs>
              <marker id="arrow-head" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L0,6 L6,3 z" fill={moveColor || '#4b5563'} />
              </marker>
            </defs>
            {(() => {
              // Use board orientation mapping to keep the arrow aligned with rendered squares
              const fromPos = getBoardDisplayPosition(parsedMove.from.index);
              const toPos = getBoardDisplayPosition(parsedMove.to.index);
              const sq = squareSizePxMap[size];
              const fromX = fromPos.col * sq + sq / 2;
              const fromY = fromPos.row * sq + sq / 2;
              const toX = toPos.col * sq + sq / 2;
              const toY = toPos.row * sq + sq / 2;
              return (
                <line
                  x1={fromX}
                  y1={fromY}
                  x2={toX}
                  y2={toY}
                  stroke={moveColor || '#4b5563'}
                  strokeWidth={4}
                  strokeOpacity={0.9}
                  markerEnd="url(#arrow-head)"
                />
              );
            })()}
          </svg>
        )}

        {/* Placeholder: graph nodes/edges layer (not implemented here) */}
      </div>

      {/* FEN string and move info */}
      <div className="absolute -bottom-12 left-0 right-0 text-xs text-gray-500 text-center space-y-1">
        {parsedMove && (
          <div className="text-green-600 font-medium">
            Move: {parsedMove.moveString}
          </div>
        )}
        <div className="truncate">
          {fen}
        </div>
      </div>
      </div>

      {/* FEN string display */}
      <div className="w-full max-w-lg text-xs text-gray-600 bg-gray-50 rounded p-2 border">
        <div className="font-medium text-gray-800 mb-1">FEN string:</div>
        <div className="font-mono text-xs break-all select-all">
          {fen}
        </div>
      </div>
      <div className="mt-2">
        {boardEvaluation ? (
          <div className="mt-1 text-sm text-gray-700">
            Win rate: {boardEvaluation[0].toFixed(2)}, draw rate: {boardEvaluation[1].toFixed(2)}, opponent win rate: {boardEvaluation[2].toFixed(2)}
          </div>
        ) : (
          <div className="mt-1 text-sm text-gray-700">Analyzing position...</div>
        )}
      </div>

      {/* Activation statistics */}
      {activations && activations.some(a => a !== 0) && (
        <div className="text-xs text-gray-600 space-y-1">
          <div>Activation stats:</div>
          <div className="flex space-x-4">
            <span>Activated squares: {activations.filter(a => a !== 0).length}</span>
            <span>Max |activation|: {Math.max(...activations.map(Math.abs)).toFixed(3)}</span>
            <span className="text-red-600">🔴 Activation value</span>
            {hoveredSquare !== null && <span className="text-blue-600">🔵 Z-pattern connections (top 8 strongest)</span>}
          </div>
        </div>
      )}

      {/* Z-pattern statistics */}
      {zPatternValues && zPatternValues.length > 0 && (
        <div className="text-xs text-gray-600 space-y-1">
          <div>Z-pattern connections: {zPatternValues.length}</div>
          <div>Strength range: {Math.min(...zPatternValues).toFixed(3)} ~ {Math.max(...zPatternValues).toFixed(3)}</div>
          <div className="flex space-x-4">
            <span className="text-blue-600">🔵 Positive connections</span>
            <span className="text-orange-600">🟠 Negative connections</span>
          </div>
        </div>
      )}

      {/* Hover state: show Z-pattern connection details */}
      {hoveredSquare !== null && (() => {
        const squareName = getSquareNameFromActivationIndex(hoveredSquare);
        const activation = activations?.[hoveredSquare] || 0;
        
        return (
          <div className="text-xs bg-blue-50 border border-blue-200 rounded p-2 space-y-1">
            <div className="font-medium text-blue-800">
              Square {squareName} (activation: {activation.toFixed(3)})
            </div>
            {zPatternTargets.length > 0 ? (
              <>
                <div className="text-blue-700">Strongest Z-pattern connections ({zPatternTargets.length}):</div>
                <div className="grid grid-cols-2 gap-1">
                  {zPatternTargets.slice(0, 6).map((target, idx) => {
                    const targetName = getSquareNameFromActivationIndex(target.square);
                    const isPositive = target.strength > 0;
                    return (
                      <div key={idx} className={isPositive ? "text-blue-600" : "text-orange-600"}>
                        → {targetName} ({target.strength.toFixed(3)})
                      </div>
                    );
                  })}
                  {zPatternTargets.length > 6 && (
                    <div className="text-blue-500 col-span-2">... plus {zPatternTargets.length - 6} more connections</div>
                  )}
                </div>
              </>
            ) : (
              <div className="text-blue-600">No Z-pattern connections</div>
            )}
          </div>
        );
      })()}
    </div>
  );
};
