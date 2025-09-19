import React, { useMemo, useState } from 'react';

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
  move?: string; // 移动字符串，如 "a2a4"
  orientation?: 'white' | 'black' | 'auto'; // 方向覆盖
}

// 棋子Unicode符号映射
const PIECE_SYMBOLS: { [key: string]: string } = {
  'K': '♔', // 白王
  'Q': '♕', // 白后
  'R': '♖', // 白车
  'B': '♗', // 白象
  'N': '♘', // 白马
  'P': '♙', // 白兵
  'k': '♚', // 黑王
  'q': '♛', // 黑后
  'r': '♜', // 黑车
  'b': '♝', // 黑象
  'n': '♞', // 黑马
  'p': '♟', // 黑兵
};

// FEN字符串解析函数
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

  // 将FEN转换为8x8数组
  const board: (string | null)[][] = [];
  
  for (let i = 0; i < 8; i++) {
    const row: (string | null)[] = [];
    const rowStr = rows[i];
    
    for (const char of rowStr) {
      if (/\d/.test(char)) {
        // 数字表示空格数量
        const emptySquares = parseInt(char);
        for (let j = 0; j < emptySquares; j++) {
          row.push(null);
        }
      } else {
        // 棋子字符
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

// 将行列坐标转换为线性索引 (0-63)
const getSquareIndex = (row: number, col: number): number => {
  return row * 8 + col;
};

// 获取激活强度的颜色
const getActivationColor = (activation: number): string => {
  if (activation === 0) return 'transparent';
  
  // 激活值统一用红色表示，根据强度调整透明度
  const intensity = Math.min(Math.abs(activation), 1);
  const opacity = Math.max(0.4, intensity);
  
  return `rgba(239, 68, 68, ${opacity})`; // 红色表示激活
};

// 获取Z模式的目标格子 - 只显示最强的几个连接
const getZPatternTargets = (sourceSquare: number, zPatternIndices?: number[][], zPatternValues?: number[]) => {
  if (!zPatternIndices || !zPatternValues) return [];
  
  const targets: { square: number; strength: number }[] = [];
  
  // 检查数据格式 - 参考feature page逻辑
  const looksLikePairList = Array.isArray(zPatternIndices[0]) && (zPatternIndices[0] as number[]).length === 2;
  
  if (looksLikePairList) {
    // 格式：[[source, target], ...] 对应 [value, ...]
    for (let i = 0; i < zPatternIndices.length; i++) {
      const pair = zPatternIndices[i] as number[];
      const value = zPatternValues[i] || 0;
      const [source, target] = pair;
      
      if (source === sourceSquare) {
        targets.push({ square: target, strength: value });
      }
    }
  } else {
    // 格式：zPatternIndices[0]为源位置数组，zPatternIndices[1]为目标位置数组
    if (zPatternIndices.length >= 2) {
      const sources = zPatternIndices[0] as number[];
      const targets_array = zPatternIndices[1] as number[];
      
      // 遍历所有连接
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
  
  // 只返回绝对值最大的前8个连接（参考feature页面逻辑）
  return targets
    .sort((a, b) => Math.abs(b.strength) - Math.abs(a.strength))
    .slice(0, 8);
};

// 解析象棋移动字符串
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
}) => {
  // Hover状态管理
  const [hoveredSquare, setHoveredSquare] = useState<number | null>(null);

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
        <p className="text-red-700 text-sm">无效的FEN格式: {fen}</p>
      </div>
    );
  }

  const { board, isWhiteToMove } = parsedBoard;

  // 黑方行棋时翻转棋盘显示，让黑方棋子在下方
  const flip = !isWhiteToMove;
  const flip_activation = true;

  // 根据 flip 决定是否翻转棋盘
  const displayBoard = useMemo(() => {
    return flip ? [...board].reverse() : board;
  }, [board, flip]);

  // 解析移动信息
  const parsedMove = useMemo(() => {
    return move ? parseMove(move) : null;
  }, [move]);

  // 根据显示位置计算实际的squareIndex（考虑翻转）
  const getActualSquareIndex = (displayRow: number, col: number): number => {
    const actualRow = flip ? (7 - displayRow) : displayRow;
    return getSquareIndex(actualRow, col);
  };

  // 激活值索引映射 - 始终保持在原始绝对位置，不受翻转影响
  const getActivationIndex = (displayRow: number, col: number): number => {
    // 将显示行转换回原始棋盘行（如果棋盘被翻转）
    const originalRow = flip_activation ? (7 - displayRow) : displayRow;
    // 激活值始终使用原始位置索引
    return originalRow * 8 + col;
  };

  // 根据实际squareIndex计算显示位置（用于箭头绘制）
  const getDisplayPosition = (actualSquareIndex: number) => {
    const actualRow = Math.floor(actualSquareIndex / 8);
    const col = actualSquareIndex % 8;
    const displayRow = flip_activation ? (7 - actualRow) : actualRow;
    return { row: displayRow, col };
  };

  // Z模式和激活值的显示位置映射 - 始终保持在原始绝对位置
  const getDisplayPositionFromActivationIndex = (activationIndex: number) => {
    const originalRow = Math.floor(activationIndex / 8);
    const col = activationIndex % 8;
    
    // 根据棋盘是否翻转来确定显示行，但激活值索引本身不变
    const displayRow = flip_activation ? (7 - originalRow) : originalRow;
    return { row: displayRow, col };
  };

  // 根据显示行索引获取棋盘行号（1-8）
  const getDisplayRowNumber = (displayRowIndex: number) => {
    // 不管是否翻转，显示行0始终对应最上面一行，应该显示最大的行号
    return 8 - displayRowIndex;
  };

  // 统一的 从索引到标准坐标的命名（不依赖行棋方）
  const getSquareNameFromActivationIndex = (activationIndex: number) => {
    const activationRow = Math.floor(activationIndex / 8);
    const col = activationIndex % 8;
    return `${String.fromCharCode(97 + col)}${8 - activationRow}`;
  };

  // 根据尺寸设置样式
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

  // 获取当前hover格子的Z模式目标
  const zPatternTargets = hoveredSquare !== null ? getZPatternTargets(hoveredSquare, zPatternIndices, zPatternValues) : [];

  return (
    <div className="flex flex-col items-center space-y-2">
      {/* 棋盘信息 */}
      <div className="text-sm text-gray-600 text-center">
        {sampleIndex !== undefined && (
          <div>样本 #{sampleIndex}</div>
        )}
        {analysisName && (
          <div>分析: {analysisName}</div>
        )}
        <div className="mt-1">
          <span className={`inline-block w-3 h-3 rounded-full mr-1 ${
            isWhiteToMove ? 'bg-white border-2 border-gray-800' : 'bg-gray-800'
          }`}></span>
          {isWhiteToMove ? '白方行棋' : '黑方行棋'}
        </div>
      </div>

      {/* 棋盘容器 */}
      <div className={`relative ${sizeClasses[size]} border-4 border-gray-800 rounded-lg overflow-hidden shadow-lg`}>
        {/* 棋盘网格 */}
        <div className="w-full h-full grid grid-cols-8 grid-rows-8">
          {displayBoard.map((row, displayRowIndex) =>
            row.map((_, colIndex) => {
              const piece = row[colIndex];
              const isLight = (displayRowIndex + colIndex) % 2 === 0;
              const squareIndex = getActualSquareIndex(displayRowIndex, colIndex);
              const activationIndex = getActivationIndex(displayRowIndex, colIndex);
              const activation = activations?.[activationIndex] || 0;
              const activationColor = getActivationColor(activation);
              
              // 检查当前格子是否是hover格子的Z模式目标
              const isZPatternTarget = zPatternTargets.some(target => target.square === activationIndex);
              const targetStrength = zPatternTargets.find(target => target.square === activationIndex)?.strength || 0;
              
              // 检查是否为移动的起点或终点
              const isMoveFromSquare = parsedMove && parsedMove.from.index === activationIndex;
              const isMoveToSquare = parsedMove && parsedMove.to.index === activationIndex;
              
              // 获取基础背景色
              const baseColor = isLight ? 'bg-amber-100' : 'bg-amber-800';
              
              // 确定最终背景色
              let finalBackgroundColor;
              const isSourceSquare = hoveredSquare === activationIndex; // 当前格子是否为源格子
              
              if (isMoveFromSquare) {
                // 移动起点用黄绿色高亮
                finalBackgroundColor = 'rgba(34, 197, 94, 0.7)'; // green-500 with opacity
              } else if (isMoveToSquare) {
                // 移动终点用更深的绿色高亮
                finalBackgroundColor = 'rgba(22, 163, 74, 0.8)'; // green-600 with opacity
              } else if (isZPatternTarget) {
                // Z模式目标格子根据强度使用不同颜色 - 参考feature页面逻辑
                const absStrength = Math.abs(targetStrength);
                const normalizedStrength = Math.min(absStrength / 0.01, 1); // 归一化到[0,1]
                const opacity = Math.max(0.3, normalizedStrength * 0.7 + 0.3);
                
                if (targetStrength > 0) {
                  // 正值用蓝色
                  finalBackgroundColor = `rgba(59, 130, 246, ${opacity})`;
                } else {
                  // 负值用橙色/红色
                  finalBackgroundColor = `rgba(249, 115, 22, ${opacity})`;
                }
              } else if (activationColor !== 'transparent' && !isSourceSquare && hoveredSquare === null) {
                // 只有在未悬停任何格子时才显示激活红色高亮
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
                  `}
                  style={{
                    backgroundColor: finalBackgroundColor,
                  }}
                  onMouseEnter={() => {
                    // 只有激活值不为0的格子才响应hover
                    if (activation !== 0) {
                      setHoveredSquare(activationIndex);
                    }
                  }}
                  onMouseLeave={() => {
                    setHoveredSquare(null);
                  }}
                  title={`${String.fromCharCode(97 + colIndex)}${getDisplayRowNumber(displayRowIndex)}${
                    activation !== 0 ? ` (激活: ${activation.toFixed(3)})` : ''
                  }${
                    isZPatternTarget ? ` [Z模式目标: ${targetStrength.toFixed(3)}]` : ''
                  }${
                    isMoveFromSquare ? ' [移动起点]' : ''
                  }${
                    isMoveToSquare ? ' [移动终点]' : ''
                  }`}
                >
                  {/* 棋子 */}
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

                  {/* 激活值显示 - 在格子内部显示，源格子时隐去 */}
                  {activation !== 0 && !isSourceSquare && hoveredSquare === null && (
                    <div className="absolute top-0 right-0 bg-blue-600 text-white text-xs rounded px-1 leading-3" style={{ fontSize: '10px' }}>
                      {Math.abs(activation).toFixed(2)}
                    </div>
                  )}

                  {/* Z模式值显示 - 在格子内部左上角显示 */}
                  {isZPatternTarget && (
                    <div className={`absolute top-0 left-0 text-white text-xs rounded px-1 leading-3 ${
                      targetStrength > 0 ? 'bg-blue-700' : 'bg-orange-700'
                    }`} style={{ fontSize: '10px' }}>
                      {targetStrength.toFixed(3)}
                    </div>
                  )}

                  {/* 坐标标记 */}
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
        
        {/* 移动箭头 */}
        {parsedMove && (() => {
          const fromDisplayPos = getDisplayPositionFromActivationIndex(parsedMove.from.index);
          const toDisplayPos = getDisplayPositionFromActivationIndex(parsedMove.to.index);
          
          return (
            <svg
              className="absolute inset-0 w-full h-full pointer-events-none"
              viewBox="0 0 800 800"
              style={{ zIndex: 10 }}
            >
              <defs>
                <marker
                  id="arrowhead"
                  markerWidth="10"
                  markerHeight="7"
                  refX="10"
                  refY="3.5"
                  orient="auto"
                >
                  <polygon points="0 0, 10 3.5, 0 7" fill="#22c55e" />
                </marker>
              </defs>
              <line
                x1={fromDisplayPos.col * 100 + 50}
                y1={fromDisplayPos.row * 100 + 50}
                x2={toDisplayPos.col * 100 + 50}
                y2={toDisplayPos.row * 100 + 50}
                stroke="#22c55e"
                strokeWidth="6"
                markerEnd="url(#arrowhead)"
              />
            </svg>
          );
        })()}
        </div>

              {/* FEN字符串和移动信息显示 */}
      <div className="absolute -bottom-12 left-0 right-0 text-xs text-gray-500 text-center space-y-1">
        {parsedMove && (
          <div className="text-green-600 font-medium">
            移动: {parsedMove.moveString}
          </div>
        )}
        <div className="truncate">
          {fen}
        </div>
      </div>
      </div>

      {/* FEN字符串显示 */}
      <div className="w-full max-w-lg text-xs text-gray-600 bg-gray-50 rounded p-2 border">
        <div className="font-medium text-gray-800 mb-1">FEN字符串:</div>
        <div className="font-mono text-xs break-all select-all">
          {fen}
        </div>
      </div>

      {/* 激活值统计 */}
      {activations && activations.some(a => a !== 0) && (
        <div className="text-xs text-gray-600 space-y-1">
          <div>激活统计:</div>
          <div className="flex space-x-4">
            <span>激活格子: {activations.filter(a => a !== 0).length}</span>
            <span>最大值: {Math.max(...activations.map(Math.abs)).toFixed(3)}</span>
            <span className="text-red-600">🔴 激活值</span>
            {hoveredSquare !== null && <span className="text-blue-600">🔵 Z模式连接 (最强8个)</span>}
          </div>
        </div>
      )}

      {/* Z模式统计 */}
      {zPatternValues && zPatternValues.length > 0 && (
        <div className="text-xs text-gray-600 space-y-1">
          <div>Z模式连接: {zPatternValues.length}个</div>
          <div>强度范围: {Math.min(...zPatternValues).toFixed(3)} ~ {Math.max(...zPatternValues).toFixed(3)}</div>
          <div className="flex space-x-4">
            <span className="text-blue-600">🔵 正值连接</span>
            <span className="text-orange-600">🟠 负值连接</span>
          </div>
        </div>
      )}

      {/* Hover状态显示Z模式连接详情 */}
      {hoveredSquare !== null && (() => {
        const squareName = getSquareNameFromActivationIndex(hoveredSquare);
        const activation = activations?.[hoveredSquare] || 0;
        
        return (
          <div className="text-xs bg-blue-50 border border-blue-200 rounded p-2 space-y-1">
            <div className="font-medium text-blue-800">
              格子 {squareName} (激活值: {activation.toFixed(3)})
            </div>
            {zPatternTargets.length > 0 ? (
              <>
                <div className="text-blue-700">Z模式最强连接 ({zPatternTargets.length}个):</div>
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
                    <div className="text-blue-500 col-span-2">... 还有 {zPatternTargets.length - 6} 个连接</div>
                  )}
                </div>
              </>
            ) : (
              <div className="text-blue-600">无Z模式连接</div>
            )}
          </div>
        );
      })()}
    </div>
  );
};
