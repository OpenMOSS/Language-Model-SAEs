import React, { useMemo, useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

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
  flip_activation?: boolean; // 控制激活值是否翻转
  onMove?: (move: string) => void; // 新增：移动回调
  onSquareClick?: (square: string) => void; // 新增：格子点击回调
  isInteractive?: boolean; // 新增：是否允许交互
  autoFlipWhenBlack?: boolean; // 新增：到黑方行棋时自动翻转
  moveColor?: string; // 新增：箭头颜色（与节点颜色一致）
  showSelfPlay?: boolean; // 新增：是否显示自对弈功能
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
  size = 'medium',
  showCoordinates = true,
  move,
  flip_activation = true,
  onMove,
  onSquareClick,
  isInteractive = false,
  autoFlipWhenBlack = false,
  moveColor,
  showSelfPlay = false,
}) => {
  // 一行精简日志：直接打印用于显示的数据结构
  console.log(`[CB#${sampleIndex ?? 'NA'}] activations:`, activations);
  console.log(`[CB#${sampleIndex ?? 'NA'}] zPatternIndices:`, zPatternIndices);
  console.log(`[CB#${sampleIndex ?? 'NA'}] zPatternValues:`, zPatternValues);

  // Hover状态管理
  const [hoveredSquare, setHoveredSquare] = useState<number | null>(null);
  
  // 新增：点击选择状态管理
  const [selectedSquare, setSelectedSquare] = useState<number | null>(null);
  const [boardEvaluation, setBoardEvaluation] = useState<number[] | null>(null);

  // 自对弈相关状态
  const [selfPlayData, setSelfPlayData] = useState<any>(null);
  const [selfPlayLoading, setSelfPlayLoading] = useState(false);
  const [selfPlayError, setSelfPlayError] = useState<string | null>(null);
  const [currentSelfPlayStep, setCurrentSelfPlayStep] = useState(0);
  const [showSelfPlayPanel, setShowSelfPlayPanel] = useState(false);

  // 修改handleAnalyze函数，移除JSON.stringify中对象的尾随逗号
  const handleAnalyze = async () => {
    try {
      console.log(`[CB#${sampleIndex ?? 'NA'}] 正在分析局面: ${fen.substring(0, 50)}...`);
      const res = await fetch(`${import.meta.env.VITE_BACKEND_URL}/analyze/board`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fen })
      });
      const data = await res.json();
      setBoardEvaluation(data.evaluation);
      console.log(`[CB#${sampleIndex ?? 'NA'}] 局面分析完成:`, data.evaluation);
    } catch (error) {
      console.error(`[CB#${sampleIndex ?? 'NA'}] 分析局面失败:`, error);
    }
  };

  // 在状态声明区域之后添加 useEffect 来自动调用 handleAnalyze，当 fen 变化时触发
  useEffect(() => {
    handleAnalyze();
  }, [fen]);

  // 自对弈函数
  const handleSelfPlay = useCallback(async () => {
    setSelfPlayLoading(true);
    setSelfPlayError(null);
    
    try {
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/self_play`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          initial_fen: fen,
          max_moves: 5, // 固定5步
          temperature: 1.0
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const result = await response.json();
      setSelfPlayData(result);
      setCurrentSelfPlayStep(0);
      setShowSelfPlayPanel(true);
      
    } catch (err) {
      console.error('自对弈失败:', err);
      setSelfPlayError(err instanceof Error ? err.message : '自对弈失败');
    } finally {
      setSelfPlayLoading(false);
    }
  }, [fen]);

  // 准备图表数据
  const chartData = useMemo(() => {
    if (!selfPlayData || !selfPlayData.wdl_history.length) return [];
    
    return selfPlayData.wdl_history.map((wdl: any, index: number) => {
      // 根据当前局面判断行棋方，转换为固定的白方/黑方胜率
      const currentFen = selfPlayData.positions[index];
      const isWhiteToMove = currentFen.includes(' w ');
      
      let whiteWinRate, blackWinRate;
      if (isWhiteToMove) {
        // 白方行棋时：win=白方胜率，loss=黑方胜率
        whiteWinRate = wdl.win;
        blackWinRate = wdl.loss;
      } else {
        // 黑方行棋时：win=黑方胜率，loss=白方胜率
        whiteWinRate = wdl.loss;
        blackWinRate = wdl.win;
      }
      
      return {
        move: index + 1,
        whiteWin: (whiteWinRate * 100).toFixed(1),
        draw: (wdl.draw * 100).toFixed(1),
        blackWin: (blackWinRate * 100).toFixed(1),
        whiteWin_num: whiteWinRate,
        draw_num: wdl.draw,
        blackWin_num: blackWinRate
      };
    });
  }, [selfPlayData]);

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

  // 黑方行棋时翻转棋盘显示（受开关控制）
  const flip = autoFlipWhenBlack ? !isWhiteToMove : false;

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

  // 激活值索引映射 - 激活值数组按照标准棋盘坐标：a1=0, b1=1, ..., h8=63
  const getActivationIndex = (displayRow: number, col: number): number => {
    // 将显示行转换回原始棋盘行（根据棋盘是否翻转）
    const originalRow = flip ? (7 - displayRow) : displayRow;
    // 根据flip_activation参数决定是否翻转激活值索引
    if (flip_activation) {
      // 翻转激活值索引：FEN第0行对应第1行，FEN第7行对应第8行
      return originalRow * 8 + col;
    } else {
      // 不翻转激活值索引：FEN第0行对应第8行，FEN第7行对应第1行
      const standardRow = 7 - originalRow;
      return standardRow * 8 + col;
    }
  };

  // 新增：根据棋盘朝向(flip)计算显示位置（用于棋盘元素/箭头对齐棋盘格子）
  const getBoardDisplayPosition = (actualSquareIndex: number) => {
    const actualRow = Math.floor(actualSquareIndex / 8);
    const col = actualSquareIndex % 8;
    const displayRow = flip ? (7 - actualRow) : actualRow;
    return { row: displayRow, col };
  };

  // 根据显示行索引获取棋盘行号（1-8）
  const getDisplayRowNumber = (displayRowIndex: number) => {
    if (flip) {
      // 翻转时：显示行0对应第1行，显示行7对应第8行
      return displayRowIndex + 1;
    } else {
      // 正常时：显示行0对应第8行，显示行7对应第1行
      return 8 - displayRowIndex;
    }
  };

  // 根据显示列索引获取棋盘列字母（a-h）
  const getDisplayColLetter = (colIndex: number) => {
    if (flip) {
      // 翻转时：列0对应h，列7对应a
      return String.fromCharCode(104 - colIndex); // 104 = 'h'
    } else {
      // 正常时：列0对应a，列7对应h
      return String.fromCharCode(97 + colIndex); // 97 = 'a'
    }
  };

  // 统一的 从索引到标准坐标的命名（不依赖行棋方）
  const getSquareNameFromActivationIndex = (activationIndex: number) => {
    const row = Math.floor(activationIndex / 8);
    const col = activationIndex % 8;
    // 根据flip_activation参数决定如何转换行号
    if (flip_activation) {
      // 翻转模式：激活索引直接对应FEN棋盘位置
      // FEN第0行对应第8行，FEN第7行对应第1行
      return `${String.fromCharCode(97 + col)}${8 - row}`;
    } else {
      // 标准模式：激活值数组按照标准棋盘坐标：a1=0, ..., h8=63
      // 标准行0对应第1行，标准行7对应第8行
      return `${String.fromCharCode(97 + col)}${row + 1}`;
    }
  };

  // 新增：处理格子点击
  const handleSquareClick = (activationIndex: number) => {
    if (!isInteractive) return;
    
    const squareName = getSquareNameFromActivationIndex(activationIndex);
    console.log('点击格子:', squareName, '激活索引:', activationIndex);
    
    if (selectedSquare === null) {
      // 第一次点击：选择起点
      setSelectedSquare(activationIndex);
      onSquareClick?.(squareName);
      console.log('选择起点:', squareName);
    } else if (selectedSquare === activationIndex) {
      // 点击同一格子：取消选择
      setSelectedSquare(null);
      console.log('取消选择');
    } else {
      // 第二次点击：尝试移动
      const fromSquare = getSquareNameFromActivationIndex(selectedSquare);
      const moveString = `${fromSquare}${squareName}`;
      
      console.log('尝试移动:', moveString);
      
      // 调用移动回调
      onMove?.(moveString);
      
      // 清除选择状态
      setSelectedSquare(null);
    }
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

  const squareSizePxMap = {
    small: 32,   // 256px / 8
    medium: 40,  // 320px / 8
    large: 48,   // 384px / 8
  } as const;
  const boardPx = squareSizePxMap[size] * 8;

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
        {isInteractive && selectedSquare !== null && (
          <div className="mt-1 text-blue-600 font-medium">
            已选择: {getSquareNameFromActivationIndex(selectedSquare)}
          </div>
        )}
      </div>

      {/* 棋盘容器 */}
      <div className={`relative ${sizeClasses[size]} border-4 border-gray-800 rounded-lg overflow-hidden shadow-lg`}>
        {/* 棋盘网格 */}
        <div className="w-full h-full grid grid-cols-8 grid-rows-8">
          {displayBoard.map((row, displayRowIndex) =>
            row.map((_, colIndex) => {
              const piece = row[colIndex];
              // 计算格子颜色：当翻转时，需要反转格子颜色
              // 在标准棋盘上，a1是黑格（dark），h1是白格（light）
              // 翻转后，所有黑格应该变成白格，所有白格应该变成黑格
              const normalIsLight = (displayRowIndex + colIndex) % 2 === 0;
              const isLight = flip ? !normalIsLight : normalIsLight;
              const squareIndex = getActualSquareIndex(displayRowIndex, colIndex);
              const activationIndex = getActivationIndex(displayRowIndex, colIndex);
              const activation = activations?.[activationIndex] || 0;
              const activationColor = getActivationColor(activation);
              
              // 检查当前格子是否是hover格子的Z模式目标
              const isZPatternTarget = zPatternTargets.some(target => target.square === activationIndex);
              const targetStrength = zPatternTargets.find(target => target.square === activationIndex)?.strength || 0;
              
              // 检查是否为移动的起点或终点
              const isMoveFromSquare = parsedMove && parsedMove.from.index === squareIndex;
              const isMoveToSquare = parsedMove && parsedMove.to.index === squareIndex;
              
              // 新增：检查是否为选中的格子
              const isSelectedSquare = selectedSquare === activationIndex;
              
              // 获取基础背景色
              const baseColor = isLight ? 'bg-amber-100' : 'bg-amber-800';
              
              // 确定最终背景色
              let finalBackgroundColor;
              const isSourceSquare = hoveredSquare === activationIndex; // 当前格子是否为源格子
              
              if (isSelectedSquare) {
                // 选中的格子用蓝色高亮
                finalBackgroundColor = 'rgba(59, 130, 246, 0.8)'; // blue-500 with opacity
              } else if (isMoveFromSquare) {
                // 移动起点使用moveColor
                finalBackgroundColor = (moveColor || 'rgba(34, 197, 94, 0.7)');
              } else if (isMoveToSquare) {
                // 移动终点使用更深的同色系
                finalBackgroundColor = (moveColor || 'rgba(22, 163, 74, 0.8)');
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
                    ${isInteractive ? 'cursor-pointer' : ''}
                    ${isSelectedSquare ? 'ring-2 ring-blue-500 ring-opacity-75' : ''}
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
                  onClick={() => handleSquareClick(activationIndex)}
                  title={`${getDisplayColLetter(colIndex)}${getDisplayRowNumber(displayRowIndex)}${
                    activation !== 0 ? ` (${activation.toFixed(3)})` : ''
                  }${
                    isZPatternTarget ? ` [目标: ${targetStrength.toFixed(3)}]` : ''
                  }${
                    isMoveFromSquare ? ' [移动起点]' : ''
                  }${
                    isMoveToSquare ? ' [移动终点]' : ''
                  }${
                    isSelectedSquare ? ' [已选中]' : ''
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

                  {/* 移除错误的对称高亮，直接按实际起点/终点高亮 */}

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
                      {/* 行号：显示在左侧（colIndex === 0）或右侧（翻转时） */}
                      {(!flip && colIndex === 0) || (flip && colIndex === 7) ? (
                        <div className={`absolute ${flip ? 'right-1' : 'left-1'} top-1 text-xs font-bold opacity-60`}>
                          {getDisplayRowNumber(displayRowIndex)}
                        </div>
                      ) : null}
                      {/* 列字母：显示在底部（displayRowIndex === 7）或顶部（翻转时） */}
                      {(!flip && displayRowIndex === 7) || (flip && displayRowIndex === 0) ? (
                        <div className={`absolute ${flip ? 'top-1' : 'bottom-1'} ${flip ? 'left-1' : 'right-1'} text-xs font-bold opacity-60`}>
                          {getDisplayColLetter(colIndex)}
                        </div>
                      ) : null}
                    </>
                  )}
                </div>
              );
            })
          )}
        
        {/* 移动箭头覆盖层 */}
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
              // 使用棋盘朝向映射，保证箭头与格子渲染一致
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
      <div className="mt-2">
        {boardEvaluation ? (
          <div className="mt-1 text-sm text-gray-700">
            w:{boardEvaluation[0].toFixed(2)}, d:{boardEvaluation[1].toFixed(2)}, l:{boardEvaluation[2].toFixed(2)}
          </div>
        ) : (
          <div className="mt-1 text-sm text-gray-700">正在分析局面...</div>
        )}
      </div>

      {/* 激活值统计 */}
      {activations && activations.some(a => a !== 0) && (
        <div className="text-xs text-gray-600 space-y-1">
          <div>激活统计:</div>
          <div className="flex space-x-4">
            <span>激活格子: {activations.filter(a => a !== 0).length}</span>
            <span>最大值: {Math.max(...activations.map(Math.abs)).toFixed(3)}</span>
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
              格子 {squareName} ({activation.toFixed(3)})
            </div>
            {zPatternTargets.length > 0 ? (
              <>
                <div className="text-blue-700">({zPatternTargets.length}个):</div>
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
              <div className="text-blue-600">无连接</div>
            )}
          </div>
        );
      })()}

      {/* 自对弈功能 */}
      {showSelfPlay && (
        <div className="mt-6 space-y-4">
          {/* 自对弈控制按钮 */}
          <div className="flex justify-center space-x-4">
            <button
              onClick={handleSelfPlay}
              disabled={selfPlayLoading}
              className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              {selfPlayLoading ? '进行中...' : '开始自对弈 (5步)'}
            </button>
            {selfPlayData && (
              <button
                onClick={() => setShowSelfPlayPanel(!showSelfPlayPanel)}
                className="px-4 py-2 bg-gray-500 text-white rounded-md hover:bg-gray-600 transition-colors"
              >
                {showSelfPlayPanel ? '隐藏结果' : '显示结果'}
              </button>
            )}
          </div>

          {/* 错误显示 */}
          {selfPlayError && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <p className="text-red-700 text-sm">{selfPlayError}</p>
            </div>
          )}

          {/* 自对弈结果显示 */}
          {selfPlayData && showSelfPlayPanel && (
            <div className="space-y-4">
              {/* 步骤导航 */}
              <div className="bg-white rounded-lg border p-4">
                <h3 className="text-lg font-semibold mb-3">自对弈步骤导航</h3>
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      当前步骤: {currentSelfPlayStep + 1} / {selfPlayData.positions.length}
                    </label>
                    <input
                      type="range"
                      min="0"
                      max={selfPlayData.positions.length - 1}
                      value={currentSelfPlayStep}
                      onChange={(e) => setCurrentSelfPlayStep(parseInt(e.target.value))}
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>
                  
                  {/* 步骤信息摘要 */}
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                    <div className="text-sm font-medium text-blue-800 mb-2">步骤 {currentSelfPlayStep + 1} 信息:</div>
                    <div className="space-y-1 text-xs text-blue-700">
                      {currentSelfPlayStep > 0 && (
                        <div>
                          <span className="font-medium">执行移动:</span> 
                          <span className="font-mono ml-1 px-1 bg-blue-100 rounded">
                            {selfPlayData.moves[currentSelfPlayStep - 1]}
                          </span>
                        </div>
                      )}
                      <div>
                        <span className="font-medium">局面FEN:</span>
                        <div className="font-mono text-xs mt-1 p-2 bg-white rounded border break-all">
                          {selfPlayData.positions[currentSelfPlayStep]}
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex justify-center space-x-2">
                    <button
                      onClick={() => setCurrentSelfPlayStep(0)}
                      className="px-3 py-1 bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors text-sm"
                    >
                      开始
                    </button>
                    <button
                      onClick={() => setCurrentSelfPlayStep(Math.max(0, currentSelfPlayStep - 1))}
                      className="px-3 py-1 bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors text-sm"
                      disabled={currentSelfPlayStep === 0}
                    >
                      上一步
                    </button>
                    <button
                      onClick={() => setCurrentSelfPlayStep(Math.min(selfPlayData.positions.length - 1, currentSelfPlayStep + 1))}
                      className="px-3 py-1 bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors text-sm"
                      disabled={currentSelfPlayStep === selfPlayData.positions.length - 1}
                    >
                      下一步
                    </button>
                    <button
                      onClick={() => setCurrentSelfPlayStep(selfPlayData.positions.length - 1)}
                      className="px-3 py-1 bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors text-sm"
                    >
                      结束
                    </button>
                  </div>
                </div>
              </div>

              {/* 自对弈棋盘 */}
              <div className="bg-white rounded-lg border p-4">
                <h3 className="text-lg font-semibold mb-3">
                  自对弈局面 {currentSelfPlayStep + 1}
                  {currentSelfPlayStep < selfPlayData.moves.length && (
                    <span className="text-green-600 ml-2">
                      (移动: {selfPlayData.moves[currentSelfPlayStep]})
                    </span>
                  )}
                </h3>
                
                {/* FEN字符串显示 */}
                <div className="mb-4 p-3 bg-gray-50 rounded-lg border">
                  <div className="text-sm font-medium text-gray-700 mb-2">当前FEN:</div>
                  <div className="font-mono text-xs text-gray-800 break-all select-all bg-white p-2 rounded border">
                    {selfPlayData.positions[currentSelfPlayStep]}
                  </div>
                </div>
                <div className="flex justify-center">
                  <div className="relative w-80 h-80 border-4 border-gray-800 rounded-lg overflow-hidden shadow-lg">
                    <div className="w-full h-full grid grid-cols-8 grid-rows-8">
                      {(() => {
                        try {
                          const parsedBoard = parseFEN(selfPlayData.positions[currentSelfPlayStep]);
                          const currentMove = currentSelfPlayStep > 0 ? selfPlayData.moves[currentSelfPlayStep - 1] : undefined;
                          const parsedMove = currentMove ? parseMove(currentMove) : null;
                          
                          return parsedBoard.board.map((row, displayRowIndex) =>
                            row.map((piece, colIndex) => {
                              const isLight = (displayRowIndex + colIndex) % 2 === 0;
                              const squareIndex = displayRowIndex * 8 + colIndex;
                              const isMoveFromSquare = parsedMove && parsedMove.from.index === squareIndex;
                              const isMoveToSquare = parsedMove && parsedMove.to.index === squareIndex;
                              
                              return (
                                <div
                                  key={`${displayRowIndex}-${colIndex}`}
                                  className={`
                                    w-10 h-10 relative flex items-center justify-center
                                    ${isLight ? 'bg-amber-100' : 'bg-amber-800'}
                                    ${isMoveFromSquare ? 'bg-green-400' : ''}
                                    ${isMoveToSquare ? 'bg-green-600' : ''}
                                  `}
                                >
                                  {piece && (
                                    <span className="text-xl select-none">
                                      {PIECE_SYMBOLS[piece] || piece}
                                    </span>
                                  )}
                                </div>
                              );
                            })
                          );
                        } catch (error) {
                          return <div className="col-span-8 row-span-8 flex items-center justify-center text-red-500">无效FEN</div>;
                        }
                      })()}
                    </div>
                    
                    {/* 移动箭头 */}
                    {(() => {
                      const currentMove = currentSelfPlayStep > 0 ? selfPlayData.moves[currentSelfPlayStep - 1] : undefined;
                      const parsedMove = currentMove ? parseMove(currentMove) : null;
                      if (!parsedMove) return null;
                      
                      const fromX = parsedMove.from.col * 40 + 20;
                      const fromY = parsedMove.from.row * 40 + 20;
                      const toX = parsedMove.to.col * 40 + 20;
                      const toY = parsedMove.to.row * 40 + 20;
                      
                      return (
                        <svg className="absolute inset-0 pointer-events-none" width="320" height="320">
                          <defs>
                            <marker id="arrow-head-selfplay" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
                              <path d="M0,0 L0,6 L6,3 z" fill="#3b82f6" />
                            </marker>
                          </defs>
                          <line
                            x1={fromX}
                            y1={fromY}
                            x2={toX}
                            y2={toY}
                            stroke="#3b82f6"
                            strokeWidth={3}
                            markerEnd="url(#arrow-head-selfplay)"
                          />
                        </svg>
                      );
                    })()}
                  </div>
                </div>
              </div>

              {/* WDL曲线图 */}
              {chartData.length > 0 && (
                <div className="bg-white rounded-lg border p-4">
                  <h3 className="text-lg font-semibold mb-3">WDL变化曲线</h3>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="move" />
                        <YAxis />
                        <Tooltip 
                          formatter={(value: any, name: string) => [
                            `${parseFloat(value).toFixed(1)}%`, 
                            name === 'whiteWin' ? '白方胜率' : name === 'draw' ? '和棋率' : name === 'blackWin' ? '黑方胜率' : name
                          ]}
                        />
                        <Legend />
                        <Line 
                          type="monotone" 
                          dataKey="whiteWin" 
                          stroke="#10b981" 
                          strokeWidth={2}
                          name="白方胜率"
                          dot={{ fill: '#10b981', strokeWidth: 2, r: 3 }}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="draw" 
                          stroke="#f59e0b" 
                          strokeWidth={2}
                          name="和棋率"
                          dot={{ fill: '#f59e0b', strokeWidth: 2, r: 3 }}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="blackWin" 
                          stroke="#ef4444" 
                          strokeWidth={2}
                          name="黑方胜率"
                          dot={{ fill: '#ef4444', strokeWidth: 2, r: 3 }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* 当前步骤WDL评估 */}
              {currentSelfPlayStep < selfPlayData.wdl_history.length && (
                <div className="bg-white rounded-lg border p-4">
                  <h3 className="text-lg font-semibold mb-3">步骤 {currentSelfPlayStep + 1} WDL评估</h3>
                  <div className="grid grid-cols-3 gap-4">
                    {(() => {
                      // 根据当前局面判断行棋方
                      const currentFen = selfPlayData.positions[currentSelfPlayStep];
                      const currentIsWhiteToMove = currentFen.includes(' w ');
                      
                      if (currentIsWhiteToMove) {
                        // 白方行棋：显示白方胜率、和棋率、黑方胜率
                        return (
                          <>
                            <div className="text-center">
                              <div className="text-2xl font-bold text-green-600">
                                {(selfPlayData.wdl_history[currentSelfPlayStep].win * 100).toFixed(1)}%
                              </div>
                              <div className="text-sm text-gray-600">白方胜率</div>
                            </div>
                            <div className="text-center">
                              <div className="text-2xl font-bold text-yellow-600">
                                {(selfPlayData.wdl_history[currentSelfPlayStep].draw * 100).toFixed(1)}%
                              </div>
                              <div className="text-sm text-gray-600">和棋率</div>
                            </div>
                            <div className="text-center">
                              <div className="text-2xl font-bold text-red-600">
                                {(selfPlayData.wdl_history[currentSelfPlayStep].loss * 100).toFixed(1)}%
                              </div>
                              <div className="text-sm text-gray-600">黑方胜率</div>
                            </div>
                          </>
                        );
                      } else {
                        // 黑方行棋：显示黑方胜率、和棋率、白方胜率
                        return (
                          <>
                            <div className="text-center">
                              <div className="text-2xl font-bold text-green-600">
                                {(selfPlayData.wdl_history[currentSelfPlayStep].win * 100).toFixed(1)}%
                              </div>
                              <div className="text-sm text-gray-600">黑方胜率</div>
                            </div>
                            <div className="text-center">
                              <div className="text-2xl font-bold text-yellow-600">
                                {(selfPlayData.wdl_history[currentSelfPlayStep].draw * 100).toFixed(1)}%
                              </div>
                              <div className="text-sm text-gray-600">和棋率</div>
                            </div>
                            <div className="text-center">
                              <div className="text-2xl font-bold text-red-600">
                                {(selfPlayData.wdl_history[currentSelfPlayStep].loss * 100).toFixed(1)}%
                              </div>
                              <div className="text-sm text-gray-600">白方胜率</div>
                            </div>
                          </>
                        );
                      }
                    })()}
                  </div>
                </div>
              )}

              {/* 移动概率 */}
              {currentSelfPlayStep < selfPlayData.move_probabilities.length && 
               Object.keys(selfPlayData.move_probabilities[currentSelfPlayStep]).length > 0 && (
                <div className="bg-white rounded-lg border p-4">
                  <h3 className="text-lg font-semibold mb-3">步骤 {currentSelfPlayStep + 1} 移动概率</h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2">
                    {Object.entries(selfPlayData.move_probabilities[currentSelfPlayStep])
                      .sort(([,a], [,b]) => (b as number) - (a as number))
                      .slice(0, 10)
                      .map(([move, prob]) => (
                        <div key={move} className="bg-gray-50 rounded p-2 text-center">
                          <div className="font-mono text-sm font-medium">{move}</div>
                          <div className="text-xs text-gray-600">
                            {((prob as number) * 100).toFixed(1)}%
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
