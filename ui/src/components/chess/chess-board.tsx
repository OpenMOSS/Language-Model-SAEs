import React, { useState, useMemo } from 'react';

interface ChessBoardProps {
  fen: string;
  activations?: number[];
  zPatternIndices?: number[][]; // 改为二维：每个目标格子的贡献来源索引列表
  zPatternValues?: number[];    // 与zPatternIndices同长度或可重复，表示贡献强度
  sampleIndex?: number;
  analysisName?: string;
  contextId?: number;
}

// 棋子符号映射
const pieceSymbols: Record<string, string> = {
  'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
  'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
};

export const ChessBoard: React.FC<ChessBoardProps> = ({
  fen,
  activations = [],
  zPatternIndices = [],
  zPatternValues = [],
  sampleIndex,
  analysisName,
  contextId
}) => {
  const [hoveredSquare, setHoveredSquare] = useState<number | null>(null);

  // 解析FEN格式
  const fenParts = fen.split(' ');
  if (fenParts.length < 6) {
    return (
      <div className="text-red-500">
        FEN格式错误: 应包含6个部分，实际包含{fenParts.length}个部分<br/>
        FEN: {fen}
      </div>
    );
  }

  const [boardFen, currentActiveColor, castling, enPassant, halfmove, fullmove] = fenParts;
  const activeColor = currentActiveColor;

  // 将FEN棋盘转换为8x8数组
  const board = useMemo(() => {
    const boardArray: string[][] = Array(8).fill(null).map(() => Array(8).fill(''));
    const rows = boardFen.split('/');
    
    if (rows.length !== 8) {
      return null;
    }
    
    for (let i = 0; i < 8; i++) {
      let col = 0;
      for (const char of rows[i]) {
        if (/\d/.test(char)) {
          const emptySquares = parseInt(char);
          col += emptySquares;
        } else {
          if (col < 8) {
            boardArray[i][col] = char;
            col++;
          }
        }
      }
    }
    return boardArray;
  }, [boardFen]);

  if (!board) {
    return (
      <div className="text-red-500">
        FEN棋盘格式错误: 应包含8行，实际包含{boardFen.split('/').length}行<br/>
        棋盘部分: {boardFen}
      </div>
    );
  }

  // 处理激活值 - 创建64长度的数组
  const boardActivations = useMemo(() => {
    const activationsArray = new Array(64).fill(0);
    if (activations && activations.length > 0) {
      if (activations.length === 64) {
        activationsArray.splice(0, 64, ...activations);
      } else {
        for (let i = 0; i < Math.min(activations.length, 64); i++) {
          activationsArray[i] = activations[i];
        }
      }
    }
    return activationsArray;
  }, [activations]);

  // 根据悬停的格子，计算对应的Z Pattern映射（长度64），参考sample.tsx的getZPatternForToken
  const hoveredZMap = useMemo(() => {
    const zMap = new Array(64).fill(0);
    if (hoveredSquare === null) return zMap;
    if (!Array.isArray(zPatternIndices) || zPatternIndices.length === 0) return zMap;

    // zPatternIndices: 每个目标格子i的贡献来源索引列表（数组），
    // 若其中包含 hoveredSquare，则该目标格子受其影响。强度取对应的zPatternValues[i]（若存在），否则置1。
    const len = Math.min(zPatternIndices.length, zPatternValues?.length || zPatternIndices.length);
    for (let target = 0; target < len; target++) {
      const sources = Array.isArray(zPatternIndices[target]) ? zPatternIndices[target] : [];
      if (sources.includes(hoveredSquare)) {
        const val = zPatternValues && zPatternValues[target] != null ? zPatternValues[target] : 1;
        zMap[target] = val;
      }
    }
    return zMap;
  }, [hoveredSquare, zPatternIndices, zPatternValues]);

  // 获取激活值对应的颜色
  const getActivationColor = (activation: number, isLight: boolean): string => {
    if (activation === 0) {
      return isLight ? '#f0d9b5' : '#b58863'; // 默认棋盘颜色
    }
    
    // 计算激活值的相对强度（相对于所有激活值的最大值）
    const maxActivation = Math.max(...boardActivations);
    const intensity = maxActivation > 0 ? Math.min(activation / maxActivation, 1) : 0;
    
    if (isLight) {
      // 浅色格子：从默认颜色渐变到红色
      const r = Math.round(240 + (255 - 240) * intensity);
      const g = Math.round(217 - 217 * intensity);
      const b = Math.round(181 - 181 * intensity);
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      // 深色格子：从默认颜色渐变到深红色
      const r = Math.round(181 + (139 - 181) * intensity);
      const g = Math.round(136 - 136 * intensity);
      const b = Math.round(99 - 99 * intensity);
      return `rgb(${r}, ${g}, ${b})`;
    }
  };

  // 获取z pattern对应的颜色（蓝色系）
  const getZPatternColor = (zValue: number, isLight: boolean): string => {
    if (zValue === 0) {
      return isLight ? '#f0d9b5' : '#b58863'; // 默认棋盘颜色
    }
    
    // 计算z pattern的相对强度
    const maxZValue = Math.max(...hoveredZMap);
    const intensity = maxZValue > 0 ? Math.min(zValue / maxZValue, 1) : 0;
    
    if (isLight) {
      // 浅色格子：从默认颜色渐变到蓝色
      const r = Math.round(240 - 100 * intensity);
      const g = Math.round(217 - 100 * intensity);
      const b = Math.round(181 + 74 * intensity);
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      // 深色格子：从默认颜色渐变到深蓝色
      const r = Math.round(181 - 100 * intensity);
      const g = Math.round(136 - 100 * intensity);
      const b = Math.round(99 + 156 * intensity);
      return `rgb(${r}, ${g}, ${b})`;
    }
  };

  // 获取方格的背景颜色
  const getSquareBackgroundColor = (row: number, col: number, activation: number, squareIndex: number): string => {
    const isLight = (row + col) % 2 === 0;
    
    // 悬停时：清除所有红色激活高亮，仅显示由 hoveredSquare 触发的 z pattern 映射
    if (hoveredSquare !== null) {
      const zVal = hoveredZMap[squareIndex] || 0;
      if (zVal > 0) {
        return getZPatternColor(zVal, isLight);
      }
      return isLight ? '#f0d9b5' : '#b58863';
    }
    
    // 默认：红色激活值高亮
    return getActivationColor(activation, isLight);
  };

  return (
    <div className="font-sans border border-gray-300 rounded-lg p-3 bg-gray-50 max-w-xs">
      <div className="flex justify-between items-center mb-2">
        <h4 className="text-gray-800 text-sm font-semibold">
          棋盘 {activations.length > 0 ? '(激活值)' : ''} {zPatternIndices.length > 0 ? '(Z Pattern)' : ''}
        </h4>
        <div className="flex gap-1 items-center">
          {sampleIndex !== undefined && (
            <div className="bg-blue-500 text-white px-2 py-1 rounded text-xs font-bold">
              #{sampleIndex}
            </div>
          )}
          {contextId !== undefined && (
            <div className="bg-green-500 text-white px-2 py-1 rounded text-xs font-bold">
              ID:{contextId}
            </div>
          )}
        </div>
      </div>
      
      {analysisName && (
        <div className="text-gray-600 text-xs mb-2">Analysis: {analysisName}</div>
      )}
      
      <div className="inline-block border-2 border-gray-700 mb-3">
        <table className="border-collapse text-xl">
          {/* 列标记 */}
          <tr>
            <td className="w-5 h-5 text-center text-xs text-gray-500"></td>
            {Array.from({ length: 8 }, (_, col) => (
              <td key={col} className="w-9 h-5 text-center text-xs text-gray-500">
                {String.fromCharCode(65 + col)}
              </td>
            ))}
          </tr>
          
          {/* 棋盘行 */}
          {Array.from({ length: 8 }, (_, row) => {
            const rowNumber = activeColor === 'b' ? (row + 1) : (8 - row);
            return (
              <tr key={row}>
                {/* 行号标记 */}
                <td className="w-5 h-9 text-center text-xs text-gray-500">
                  {rowNumber}
                </td>
                
                {/* 棋盘方格 */}
                {Array.from({ length: 8 }, (_, col) => {
                  // 激活值数组的索引：始终翻转，让激活值显示在行棋方这一侧
                  const activationIndex = (7 - row) * 8 + col;
                  const activation = boardActivations[activationIndex] || 0;
                  
                  // 棋盘位置：黑方行棋时只进行上下翻转，保持A列在最左边
                  let boardRow, boardCol;
                  if (activeColor === 'b') {
                    boardRow = 7 - row;  // 行翻转
                    boardCol = col;      // 列不翻转，保持A列在最左边
                  } else {
                    boardRow = row;
                    boardCol = col;
                  }
                  
                  const piece = board[boardRow][boardCol];
                  const pieceSymbol = pieceSymbols[piece] || '';
                  const textColor = piece && piece >= 'A' && piece <= 'Z' ? '#fff' : '#000';
                  const textShadow = piece && piece >= 'A' && piece <= 'Z' 
                    ? '1px 1px 2px rgba(0,0,0,0.8)' 
                    : '1px 1px 1px rgba(255,255,255,0.5)';
                  
                  const backgroundColor = getSquareBackgroundColor(row, col, activation, activationIndex);
                  
                  return (
                    <td
                      key={col}
                      className="w-9 h-9 text-center align-middle border border-gray-600 font-bold relative cursor-pointer transition-colors duration-300 overflow-hidden"
                      style={{
                        backgroundColor,
                        color: textColor,
                        textShadow
                      }}
                      onMouseEnter={() => {
                        // 只有当方格有激活值时才设置悬停状态
                        if (activation > 0) {
                          setHoveredSquare(activationIndex);
                        }
                      }}
                      onMouseLeave={() => {
                        setHoveredSquare(null);
                      }}
                    >
                      {pieceSymbol}
                      
                      {/* 激活值显示（非悬停状态下，固定6px并置于右上角）*/}
                      {hoveredSquare === null && activation > 0 && (
                        <div
                          style={{
                            position: 'absolute',
                            top: 1,
                            right: 1,
                            fontSize: 6,
                            lineHeight: '8px',
                            color: '#333',
                            backgroundColor: 'rgba(255,255,255,0.7)',
                            padding: '0px 1px',
                            borderRadius: 2,
                            pointerEvents: 'none'
                          }}
                        >
                          {activation.toFixed(3)}
                        </div>
                      )}
                      
                      {/* Z Pattern值提示（悬停映射处，固定6px并置于左下角）*/}
                      {hoveredSquare !== null && hoveredZMap[activationIndex] > 0 && (
                        <div
                          style={{
                            position: 'absolute',
                            bottom: 1,
                            left: 1,
                            fontSize: 6,
                            lineHeight: '8px',
                            color: '#374151',
                            backgroundColor: 'rgba(103, 232, 249, 0.7)', // cyan-300 70%
                            padding: '0px 1px',
                            borderRadius: 2,
                            pointerEvents: 'none'
                          }}
                        >
                          {`Z:${hoveredZMap[activationIndex].toFixed(3)}`}
                        </div>
                      )}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </table>
      </div>
      
      <div className="text-xs text-gray-600 space-y-2">
        {/* 激活值说明 */}
        {boardActivations.some(v => v > 0) && (
          <div className="p-2 bg-blue-50 rounded border-l-4 border-blue-400">
            <strong>激活范围:</strong> {Math.min(...boardActivations.filter(v => v > 0)).toFixed(3)} ~ {Math.max(...boardActivations).toFixed(3)}<br/>
            <small className="text-gray-500">红色：激活值强度（鼠标悬停时隐藏）</small>
          </div>
        )}

        {/* Z Pattern说明（以悬停映射为准）*/}
        {hoveredSquare !== null && hoveredZMap.some(v => v > 0) && (
          <div className="p-2 bg-cyan-50 rounded border-l-4 border-cyan-400">
            <strong>Z Pattern范围:</strong> {Math.min(...hoveredZMap.filter(v => v > 0)).toFixed(3)} ~ {Math.max(...hoveredZMap).toFixed(3)}<br/>
            <small className="text-gray-500">蓝色：悬停格子的Z Pattern强度</small>
          </div>
        )}
        
        {/* 游戏信息 */}
        <div>
          <strong>当前:</strong> {activeColor === 'w' ? '白方' : '黑方'} | <strong>回合:</strong> {fullmove}
        </div>
        
        {/* 显示完整FEN */}
        <div className="font-mono bg-blue-50 p-2 rounded border-l-4 border-blue-400 text-[9px] break-all leading-tight">
          <strong>FEN:</strong> {fen}
        </div>
        
        {/* 使用说明 */}
        <div className="p-2 bg-yellow-50 rounded border-l-4 border-yellow-400">
          <div className="text-yellow-700">
            <strong>💡 使用说明:</strong><br/>
            • 默认显示：红色高亮表示激活值强度<br/>
            • 鼠标悬停：清除红色高亮，仅显示该格子对应的Z Pattern蓝色高亮<br/>
            • 触发条件：只有当格子有激活值时才触发悬停映射<br/>
            • 鼠标移开：恢复红色激活值高亮
          </div>
        </div>
      </div>
    </div>
  );
}; 