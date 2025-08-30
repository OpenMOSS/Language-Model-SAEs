import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface ChessPosition {
  longfen: string;
  currentPlayer?: 'white' | 'black';
  enPassant?: string;
  castling?: {
    whiteKing: boolean;
    whiteQueen: boolean;
    blackKing: boolean;
    blackQueen: boolean;
  };
  currentMove?: string;
}

interface ChessBoardProps {
  position: ChessPosition;
}

const parseLongfen = (longfen: string) => {
  // 解析longfen字符串，提取棋盘位置和其他信息
  const parts = longfen.split('.');
  const board = parts[0];
  const moveCount = parts[1] || '0';
  const halfMoveCount = parts[2] || '0';
  const currentMove = parts[3] || '';
  
  // 将棋盘字符串转换为8x8的棋盘数组
  const boardArray: (string | null)[][] = Array(8).fill(null).map(() => Array(8).fill(null));
  
  let row = 0;
  let col = 0;
  
  for (const char of board) {
    if (char === '.') {
      row++;
      col = 0;
    } else if (char >= '1' && char <= '8') {
      col += parseInt(char);
    } else {
      if (row < 8 && col < 8) {
        boardArray[row][col] = char;
        col++;
      }
    }
  }
  
  return {
    board: boardArray,
    moveCount: parseInt(moveCount),
    halfMoveCount: parseInt(halfMoveCount),
    currentMove
  };
};

const getPieceSymbol = (piece: string): string => {
  const pieceMap: { [key: string]: string } = {
    'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
  };
  return pieceMap[piece] || '';
};

const getPieceColor = (piece: string): 'white' | 'black' => {
  return piece >= 'A' && piece <= 'Z' ? 'white' : 'black';
};

export const ChessBoard: React.FC<ChessBoardProps> = ({ position }) => {
  const { board, moveCount, halfMoveCount, currentMove } = parseLongfen(position.longfen);
  
  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <span>国际象棋棋盘</span>
          <Badge variant="outline">
            {position.currentPlayer === 'white' ? '白方走棋' : '黑方走棋'}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col gap-4">
          {/* 棋盘 */}
          <div className="grid grid-cols-8 gap-0 border-2 border-gray-800 w-fit mx-auto">
            {board.map((row, rowIndex) =>
              row.map((piece, colIndex) => {
                const isLight = (rowIndex + colIndex) % 2 === 0;
                const squareColor = isLight ? 'bg-amber-100' : 'bg-amber-600';
                
                return (
                  <div
                    key={`${rowIndex}-${colIndex}`}
                    className={`w-12 h-12 flex items-center justify-center ${squareColor} border border-gray-400`}
                  >
                    {piece && (
                      <span 
                        className={`text-2xl select-none ${
                          getPieceColor(piece) === 'white' ? 'text-white drop-shadow-lg' : 'text-black'
                        }`}
                        style={{ textShadow: getPieceColor(piece) === 'white' ? '1px 1px 2px rgba(0,0,0,0.8)' : 'none' }}
                      >
                        {getPieceSymbol(piece)}
                      </span>
                    )}
                  </div>
                );
              })
            )}
          </div>
          
          {/* 游戏信息 */}
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="font-medium">移动次数:</span>
                <span>{moveCount}</span>
              </div>
              <div className="flex justify-between">
                <span className="font-medium">半步计数:</span>
                <span>{halfMoveCount}</span>
              </div>
            </div>
            
            <div className="space-y-2">
              {position.enPassant && (
                <div className="flex justify-between">
                  <span className="font-medium">吃过路兵:</span>
                  <span>{position.enPassant}</span>
                </div>
              )}
              
              {position.castling && (
                <div className="flex justify-between">
                  <span className="font-medium">王车易位:</span>
                  <div className="flex gap-1">
                    {position.castling.whiteKing && <Badge variant="outline" className="text-xs">白王</Badge>}
                    {position.castling.whiteQueen && <Badge variant="outline" className="text-xs">白后</Badge>}
                    {position.castling.blackKing && <Badge variant="outline" className="text-xs">黑王</Badge>}
                    {position.castling.blackQueen && <Badge variant="outline" className="text-xs">黑后</Badge>}
                  </div>
                </div>
              )}
              
              {position.currentMove && (
                <div className="flex justify-between">
                  <span className="font-medium">当前思考:</span>
                  <span className="font-mono">{position.currentMove}</span>
                </div>
              )}
            </div>
          </div>
          
          {/* Longfen字符串 */}
          <div className="mt-4 p-2 bg-gray-100 rounded text-xs font-mono break-all">
            <div className="font-medium mb-1">Longfen:</div>
            <div>{position.longfen}</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}; 