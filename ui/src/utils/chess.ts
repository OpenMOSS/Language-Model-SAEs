/**
 * Chess utility functions for parsing longfen strings and handling chess-related logic
 */

// 定义棋子类型
export type ChessPiece = {
  type: 'pawn' | 'rook' | 'knight' | 'bishop' | 'queen' | 'king';
  color: 'white' | 'black';
  symbol: string;
};

// 定义棋盘位置
export type ChessPosition = {
  row: number;
  col: number;
};

// 定义棋盘状态
export type ChessState = {
  board: (ChessPiece | null)[][];
  activeColor: 'white' | 'black';
  castlingRights: {
    whiteKingSide: boolean;
    whiteQueenSide: boolean;
    blackKingSide: boolean;
    blackQueenSide: boolean;
  };
  enPassantTarget: ChessPosition | null;
  halfmoveClock: number;
  fullmoveNumber: number;
  currentThinkingMove?: string;
};

// 棋子符号映射
const PIECE_SYMBOLS: Record<string, ChessPiece> = {
  'P': { type: 'pawn', color: 'white', symbol: '♙' },
  'R': { type: 'rook', color: 'white', symbol: '♖' },
  'N': { type: 'knight', color: 'white', symbol: '♘' },
  'B': { type: 'bishop', color: 'white', symbol: '♗' },
  'Q': { type: 'queen', color: 'white', symbol: '♕' },
  'K': { type: 'king', color: 'white', symbol: '♔' },
  'p': { type: 'pawn', color: 'black', symbol: '♟' },
  'r': { type: 'rook', color: 'black', symbol: '♜' },
  'n': { type: 'knight', color: 'black', symbol: '♞' },
  'b': { type: 'bishop', color: 'black', symbol: '♝' },
  'q': { type: 'queen', color: 'black', symbol: '♛' },
  'k': { type: 'king', color: 'black', symbol: '♚' },
};

/**
 * Parse a longfen string into a chess state
 * @param longfen - The longfen string to parse
 * @returns Parsed chess state
 */
export function parseLongfen(longfen: string): ChessState {
  // 去除末尾的数字部分（如果有）
  const cleanLongfen = longfen.replace(/\d+$/, '');
  
  // 分割longfen字符串的各个部分
  const parts = cleanLongfen.split('.');
  
  // 初始化8x8棋盘
  const board: (ChessPiece | null)[][] = Array(8).fill(null).map(() => Array(8).fill(null));
  
  // 解析棋盘部分
  const boardPart = parts.slice(0, 8).join('.');
  let position = 0;
  
  for (let i = 0; i < boardPart.length; i++) {
    const char = boardPart[i];
    
    if (char === '.') {
      continue; // 跳过分隔符
    }
    
    if (char in PIECE_SYMBOLS) {
      const row = Math.floor(position / 8);
      const col = position % 8;
      if (row < 8 && col < 8) {
        board[row][col] = PIECE_SYMBOLS[char];
      }
      position++;
    } else if (char >= '1' && char <= '8') {
      // 空格数量
      position += parseInt(char);
    }
  }
  
  // 解析其他信息
  const activeColor = parts[8] === 'b' ? 'black' : 'white';
  const castlingRights = parseCastlingRights(parts[9] || '');
  const enPassantTarget = parseEnPassantTarget(parts[10] || '');
  const halfmoveClock = parseInt(parts[11] || '0');
  const fullmoveNumber = parseInt(parts[12] || '1');
  
  return {
    board,
    activeColor,
    castlingRights,
    enPassantTarget,
    halfmoveClock,
    fullmoveNumber,
  };
}

/**
 * Parse castling rights from the castling part of longfen
 */
function parseCastlingRights(castling: string): ChessState['castlingRights'] {
  return {
    whiteKingSide: castling.includes('K'),
    whiteQueenSide: castling.includes('Q'),
    blackKingSide: castling.includes('k'),
    blackQueenSide: castling.includes('q'),
  };
}

/**
 * Parse en passant target square
 */
function parseEnPassantTarget(enPassant: string): ChessPosition | null {
  if (!enPassant || enPassant === '-') {
    return null;
  }
  
  const col = enPassant.charCodeAt(0) - 'a'.charCodeAt(0);
  const row = 8 - parseInt(enPassant.charAt(1));
  
  return { row, col };
}

/**
 * Convert chess position to algebraic notation
 */
export function positionToAlgebraic(pos: ChessPosition): string {
  const file = String.fromCharCode('a'.charCodeAt(0) + pos.col);
  const rank = (8 - pos.row).toString();
  return file + rank;
}

/**
 * Check if a square is light or dark
 */
export function isLightSquare(row: number, col: number): boolean {
  return (row + col) % 2 === 0;
} 