import { ChessState, parseLongfen, isLightSquare } from "@/utils/chess";

export type ChessMove = {
  from: { row: number; col: number };
  to: { row: number; col: number };
  promotion?: string;
};

export type SimpleChessBoardProps = {
  fen?: string;
  longfen?: string;
  move?: ChessMove;
  className?: string;
  size?: number;
};

// 解析UCI移动格式 (如 "e2e4", "e7e8q")
function parseUciMove(uciMove: string): ChessMove | null {
  if (uciMove.length < 4) return null;
  
  const fromFile = uciMove.charCodeAt(0) - 97; // 'a' = 0
  const fromRank = 8 - parseInt(uciMove[1]); // '1' = 7, '8' = 0
  const toFile = uciMove.charCodeAt(2) - 97;
  const toRank = 8 - parseInt(uciMove[3]);
  
  const move: ChessMove = {
    from: { row: fromRank, col: fromFile },
    to: { row: toRank, col: toFile }
  };
  
  // 检查是否有升变
  if (uciMove.length === 5) {
    move.promotion = uciMove[4];
  }
  
  return move;
}

// 从longfen字符串中提取移动信息
function extractMoveFromLongfen(longfen: string): ChessMove | null {
  // longfen格式: ...move0，移动信息在倒数第5个字符开始
  if (longfen.length < 5) return null;
  
  const movePart = longfen.slice(-5);
  // 移除末尾的"0"
  const uciMove = movePart.replace(/0$/, '');
  
  return parseUciMove(uciMove);
}

export function createChessBoardElement({ 
  fen, 
  longfen, 
  move, 
  className = "", 
  size = 400 
}: SimpleChessBoardProps): HTMLElement {
  let chessState: ChessState | null = null;
  let displayMove: ChessMove | null = move || null;
  
  // 解析棋盘状态
  if (longfen) {
    chessState = parseLongfen(longfen);
  } else if (fen) {
    // 如果有FEN但没有longfen，尝试构造longfen
    const tempLongfen = fen + "........0";
    chessState = parseLongfen(tempLongfen);
  }
  
  // 如果没有提供move，尝试从longfen中提取
  if (!displayMove && longfen) {
    displayMove = extractMoveFromLongfen(longfen);
  }
  
  if (!chessState) {
    const errorDiv = document.createElement('div');
    errorDiv.className = `flex items-center justify-center border rounded ${className}`;
    errorDiv.style.width = `${size}px`;
    errorDiv.style.height = `${size}px`;
    errorDiv.innerHTML = '<span class="text-gray-500">无法解析棋盘状态</span>';
    return errorDiv;
  }
  
  const squareSize = size / 8;
  
  // 创建主容器
  const container = document.createElement('div');
  container.className = `relative border-2 border-gray-800 ${className}`;
  container.style.width = `${size}px`;
  container.style.height = `${size}px`;
  
  // 创建棋盘格子
  chessState.board.forEach((row, rowIndex) => {
    row.forEach((piece, colIndex) => {
      const isLight = isLightSquare(rowIndex, colIndex);
      const squareName = `${String.fromCharCode(97 + colIndex)}${8 - rowIndex}`;
      
      const square = document.createElement('div');
      square.className = `absolute flex items-center justify-center text-2xl font-bold border border-gray-400 ${
        isLight ? 'bg-amber-100' : 'bg-amber-800'
      }`;
      square.style.left = `${colIndex * squareSize}px`;
      square.style.top = `${rowIndex * squareSize}px`;
      square.style.width = `${squareSize}px`;
      square.style.height = `${squareSize}px`;
      
      if (piece) {
        const pieceSpan = document.createElement('span');
        pieceSpan.className = piece.color === 'white' ? 'text-white' : 'text-black';
        pieceSpan.textContent = piece.symbol;
        square.appendChild(pieceSpan);
      }
      
      // 显示格子坐标
      const coordSpan = document.createElement('span');
      coordSpan.className = 'absolute bottom-0 right-0 text-xs text-gray-600 opacity-50';
      coordSpan.textContent = squareName;
      square.appendChild(coordSpan);
      
      container.appendChild(square);
    });
  });
  
  // 创建移动箭头
  if (displayMove) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'absolute inset-0 pointer-events-none');
    svg.setAttribute('width', size.toString());
    svg.setAttribute('height', size.toString());
    
    // 创建箭头定义
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
    marker.setAttribute('id', 'arrowhead');
    marker.setAttribute('markerWidth', '10');
    marker.setAttribute('markerHeight', '7');
    marker.setAttribute('refX', '9');
    marker.setAttribute('refY', '3.5');
    marker.setAttribute('orient', 'auto');
    
    const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    polygon.setAttribute('points', '0 0, 10 3.5, 0 7');
    polygon.setAttribute('fill', '#ff4444');
    polygon.setAttribute('stroke', '#cc0000');
    polygon.setAttribute('stroke-width', '1');
    
    marker.appendChild(polygon);
    defs.appendChild(marker);
    svg.appendChild(defs);
    
    // 创建箭头线
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', (displayMove.from.col * squareSize + squareSize / 2).toString());
    line.setAttribute('y1', (displayMove.from.row * squareSize + squareSize / 2).toString());
    line.setAttribute('x2', (displayMove.to.col * squareSize + squareSize / 2).toString());
    line.setAttribute('y2', (displayMove.to.row * squareSize + squareSize / 2).toString());
    line.setAttribute('stroke', '#ff4444');
    line.setAttribute('stroke-width', '3');
    line.setAttribute('marker-end', 'url(#arrowhead)');
    line.setAttribute('opacity', '0.8');
    
    svg.appendChild(line);
    
    // 创建起点圆圈
    const fromCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    fromCircle.setAttribute('cx', (displayMove.from.col * squareSize + squareSize / 2).toString());
    fromCircle.setAttribute('cy', (displayMove.from.row * squareSize + squareSize / 2).toString());
    fromCircle.setAttribute('r', '8');
    fromCircle.setAttribute('fill', 'none');
    fromCircle.setAttribute('stroke', '#ff4444');
    fromCircle.setAttribute('stroke-width', '2');
    fromCircle.setAttribute('opacity', '0.8');
    
    svg.appendChild(fromCircle);
    
    // 创建终点圆圈
    const toCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    toCircle.setAttribute('cx', (displayMove.to.col * squareSize + squareSize / 2).toString());
    toCircle.setAttribute('cy', (displayMove.to.row * squareSize + squareSize / 2).toString());
    toCircle.setAttribute('r', '6');
    toCircle.setAttribute('fill', '#ff4444');
    toCircle.setAttribute('opacity', '0.6');
    
    svg.appendChild(toCircle);
    
    container.appendChild(svg);
    
    // 创建移动信息显示
    const moveInfo = document.createElement('div');
    moveInfo.className = 'absolute top-2 left-2 bg-black bg-opacity-75 text-white px-2 py-1 rounded text-sm';
    moveInfo.textContent = `${String.fromCharCode(97 + displayMove.from.col)}${8 - displayMove.from.row}${String.fromCharCode(97 + displayMove.to.col)}${8 - displayMove.to.row}${displayMove.promotion || ''}`;
    
    container.appendChild(moveInfo);
  }
  
  return container;
} 