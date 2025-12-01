// Chess Board Visualization using D3.js
// Based on chess-board.tsx functionality

const PIECE_SYMBOLS = {
    'K': '♔', // White King
    'Q': '♕', // White Queen
    'R': '♖', // White Rook
    'B': '♗', // White Bishop
    'N': '♘', // White Knight
    'P': '♙', // White Pawn
    'k': '♚', // Black King
    'q': '♛', // Black Queen
    'r': '♜', // Black Rook
    'b': '♝', // Black Bishop
    'n': '♞', // Black Knight
    'p': '♟', // Black Pawn
  };
  
  // Chess board state
  let boardState = {
    fen: '2k5/4Q3/3P4/8/6p1/4p3/q1pbK3/1R6 b - - 0 32',
    flipped: false,
    showCoordinates: true,
    showActivations: false,
    selectedSquare: null,
    hoveredSquare: null,
    activations: null,
    zPatternIndices: null,
    zPatternValues: null
  };
  
  // board size
  const BOARD_SIZE = 400;
  const SQUARE_SIZE = BOARD_SIZE / 8;
  
  // global SVG offset - control the position of all SVG elements
  const GLOBAL_OFFSET_X = 150;  // move to the right
  const GLOBAL_OFFSET_Y = 1525;  // move down
  
  // board transformation parameters - all functions use the same values
  const BOARD_MARGIN = 50;           // basic inner margin at top/left
  const BOARD_OFFSET_X = 120;        // extra offset on the horizontal direction of the board
  const BOARD_SKEW_ANGLE = -10;      // plane skew angle (deg), negative value to tilt left, positive value to tilt right
  
    // node and edge parameters
  const BOARD_PERSPECTIVE = 800;     // perspective distance
  const BOARD_ROTATE_X = 60;         // rotate around the X axis
  const LAYER_HEIGHT_FACTOR = 1.0;   // node relative to Embedding layer height factor (can be lowered to reduce height)
  const EMBEDDING_OFFSET_X = -20;     // red dot screen pixel -> graph internal coordinates: X adjustment
  const EMBEDDING_OFFSET_Y = -40;    // red dot screen pixel -> graph internal coordinates: Y adjustment
  const TEXT_BOX_DISTANCE = 8;      // distance between text and box (pixels)
  const NODE_WIDTH_FACTOR = 0.6;    // node width relative to square size
  const NODE_HEIGHT_FACTOR = 0.4;   // node height relative to square size
  const SHOW_VIRTUAL_EDGES = true;  // whether to show virtual edges
  const CIRCLE_RADIUS = 3;          // radius of small circles inside node boxes
  const CIRCLE_SPACING = 8;         // spacing between circles
  const MAX_CIRCLE_WIDTH = 40;      // maximum width for circles before stacking

  let redDotScreenCoords = new Map(); // squareIndex -> {x, y}
  let layerMapping = new Map(); // original layer -> display layer mapping
  
  // get the actual screen pixel coordinates of the red dot
  function getRedDotScreenCoords(squareIndex) {
  const coords = redDotScreenCoords.get(squareIndex) || { x: 0, y: 0 };
//   return { x: coords.x, y: coords.y};
  return { x: coords.x + EMBEDDING_OFFSET_X, y: coords.y + EMBEDDING_OFFSET_Y };
  }
  
  // convert screen coordinates to coordinates relative to the graph container
  function convertScreenToGraphCoords(screenCoords) {
  return {
    x: screenCoords.x - GLOBAL_OFFSET_X,
    y: screenCoords.y - GLOBAL_OFFSET_Y
  };
  }

  // create layer mapping from original layers to display layers
  function createLayerMapping(graph) {
    // get all unique layers (excluding embedding nodes with layer 0)
    const originalLayers = [...new Set(graph.nodes
      .filter(n => n.type !== 'embedding' && n.layer !== 0)
      .map(n => n.layer)
    )].sort((a, b) => a - b);
    
    // create mapping: original layer -> display layer (1-based)
    layerMapping.clear();
    originalLayers.forEach((originalLayer, index) => {
      layerMapping.set(originalLayer, index + 1);
    });
    
    return layerMapping;
  }

  // get display layer for a node
  function getDisplayLayer(node) {
    if (node.type === 'embedding' || node.layer === 0) {
      return 0; // embedding nodes stay at layer 0
    }
    
    // Check if we should use custom display layer
    if (typeof USE_CUSTOM_DISPLAY_LAYER !== 'undefined' && USE_CUSTOM_DISPLAY_LAYER && node.display_layer !== null) {
      return node.display_layer;
    }
    
    // Use auto-sorted layer mapping
    return layerMapping.get(node.layer) || node.layer;
  }
  
  // coordinate transformation function
  const CoordinateTransforms = {
  // 1. board coordinates (x, 0, z) - completely vertical coordinates
  getBoardCoords: (squareIndex) => {
    const row = Math.floor(squareIndex / 8);
    const col = squareIndex % 8;
    const displayRow = boardState.flipped ? (7 - row) : row;
    
    const x = col * SQUARE_SIZE + SQUARE_SIZE / 2;
    const z = displayRow * SQUARE_SIZE + SQUARE_SIZE / 2;
    
    return { x, y: 0, z };
  },
  
  // 2. 3D coordinates (x', y', z') - coordinates after rotating around the X axis by 60 degrees
  get3DCoords: (boardCoords) => {
    const rotateXAngle = BOARD_ROTATE_X * Math.PI / 180;
    const cosX = Math.cos(rotateXAngle);
    const sinX = Math.sin(rotateXAngle);
    
    const x_prime = boardCoords.x;
    const y_prime = boardCoords.y * cosX - boardCoords.z * sinX;
    const z_prime = boardCoords.y * sinX + boardCoords.z * cosX;
    
    return { x: x_prime, y: y_prime, z: z_prime };
  },
  
  // 3. page display coordinates (X, Y) - the final display coordinates mapped according to the perspective principle
  getDisplayCoords: (coords3D) => {
    const margin = BOARD_MARGIN;
    const offsetX = BOARD_OFFSET_X;
    const skewAngle = BOARD_SKEW_ANGLE; // use the global skew angle
    
    // apply perspective projection and skew transformation
    const tanA = Math.tan(skewAngle * Math.PI / 180);
    
    // first apply perspective projection, then apply skew transformation
    const perspectiveX = coords3D.x;
    const perspectiveY = coords3D.y;
    
    // apply skew transformation: add tan(skewAngle) * y to the x direction
    const skewedX = perspectiveX + tanA * perspectiveY;
    const skewedY = perspectiveY;
    
    // add the offset
    const X = GLOBAL_OFFSET_X + margin + offsetX + skewedX;
    const Y = GLOBAL_OFFSET_Y + margin + skewedY;
    
    return { X, Y };
  },
  
  // complete conversion: from board index to final display coordinates
  getFullTransform: (squareIndex) => {
    const boardCoords = CoordinateTransforms.getBoardCoords(squareIndex);
    const coords3D = CoordinateTransforms.get3DCoords(boardCoords);
    const displayCoords = CoordinateTransforms.getDisplayCoords(coords3D);
    
    return {
      board: boardCoords,
      coords3D: coords3D,
      display: displayCoords
    };
  }
  };
    
  // parse the FEN string
  function parseFEN(fen) {
    const parts = fen.trim().split(' ');
    if (parts.length < 4) {
        throw new Error('Invalid FEN string');
    }
  
    const [boardPart, activeColor, castling, enPassant] = parts;
    const rows = boardPart.split('/');
  
    if (rows.length !== 8) {
        throw new Error('Invalid board configuration in FEN');
    }
  
    // convert the FEN to an 8x8 array
    const board = [];
    
    for (let i = 0; i < 8; i++) {
        const row = [];
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
  }
  
  // convert the row and column coordinates to a linear index (0-63)
  function getSquareIndex(row, col) {
    return row * 8 + col;
  }

  function createChessBoard() {
  const container = d3.select('#visualization');
  container.selectAll('.chess-board-svg').remove();
  const svg = container
      .append('svg')
      .attr('class', 'chess-board-svg')
      .attr('width', BOARD_SIZE + 300) // leave more horizontal space in case skew exceeds
      .attr('height', BOARD_SIZE + 140)
      .attr('transform', `translate(${GLOBAL_OFFSET_X}, ${GLOBAL_OFFSET_Y})`)
      .style('border-radius', '8px')
      .style('background', '#f8f9fa')
      .style('opacity', '0.4') // make the board slightly transparent to see the graph behind
      .style('transform', `translate(${GLOBAL_OFFSET_X}px, ${GLOBAL_OFFSET_Y}px) perspective(${BOARD_PERSPECTIVE}px) rotateX(${BOARD_ROTATE_X}deg)`)
      .style('overflow', 'visible');
  
  // use the global board transformation parameters
  const margin = BOARD_MARGIN;
  const offsetX = BOARD_OFFSET_X;
  const skewAngle = BOARD_SKEW_ANGLE;
  const rowLabelOffsetX = -28;    // row number in the local coordinate system of the board (negative value to the left)
  const colLabelYOffset = 18;     // vertical offset from the column label to the bottom of the board (pixels)
  // end params
  
  // pre-calculate: tan value used to project the local coordinates onto the screen
  const tanA = Math.tan(skewAngle * Math.PI / 180);
  
  // labelsGroup is kept (if you want to put the column labels in a group), but we will use absolute coordinates to calculate the position of the column labels
  const labelsGroup = svg.append('g'); // do not set transform (we will add margin/offsetX when calculating coordinates)
  
  // skewedBoardGroup: the main body of the board (all placed here, ensuring that it is affected by skew)
  const skewedBoardGroup = svg.append('g')
      .attr('transform', `translate(${margin + offsetX}, ${margin}) skewX(${skewAngle})`);
  
  // draw a rectangle border with skewedBoardGroup (will deform together with skew)
  skewedBoardGroup.append('rect')
      .attr('x', -6)
      .attr('y', -6)
      .attr('width', BOARD_SIZE + 12)
      .attr('height', BOARD_SIZE + 12)
      .style('fill', 'none')
      .style('stroke', globalColors.deepBlue)
      .style('stroke-width', 4)
      .style('rx', 8);
  
  // parse the FEN and determine the display order
  const parsedBoard = parseFEN(boardState.fen);
  const { board } = parsedBoard;
  const displayBoard = boardState.flipped ? [...board].reverse() : board;
  container.style('position', 'relative');
  // create squares (placed in skewedBoardGroup)
  const squares = skewedBoardGroup.selectAll('.square')
      .data(d3.range(64))
      .enter()
      .append('g')
      .attr('class', 'square')
      .attr('transform', (d) => {
          const row = Math.floor(d / 8);
          const col = d % 8;
          const displayRow = boardState.flipped ? (7 - row) : row;

          const transforms = CoordinateTransforms.getFullTransform(d);
          
          // store the final display coordinates
        //   squareCenterCoords.set(d, { x: transforms.display.X, y: transforms.display.Y });
        //   console.log(`Square ${d} 显示坐标: x=${transforms.display.X.toFixed(1)}, y=${transforms.display.Y.toFixed(1)}`);
          
          // keep the original board rendering logic
          return `translate(${col * SQUARE_SIZE}, ${displayRow * SQUARE_SIZE})`;
      });
  
  // add square background and interaction
  squares.append('rect')
      .attr('width', SQUARE_SIZE)
      .attr('height', SQUARE_SIZE)
      .attr('class', 'chess-square')
      .style('fill', (d) => {
          const row = Math.floor(d / 8);
          const col = d % 8;
          return (row + col) % 2 === 0 ? globalColors.lightOrange : globalColors.orange;
      })
      .style('stroke', 'none')
      .on('mouseenter', function(event, d) {
          if (boardState.activations && boardState.activations[d] !== 0) {
              boardState.hoveredSquare = d;
              updateBoard();
          }
      })
      .on('mouseleave', function(event, d) {
          boardState.hoveredSquare = null;
          updateBoard();
      })
      .on('click', function(event, d) {
          if (boardState.selectedSquare === null) {
              boardState.selectedSquare = d;
          } else if (boardState.selectedSquare === d) {
              boardState.selectedSquare = null;
          } else {
              const fromSquare = boardState.selectedSquare;
              const toSquare = d;
              boardState.selectedSquare = null;
          }
          updateBoard();
      });
  
  // Red dots removed - coordinates are calculated directly from square positions
  squares.each(function(d) {
    // Calculate coordinates directly from square position without visual red dots
    setTimeout(() => {
      const element = this;
      const rect = element.getBoundingClientRect();
      const screenX = rect.left + rect.width / 2;
      const screenY = rect.top + rect.height / 2;
      
      // Save the actual screen pixel coordinates for embedding node positioning
      redDotScreenCoords.set(d, { x: screenX, y: screenY });
    }, 50);
  });
  
  
    // add chess pieces (placed in each square)
  squares.each(function(d) {
      const row = Math.floor(d / 8);
      const col = d % 8;
      const displayRow = boardState.flipped ? (7 - row) : row;
      const piece = displayBoard[displayRow][col];
  
      if (piece) {
          d3.select(this)
              .append('text')
              .attr('x', SQUARE_SIZE / 2)
              .attr('y', SQUARE_SIZE / 2)
              .attr('text-anchor', 'middle')
              .attr('dominant-baseline', 'middle')
              .attr('font-size', SQUARE_SIZE * 0.6)
              .attr('font-family', 'serif')
              .style('filter', 'drop-shadow(2px 2px 3px rgba(0,0,0,0.5))')
              .style('fill', piece === piece.toLowerCase() ? '#000' : '#fff')
              .style('text-shadow', piece === piece.toLowerCase() ? '1px 1px 1px rgba(255,255,255,0.8)' : '1px 1px 1px rgba(0,0,0,0.8)')
              .text(PIECE_SYMBOLS[piece] || piece);
      }
  });
  
  // coordinate labels:
  if (boardState.showCoordinates) {
      // row number (1-8) — placed in skewedBoardGroup (affected by the skew of the board), and push the text out of the border to the left (rowLabelOffsetX)
      for (let i = 0; i < 8; i++) {
          const displayRow = boardState.flipped ? i : (7 - i);
          const x = rowLabelOffsetX; // for example -28, control the distance between the text and the border
          const y = i * SQUARE_SIZE + SQUARE_SIZE / 2;
  
          skewedBoardGroup.append('text')
              .attr('x', x)
              .attr('y', y)
              .attr('text-anchor', 'middle')
              .attr('dominant-baseline', 'middle')
              .attr('font-size', '12px')
              .attr('font-weight', 'bold')
              .style('fill', globalColors.deepBlue)
              .text(displayRow + 1);
      }
  
      // column number (a-h) — need to calculate the actual screen x coordinates according to the skew projection, the column labels remain horizontal (not affected by skew)
      // calculate the basic offset (translate of skewedBoardGroup)
      const baseTx = margin + offsetX;
      const baseTy = margin;
      const colY_localForBottom = BOARD_SIZE; // we regard the column labels as sticking to the bottom of the board (local y = BOARD_SIZE)
      const screenY = baseTy + colY_localForBottom + colLabelYOffset;
  
      for (let col = 0; col < 8; col++) {
          // local center of the column (in the local coordinate system of the board)
          const localX = col * SQUARE_SIZE + SQUARE_SIZE / 2;
          const localY = colY_localForBottom; // the y of the bottom row (used for skew influence calculation)
  
          // skewX mapping: screenX = baseTx + (localX + tanA * localY)
          const screenX = baseTx + (localX + tanA * localY);
  
          labelsGroup.append('text')
              .attr('x', screenX)
              .attr('y', screenY)
              .attr('text-anchor', 'middle')
              .attr('dominant-baseline', 'middle')
              .attr('font-size', '12px')
              .attr('font-weight', 'bold')
              .style('fill', globalColors.deepBlue)
              .text(String.fromCharCode(97 + (boardState.flipped ? (7 - col) : col)));
      }
  }
  
  // activate overlay (if enabled)
  if (boardState.showActivations && boardState.activations) {
      squares.each(function(d) {
          const activation = boardState.activations[d] || 0;
          if (activation !== 0) {
              const activationColor = getActivationColor(activation);
              d3.select(this)
                  .append('rect')
                  .attr('width', SQUARE_SIZE)
                  .attr('height', SQUARE_SIZE)
                  .attr('class', 'activation-overlay')
                  .style('fill', activationColor)
                  .style('pointer-events', 'none');
  
              d3.select(this)
                  .append('text')
                  .attr('x', SQUARE_SIZE - 5)
                  .attr('y', 15)
                  .attr('text-anchor', 'end')
                  .attr('font-size', '10px')
                  .attr('font-weight', 'bold')
                  .style('fill', '#fff')
                  .style('text-shadow', '1px 1px 1px rgba(0,0,0,0.8)')
                  .text(Math.abs(activation).toFixed(2));
          }
      });
  }
  
  // selected box (placed in skewedBoardGroup)
  if (boardState.selectedSquare !== null) {
      const selectedRow = Math.floor(boardState.selectedSquare / 8);
      const selectedCol = boardState.selectedSquare % 8;
      const displayRow = boardState.flipped ? (7 - selectedRow) : selectedRow;
      skewedBoardGroup.append('rect')
          .attr('x', selectedCol * SQUARE_SIZE)
          .attr('y', displayRow * SQUARE_SIZE)
          .attr('width', SQUARE_SIZE)
          .attr('height', SQUARE_SIZE)
          .style('fill', 'none')
          .style('stroke', '#007bff')
          .style('stroke-width', 3)
          .style('pointer-events', 'none');
  }

  }
  
  
  function renderGraphNodes(svg, graph) {
  // create layer mapping first
  createLayerMapping(graph);
  
  // create a completely independent container,不受棋盘 svg 的 transform 影响
  const container = d3.select('#visualization');
  
  // use the global offset
  
  const graphContainer = container.append('div')
    .style('position', 'absolute')
    .style('top', GLOBAL_OFFSET_Y + 'px')
    .style('left', GLOBAL_OFFSET_X + 'px')
    .style('pointer-events', 'none')
    .style('z-index', '10');
  
  const graphSvg = graphContainer.append('svg')
    .attr('class', 'graph-svg')
    .attr('width', BOARD_SIZE + 300)
    .attr('height', BOARD_SIZE + 140)
    .style('background', 'transparent');
  
  // dynamically extend the canvas height and move the node layers down as a whole to prevent the high-level nodes from being clipped
  const allNonEmbeddingLayers = graph.nodes
    .filter(n => n.type !== 'embedding' && typeof n.layer === 'number' && !isNaN(n.layer))
    .map(n => n.layer);
  const maxLayer = allNonEmbeddingLayers.length ? Math.max(...allNonEmbeddingLayers) : 0;
  const layerHeightPx = SQUARE_SIZE * LAYER_HEIGHT_FACTOR;
  const extraTop = (maxLayer + 2) * layerHeightPx; // extra space for one more layer
  graphSvg.attr('height', BOARD_SIZE + 140 + extraTop).style('overflow', 'visible');

  // Create separate layers for edges and nodes
  const edgeLayer = graphSvg.append('g')
    .attr('class', 'graph-edges');
  
  const nodeLayer = graphSvg.append('g')
    .attr('class', 'graph-nodes')
    // .attr('transform', `translate(0, ${extraTop})`);
  
  // render edges first (so they appear behind nodes)
  // debug: print all edges and category counts
  try {
    const allEdgesBrief = graph.edges.map(e => ({ source: e.sourceId, target: e.targetId, weight: e.weight }));
    const edgeStats = (() => {
      let total = graph.edges.length;
      let embToX = 0, xToEmb = 0, xToX = 0;
      for (const e of graph.edges) {
        const s = graph.getNodeById(e.sourceId);
        const t = graph.getNodeById(e.targetId);
        if (!s || !t) continue;
        const sEmb = s.type === 'embedding';
        const tEmb = t.type === 'embedding';
        if (sEmb && !tEmb) embToX++;
        else if (!sEmb && tEmb) xToEmb++;
        else if (!sEmb && !tEmb) xToX++;
      }
      return { total, embToX, xToEmb, xToX };
    })();
  } catch (err) {
    // Error handling for edge information
  }

  // auxiliary: calculate the node rectangle position (non-embedding) and anchor point(s)
  const NODE_WIDTH_PX = SQUARE_SIZE * NODE_WIDTH_FACTOR;
  const NODE_HEIGHT_PX = SQUARE_SIZE * NODE_HEIGHT_FACTOR;
  const LAYER_STEP_PX = SQUARE_SIZE * LAYER_HEIGHT_FACTOR;

  function getNodeRectFor(node) {
    if (!node) return null;
    if (node.type === 'embedding') {
      const dot = convertScreenToGraphCoords(getRedDotScreenCoords(node.position));
      return {
        isEmbedding: true,
        centerX: dot.x,
        topY: dot.y,
        bottomY: dot.y,
        x: dot.x,
        y: dot.y,
        width: 0,
        height: 0,
      };
    }
    const dot = convertScreenToGraphCoords(getRedDotScreenCoords(node.position));
    const displayLayer = getDisplayLayer(node);
    const topY = dot.y - (displayLayer + 1) * LAYER_STEP_PX;
    const x = dot.x - NODE_WIDTH_PX / 2;
    return {
      isEmbedding: false,
      centerX: dot.x,
      topY,
      bottomY: topY + NODE_HEIGHT_PX + TEXT_BOX_DISTANCE + 10, // 10 is approximate font height
      x,
      y: topY,
      width: NODE_WIDTH_PX,
      height: NODE_HEIGHT_PX,
    };
  }

  graph.edges.forEach(edge => {
    const source = graph.getNodeById(edge.sourceId);
    const target = graph.getNodeById(edge.targetId);
    if (!source || !target) return;
  
    let x1, y1, x2, y2;
    const srcRect = getNodeRectFor(source);
    const tgtRect = getNodeRectFor(target);

    // first set the default anchor points according to the embedding situation, avoid being overwritten later
    if (source.type === 'embedding') {
      x1 = srcRect.centerX;
      y1 = srcRect.bottomY; // the center of the red dot
    } else {
      x1 = srcRect.centerX;
      y1 = srcRect.bottomY; // non-embedding default from the bottom
    }

    if (target.type === 'embedding') {
      x2 = tgtRect.centerX;
      y2 = tgtRect.topY; // the center of the red dot
    } else {
      x2 = tgtRect.centerX;
      y2 = tgtRect.topY; // non-embedding default to the top
    }

    // according to the layer relationship, low-level top -> high-level bottom; high-level bottom -> low-level top
    if (source.type !== 'embedding' && target.type !== 'embedding') {
      const sL = getDisplayLayer(source);
      const tL = getDisplayLayer(target);
      if (sL < tL) {
        y1 = srcRect.topY;
        y2 = tgtRect.bottomY;
      } else if (sL > tL) {
        y1 = srcRect.bottomY;
        y2 = tgtRect.topY;
      } else {
        y1 = srcRect.bottomY;
        y2 = tgtRect.topY;
      }
    } else if (source.type !== 'embedding' && target.type === 'embedding') {
      y1 = srcRect.bottomY;
      y2 = tgtRect.topY;
    } else if (source.type === 'embedding' && target.type !== 'embedding') {
      y1 = srcRect.topY;
      y2 = tgtRect.bottomY;
    }

    // 根据边的类型设置不同的样式
    const isVirtual = edge.isVirtual;
    const lineElement = edgeLayer.append('line')
      .attr('x1', x1)
      .attr('y1', y1)
      .attr('x2', x2)
      .attr('y2', y2)
      .style('stroke', globalColors.lightBlue)  // light blue color
      .style('stroke-opacity', 0.7)  // 70% opacity
      .style('stroke-width', 3)
      .style('stroke-linecap', 'round');

    // 如果是虚拟边，设置为虚线
    if (isVirtual && SHOW_VIRTUAL_EDGES) {
      lineElement.style('stroke-dasharray', '5,5');
    }
  });
  
  // Print display layer for each node before rendering
  console.log('=== Node Display Layers ===');
  graph.nodes.forEach(node => {
    const displayLayer = getDisplayLayer(node);
    console.log(`Node ${node.id}: original layer=${node.layer}, display_layer=${node.display_layer}, final display layer=${displayLayer}, type=${node.type}`);
  });
  console.log('========================');

  graph.nodes.forEach(node => {
    if (node.type === 'embedding') return;
  
    // get the display layer for the node
    const displayLayer = getDisplayLayer(node);
    
    // check if the layer is a valid number
    if (isNaN(displayLayer) || displayLayer === undefined) {
      return;
    }
    
    const layerHeight = SQUARE_SIZE * LAYER_HEIGHT_FACTOR; // the distance between each layer
    
    // Calculate dynamic width based on children count
    const childrenCount = node.childrenIds.length;
    const baseWidth = SQUARE_SIZE * NODE_WIDTH_FACTOR;
    let dynamicWidth;
    
    if (childrenCount === 0) {
      dynamicWidth = baseWidth;
    } else {
      // Calculate required width for all circles in one row
      const requiredWidth = (childrenCount - 1) * CIRCLE_SPACING + 2 * CIRCLE_RADIUS + 8; // 8px padding
      dynamicWidth = Math.max(requiredWidth, baseWidth); // ensure minimum width
      // No maximum width limit - allow circles to overlap if needed
    }
    
    const { x, y } = (() => {
      // get the red dot coordinates of the corresponding square (the position of the embedding node)
      const squareCoords = getRedDotScreenCoords(node.position);
      const convertedCoords = convertScreenToGraphCoords(squareCoords);
      
      return {
        x: convertedCoords.x - dynamicWidth/2,
        y: convertedCoords.y - (displayLayer + 1) * layerHeight // on the corresponding square above
      };
    })();

    // rectangle with background to hide edges behind it
    const rect = nodeLayer.append('rect')
      .attr('x', x)
      .attr('y', y)
      .attr('width', dynamicWidth)
      .attr('height', NODE_HEIGHT_PX)
      .style('fill', '#ffffff')  // white background to hide edges
      .style('stroke', globalColors.deepBlue)
      .style('stroke-width', 3)
      .style('rx', 6);

    // Add small circles inside the rectangle
    if (childrenCount > 0) {
      const circlesGroup = nodeLayer.append('g')
        .attr('class', `circles-${node.id}`);
      
      // Calculate total width needed for all circles
      const totalCircleWidth = (childrenCount - 1) * CIRCLE_SPACING + 2 * CIRCLE_RADIUS;
      
      // Calculate starting position to center the circles in the rectangle
      const startX = x + (dynamicWidth - totalCircleWidth) / 2;
      const centerY = y + NODE_HEIGHT_PX / 2;
      
      // Get all feature weights to determine color range
      const weights = node.features.map(f => f.weight);
      const minWeight = Math.min(...weights);
      const maxWeight = Math.max(...weights);
      const weightRange = maxWeight - minWeight;
      
      for (let i = 0; i < childrenCount; i++) {
        const circleX = startX + i * CIRCLE_SPACING + CIRCLE_RADIUS;
        const feature = node.features[i];
        
        // Calculate green intensity based on weight (0-1 range)
        let greenIntensity = 0;
        if (weightRange > 0) {
          greenIntensity = (feature.weight - minWeight) / weightRange;
        } else {
          greenIntensity = 1; // if all weights are the same, use full intensity
        }
        
        // Convert to 0-255 range and create green color
        const greenValue = Math.round(greenIntensity * 255);
        const fillColor = `rgb(0, ${greenValue}, 0)`;
        
        circlesGroup.append('circle')
          .attr('cx', circleX)
          .attr('cy', centerY)
          .attr('r', CIRCLE_RADIUS)
          .style('fill', fillColor)
          .style('stroke', globalColors.deepBlue)
          .style('stroke-width', 1.5)
          .style('pointer-events', 'all')
          .on('mouseenter', function() {
            d3.select(this).style('stroke', '#ff0000'); // red on hover
          })
          .on('mouseleave', function() {
            d3.select(this).style('stroke', globalColors.deepBlue); // back to original
          });
      }
    }

    // Add small square for logit nodes
    if (node.type === 'logit') {
      const squareSize = CIRCLE_RADIUS * 2;
      const squareX = x + (dynamicWidth - squareSize) / 2;
      const squareY = y + (NODE_HEIGHT_PX - squareSize) / 2;
      
      nodeLayer.append('rect')
        .attr('x', squareX)
        .attr('y', squareY)
        .attr('width', squareSize)
        .attr('height', squareSize)
        .style('fill', 'none')
        .style('stroke', globalColors.deepBlue)
        .style('stroke-width', 1.5)
        .style('pointer-events', 'all')
        .on('mouseenter', function() {
          d3.select(this).style('stroke', '#ff0000'); // red on hover
        })
        .on('mouseleave', function() {
          d3.select(this).style('stroke', globalColors.deepBlue); // back to original
        });
    }

    // clerp text - placed below the rectangle
    nodeLayer.append('text')
      .attr('x', x + dynamicWidth/2)
      .attr('y', y + NODE_HEIGHT_PX + TEXT_BOX_DISTANCE) // below the rectangle, using the global distance variable
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'hanging')
      .attr('font-size', '10px')
      .style('fill', globalColors.deepBlue)
      .text(node.clerp);
  });
  
  // keep the graph container completely opaque
  graphContainer.style('opacity', '1');
  
  // store the reference of the graph container, so that we can control the opacity later
  window.graphContainer = graphContainer;
  }
  
  // update the board
  function updateBoard() {
    createChessBoard();
  }
  
  /**
  * 初始化并渲染 graph 节点与边，不受棋盘 skew 变化影响，
  * 节点矩形显示在对应棋盘格上方（矩形下边与该格上边对齐），
  * 同时在中央显示 clerp 文本，子节点在棋盘格中心以圆形显示。
  *
  * @param {Object} graph - 全局图对象，包含 nodes 与 edges
  */
  function initializeGraph(graph) {
  // get the created svg object from the current page
  const svg = d3.select('#visualization svg');
  if (svg.empty()) {
    return;
  }
  
  // delay rendering the graph nodes, ensure the red dot coordinates have been obtained
  setTimeout(() => {
    renderGraphNodes(svg, graph);
  }, 200);
  }
  
  // initialize
  function initializeBoard() {
    // create the board
    createChessBoard();
    
    // bind the control button event
    d3.select('#flip-board').on('click', function() {
        boardState.flipped = !boardState.flipped;
        updateBoard();
    });
    
    d3.select('#toggle-coordinates').on('click', function() {
        boardState.showCoordinates = !boardState.showCoordinates;
        d3.select(this).text(boardState.showCoordinates ? 'Hide Coordinates' : 'Show Coordinates');
        updateBoard();
    });
  }
  
  
  
  // initialize after the page is loaded
  document.addEventListener('DOMContentLoaded', () => {
  // clear the coordinate cache
//   squareCenterCoords.clear();
  redDotScreenCoords.clear();
  
  initializeBoard();
  initializeGraph(graph);
  
  });
  