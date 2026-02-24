/**
 * FEN (Forsyth-Edwards Notation) utility functions
 * Used for extracting and validating FEN strings from circuit data
 */

/**
 * Extract FEN string from text
 */
export const extractFenFromText = (text: string): string | null => {
  if (!text) return null;
  
  const lines = text.split('\n');
  for (const line of lines) {
    const trimmed = line.trim();
    // Check if contains FEN format - includes slash and enough characters
    if (trimmed.includes('/')) {
      const parts = trimmed.split(/\s+/);
      if (parts.length >= 6) {
        const [boardPart, activeColor] = parts;
        const boardRows = boardPart.split('/');
        
        if (boardRows.length === 8 && /^[wb]$/.test(activeColor)) {
          return trimmed;
        }
      }
    }
  }
  
  // If no complete FEN found, try simpler matching
  const simpleMatch = text.match(/[rnbqkpRNBQKP1-8\/]{15,}\s+[wb]\s+[KQkqA-Za-z-]+\s+[a-h][36-]?\s*\d*\s*\d*/);
  return simpleMatch ? simpleMatch[0] : null;
};

/**
 * Extract output move from text or metadata
 */
export const extractMoveFromData = (data: {
  metadata?: {
    target_move?: string;
    logit_moves?: string[];
    prompt_tokens?: string[];
  };
}): string | null => {
  if (!data) return null;

  // 1) Priority: read from metadata target_move or logit_moves[0]
  const tm = data.metadata?.target_move;
  if (typeof tm === 'string' && /^[a-h][1-8][a-h][1-8]([qrbn])?$/i.test(tm)) {
    return tm.toLowerCase();
  }
  const lm0 = data.metadata?.logit_moves?.[0];
  if (typeof lm0 === 'string' && /^[a-h][1-8][a-h][1-8]([qrbn])?$/i.test(lm0)) {
    return lm0.toLowerCase();
  }

  // 2) Fallback: parse from prompt_tokens
  if (!data.metadata?.prompt_tokens) return null;
  const promptText = Array.isArray(data.metadata.prompt_tokens)
    ? data.metadata.prompt_tokens.join(' ')
    : String(data.metadata.prompt_tokens);

  const movePatterns = [
    // Match explicit "Output:" or "Move:" annotations before a UCI move
    /(?:Output|Move)[:：]\s*([a-h][1-8][a-h][1-8])/i,
    // Fallback: any standalone UCI move token
    /\b([a-h][1-8][a-h][1-8])\b/g
  ];

  for (const pattern of movePatterns) {
    const matches = promptText.match(pattern);
    if (matches) {
      const lastMatch = Array.isArray(matches) ? matches[matches.length - 1] : matches;
      const moveMatch = lastMatch.match(/[a-h][1-8][a-h][1-8]/);
      if (moveMatch) {
        return moveMatch[0].toLowerCase();
      }
    }
  }

  return null;
};

/**
 * Validate FEN format
 */
export const validateFen = (fen: string): boolean => {
  if (!fen || typeof fen !== 'string') return false;
  
  const parts = fen.trim().split(/\s+/);
  if (parts.length < 6) return false;
  
  const [boardPart, activeColor] = parts;
  const boardRows = boardPart.split('/');
  
  if (boardRows.length !== 8) return false;
  if (!/^[wb]$/.test(activeColor)) return false;
  
  // Validate each row
  for (const row of boardRows) {
    if (!/^[rnbqkpRNBQKP1-8]+$/.test(row)) return false;
    
    let rowSquares = 0;
    for (const char of row) {
      if (/\d/.test(char)) {
        rowSquares += parseInt(char);
      } else {
        rowSquares += 1;
      }
    }
    if (rowSquares !== 8) return false;
  }
  
  return true;
};
