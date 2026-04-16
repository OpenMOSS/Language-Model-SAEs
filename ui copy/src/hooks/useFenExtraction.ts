/**
 * Custom hook for extracting FEN strings and moves from circuit data
 */

import { useCallback, useMemo } from "react";
import { extractFenFromText, extractMoveFromData } from "@/utils/fenUtils";

interface UseFenExtractionOptions {
  linkGraphData: any;
  originalCircuitJson: any;
  multiOriginalJsons: { json: any; fileName: string }[];
}

export const useFenExtraction = ({
  linkGraphData,
  originalCircuitJson,
  multiOriginalJsons,
}: UseFenExtractionOptions) => {
  /**
   * Extract FEN from prompt tokens in linkGraphData
   */
  const extractFenFromPrompt = useCallback((): string | null => {
    if (!linkGraphData?.metadata?.prompt_tokens) return null;
    
    const promptText = Array.isArray(linkGraphData.metadata.prompt_tokens)
      ? linkGraphData.metadata.prompt_tokens.join(' ')
      : String(linkGraphData.metadata.prompt_tokens);
    
    return extractFenFromText(promptText);
  }, [linkGraphData]);

  /**
   * Extract FEN from specific circuit JSON
   */
  const extractFenFromCircuitJson = useCallback((json: any): string | null => {
    const tokens = json?.metadata?.prompt_tokens;
    if (!tokens) return null;
    const promptText = Array.isArray(tokens) ? tokens.join(' ') : String(tokens);
    return extractFenFromText(promptText);
  }, []);

  /**
   * Extract output move from linkGraphData
   */
  const extractOutputMove = useCallback((): string | null => {
    if (!linkGraphData) return null;
    return extractMoveFromData(linkGraphData);
  }, [linkGraphData]);

  /**
   * Extract output move from specific circuit JSON
   */
  const extractOutputMoveFromCircuitJson = useCallback((json: any): string | null => {
    if (!json) return null;
    return extractMoveFromData(json);
  }, []);

  // Memoized values
  const fen = useMemo(() => extractFenFromPrompt(), [extractFenFromPrompt]);
  const outputMove = useMemo(() => extractOutputMove(), [extractOutputMove]);

  return {
    fen,
    outputMove,
    extractFenFromPrompt,
    extractFenFromCircuitJson,
    extractOutputMove,
    extractOutputMoveFromCircuitJson,
  };
};
