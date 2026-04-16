/**
 * Custom hook for handling circuit file uploads
 * Supports single and multiple file uploads with validation
 */

import { useCallback } from "react";
import { CircuitJsonData } from "@/components/circuits/link-graph/utils";
import { transformCircuitData } from "@/components/circuits/link-graph/utils";
import { mergeCircuitGraphs } from "@/utils/graphMergeUtils";
import { isValidBt4AnalysisName } from "@/utils/bt4Sae";

interface CheckAnalysisNamesResult {
  isValid: boolean;
  warnings: string[];
}

/**
 * Check if analysis_name is in known combinations
 */
const checkAnalysisNames = (metadata: any): CheckAnalysisNamesResult => {
  const warnings: string[] = [];
  const lorsaAnalysisName = metadata?.lorsa_analysis_name;
  const tcAnalysisName = metadata?.tc_analysis_name || metadata?.clt_analysis_name;
  
  if (lorsaAnalysisName && typeof lorsaAnalysisName === 'string') {
    if (!isValidBt4AnalysisName(lorsaAnalysisName)) {
      warnings.push(
        `⚠️ Lorsa analysis_name "${lorsaAnalysisName}" is not a recognized BT4 combo name.`
      );
    }
  }
  
  if (tcAnalysisName && typeof tcAnalysisName === 'string') {
    if (!isValidBt4AnalysisName(tcAnalysisName)) {
      warnings.push(
        `⚠️ TC analysis_name "${tcAnalysisName}" is not a recognized BT4 combo name.`
      );
    }
  }
  
  return {
    isValid: warnings.length === 0,
    warnings
  };
};

interface UseCircuitFileUploadOptions {
  onSuccess: (data: any, originalJson: any, fileName: string, multiJsons?: { json: CircuitJsonData; fileName: string }[]) => void;
  onError: (error: string) => void;
  setLoading: (loading: boolean) => void;
}

export const useCircuitFileUpload = ({
  onSuccess,
  onError,
  setLoading,
}: UseCircuitFileUploadOptions) => {
  /**
   * Handle single file upload
   */
  const handleSingleFileUpload = useCallback(async (file: File) => {
    if (!file.name.endsWith('.json')) {
      onError('Please upload a JSON file');
      return;
    }

    try {
      setLoading(true);
      onError(null as any);
      
      const text = await file.text();
      const jsonData: CircuitJsonData = JSON.parse(text);
      
      // Check analysis_name
      const metadata = jsonData?.metadata || {};
      const { isValid, warnings } = checkAnalysisNames(metadata);
      
      if (!isValid && warnings.length > 0) {
        const warningMessage = warnings.join('\n');
        console.warn('⚠️ Circuit file analysis_name check:', warnings);
        alert(warningMessage);
      }
      
      // Transform data
      const data = transformCircuitData(jsonData);
      // Inject source information (single file index is 0)
      const annotated = {
        ...data,
        nodes: data.nodes.map(n => ({
          ...n,
          sourceIndex: 0,
          sourceIndices: [0],
          sourceFiles: [file.name],
        })),
        metadata: {
          ...data.metadata,
          sourceFileNames: [file.name],
        }
      } as any;

      onSuccess(annotated, jsonData, file.name, [{ json: jsonData, fileName: file.name }]);
    } catch (err) {
      console.error('Failed to load circuit data:', err);
      onError(err instanceof Error ? err.message : 'Failed to load circuit data');
    } finally {
      setLoading(false);
    }
  }, [onSuccess, onError, setLoading]);

  /**
   * Handle multiple file upload (1-4 files)
   */
  const handleMultiFilesUpload = useCallback(async (files: FileList | File[]) => {
    const list = Array.from(files).filter(f => f.name.endsWith('.json')).slice(0, 4);
    if (list.length === 0) {
      onError('Please upload 1-4 JSON files');
      return;
    }

    try {
      setLoading(true);
      onError(null as any);

      const texts = await Promise.all(list.map(f => f.text()));
      const jsons: CircuitJsonData[] = texts.map(t => JSON.parse(t));
      const fileNames = list.map(f => f.name);

      // Check analysis_name for all files
      const allWarnings: string[] = [];
      jsons.forEach((json, index) => {
        const metadata = json?.metadata || {};
        const { warnings } = checkAnalysisNames(metadata);
        if (warnings.length > 0) {
          warnings.forEach(w => {
            allWarnings.push(`[File ${fileNames[index]}]: ${w}`);
          });
        }
      });
      
      if (allWarnings.length > 0) {
        const warningMessage = allWarnings.join('\n');
        console.warn('⚠️ Circuit file analysis_name check:', allWarnings);
        alert(warningMessage);
      }

      // Merge graphs (validates lorsa/tc_analysis_name consistency when 2+ files)
      const merged = jsons.length === 1 
        ? (() => {
            const data = transformCircuitData(jsons[0]);
            return {
              ...data,
              nodes: data.nodes.map(n => ({
                ...n,
                sourceIndex: 0,
                sourceIndices: [0],
                sourceFiles: [fileNames[0]],
              })),
              metadata: {
                ...data.metadata,
                sourceFileNames: [fileNames[0]],
              }
            };
          })()
        : mergeCircuitGraphs(jsons, fileNames);

      const originalJson = jsons.length === 1 ? jsons[0] : merged;
      const fileName = list.length === 1 ? list[0].name : `merged_${list.length}_graphs.json`;
      const multiJsons = list.map((f, i) => ({ json: jsons[i], fileName: f.name }));

      onSuccess(merged, originalJson, fileName, multiJsons);
    } catch (err) {
      console.error('Failed to load circuit data:', err);
      onError(err instanceof Error ? err.message : 'Failed to load circuit data');
    } finally {
      setLoading(false);
    }
  }, [onSuccess, onError, setLoading]);

  return {
    handleSingleFileUpload,
    handleMultiFilesUpload,
  };
};
