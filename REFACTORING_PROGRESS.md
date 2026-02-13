# Circuit Visualization Refactoring Progress

## Completed Work

### 1. Utility Functions Created ✅
- **`ui/src/utils/colorUtils.ts`**: Color conversion utilities (hexToRgb, rgbToHex, rgbToHsl, hslToRgb, mixHexColorsVivid)
- **`ui/src/utils/fenUtils.ts`**: FEN string extraction and validation utilities
- **`ui/src/utils/activationUtils.ts`**: Node activation data parsing and matching utilities
- **`ui/src/utils/graphMergeUtils.ts`**: Graph merging logic with color mixing

### 2. Custom Hooks Created ✅
- **`ui/src/hooks/useCircuitFileUpload.ts`**: File upload handling (single and multiple files)
- **`ui/src/hooks/useActivationData.ts`**: Node activation data retrieval
- **`ui/src/hooks/useFenExtraction.ts`**: FEN and move extraction from circuit data
- **`ui/src/hooks/useDictionaryName.ts`**: Dictionary name generation based on layer and feature type
- **`ui/src/hooks/useCircuitStateReducer.ts`**: useReducer-based state management (replaces 30+ useState calls)

### 3. UI Components Created ✅
- **`ui/src/components/circuits/FileUploadZone.tsx`**: File upload zone component

## Remaining Work

### 1. Complete State Migration
The main file still uses many `useState` calls that need to be migrated to `useCircuitStateReducer`. Current state variables that need migration:
- All file-related state → `circuitState.state.file.*`
- All activation-related state → `circuitState.state.activation.*`
- All display-related state → `circuitState.state.display.*`
- All feature diffing state → `circuitState.state.featureDiffing.*`
- All position mapping state → `circuitState.state.positionMapping.*`
- All dense state → `circuitState.state.dense.*`
- All sync state → `circuitState.state.sync.*`
- All clerp state → `circuitState.state.clerp.*`
- All steering state → `circuitState.state.steering.*`
- All posFeature state → `circuitState.state.posFeature.*`

### 2. Replace Function Calls
Replace all function calls throughout the file:
- `setOriginalCircuitJson` → `actions.file.setOriginalJson`
- `setTopActivations` → `actions.activation.setTopActivations`
- `setLoadingTopActivations` → `actions.activation.setLoadingTopActivations`
- `setTokenPredictions` → `actions.activation.setTokenPredictions`
- `setLoadingTokenPredictions` → `actions.activation.setLoadingTokenPredictions`
- `setShowAllPositions` → `actions.display.setShowAllPositions`
- `setShowSubgraph` → `actions.display.setShowSubgraph`
- `setPerturbedFen` → `actions.featureDiffing.setPerturbedFen`
- `setInactiveNodes` → `actions.featureDiffing.setInactiveNodes`
- `setEnablePositionMapping` → `actions.positionMapping.setEnablePositionMapping`
- `setDenseNodes` → `actions.dense.setDenseNodes`
- `setSyncingToBackend` → `actions.sync.setSyncingToBackend`
- `setEditingClerp` → `actions.clerp.setEditingClerp`
- `setSteeringScale` → `actions.steering.setSteeringScale`
- And many more...

### 3. Replace Utility Functions
- Replace `mergeGraphs` function with `mergeCircuitGraphs` from `graphMergeUtils.ts`
- Replace `normalizeZPattern` with import from `activationUtils.ts`
- Replace `parseNodeIdParts` with `parseNodeId` from `activationUtils.ts`
- Replace `extractFenFromPrompt` with `fenExtraction.extractFenFromPrompt`
- Replace `extractOutputMove` with `fenExtraction.extractOutputMove`
- Replace `getDictionaryName` with `dictionaryName.getDictionaryName`
- Replace `getNodeActivationData` with `activationDataHook.getNodeActivationData`

### 4. Translate All Chinese Comments
All Chinese comments need to be translated to English:
- Line 18: "定义节点激活数据的类型" → "Node activation data type definition"
- Line 27: "智能显示概率的辅助函数" → "Helper function for intelligently displaying probabilities"
- Line 59: "不再使用全局状态，改为直接检查后端状态" → "No longer using global state, directly checking backend status"
- Line 64: "存储原始JSON数据" → "Store original JSON data"
- Line 65: "当前编辑的clerp" → "Currently editing clerp"
- Line 66: "保存状态" → "Saving state"
- And many more throughout the file...

### 5. Translate All Chinese UI Text
All Chinese UI text needs to be translated:
- "上传Clerp" → "Upload Clerp"
- "下载Clerp" → "Download Clerp"
- "判断Dense" → "Check Dense"
- "比较激活差异" → "Compare Activation Differences"
- "显示日志" / "隐藏日志" → "Show Logs" / "Hide Logs"
- "单位置模式" / "所有位置模式" → "Single Position Mode" / "All Positions Mode"
- And many more...

### 6. Split UI Components
Create additional UI components:
- **ControlBar.tsx**: Header controls (upload button, sync buttons, dense check, etc.)
- **ChessBoardSection.tsx**: Chess board display section
- **TopActivationsSection.tsx**: Top activations display
- **TokenPredictionsSection.tsx**: Token predictions display
- **FeatureInterpretationEditor.tsx**: Feature interpretation editor
- **SubgraphControls.tsx**: Subgraph mode controls
- **PositionMappingControls.tsx**: Position mapping controls

### 7. Ensure No Duplicate Model Loading
The backend already has caching mechanisms in place:
- `get_hooked_model()` in `server/app.py` checks `_hooked_models` cache
- `get_cached_models()` in `server/circuits_service.py` checks `_global_hooked_models` cache
- The frontend should check backend status before loading (already implemented in `checkSaeLoaded`)

## Next Steps

1. **Continue state migration**: Replace remaining `useState` calls with reducer actions
2. **Replace function calls**: Update all function references to use new hooks and utilities
3. **Translate comments**: Systematically go through and translate all Chinese comments
4. **Translate UI text**: Replace all Chinese UI strings with English equivalents
5. **Split components**: Extract large UI sections into separate components
6. **Test thoroughly**: Ensure all functionality still works after refactoring

## Notes

- The file is very large (4268 lines), so refactoring should be done incrementally
- Test after each major change to ensure functionality is preserved
- The backend caching mechanisms are already in place, so no duplicate loading should occur
- All new code follows TypeScript best practices with proper type hints
