# Batch Replacement Guide for Circuit Visualization Refactoring

This document provides a systematic guide for completing the refactoring of `circuit-visualization.tsx`.

## State Variable Replacements

### File State
- `originalCircuitJson` → `file.originalCircuitJson`
- `setOriginalCircuitJson(...)` → `actions.file.setOriginalJson(...)`
- `originalFileName` → `file.originalFileName`
- `setOriginalFileName(...)` → `actions.file.setOriginalFileName(...)`
- `multiOriginalJsons` → `file.multiOriginalJsons`
- `setMultiOriginalJsons(...)` → `actions.file.setMultiOriginalJsons(...)`
- `hasUnsavedChanges` → `file.hasUnsavedChanges`
- `setHasUnsavedChanges(...)` → `actions.file.setHasUnsavedChanges(...)`
- `saveHistory` → `file.saveHistory`
- `setSaveHistory(...)` → `actions.file.addSaveHistory(...)` or `actions.file.clearSaveHistory()`

### Activation State
- `topActivations` → `activation.topActivations`
- `setTopActivations(...)` → `actions.activation.setTopActivations(...)`
- `loadingTopActivations` → `activation.loadingTopActivations`
- `setLoadingTopActivations(...)` → `actions.activation.setLoadingTopActivations(...)`
- `tokenPredictions` → `activation.tokenPredictions`
- `setTokenPredictions(...)` → `actions.activation.setTokenPredictions(...)`
- `loadingTokenPredictions` → `activation.loadingTokenPredictions`
- `setLoadingTokenPredictions(...)` → `actions.activation.setLoadingTokenPredictions(...)`
- `allPositionsActivationData` → `activation.allPositionsActivationData`
- `setAllPositionsActivationData(...)` → `actions.activation.setAllPositionsData(...)`
- `loadingAllPositions` → `activation.loadingAllPositions`
- `setLoadingAllPositions(...)` → `actions.activation.setLoadingAllPositions(...)`
- `multiGraphActivationData` → `activation.multiGraphActivationData`
- `setMultiGraphActivationData(...)` → `actions.activation.setMultiGraphData(...)`
- `loadingBackendZPattern` → `activation.loadingBackendZPattern`
- `setLoadingBackendZPattern(...)` → `actions.activation.setLoadingBackendZPattern(...)`
- `backendZPatternByNode` → `activation.backendZPatternByNode`
- `setBackendZPatternByNode(...)` → `actions.activation.setBackendZPatternByNode(...)`

### Display State
- `showAllPositions` → `display.showAllPositions`
- `setShowAllPositions(...)` → `actions.display.setShowAllPositions(...)`
- `showSubgraph` → `display.showSubgraph`
- `setShowSubgraph(...)` → `actions.display.setShowSubgraph(...)`
- `subgraphData` → `display.subgraphData`
- `setSubgraphData(...)` → `actions.display.setSubgraphData(...)`
- `subgraphRootNodeId` → `display.subgraphRootNodeId`
- `setSubgraphRootNodeId(...)` → `actions.display.setSubgraphRootNodeId(...)`
- `showDiffingLogs` → `display.showDiffingLogs`
- `setShowDiffingLogs(...)` → `actions.display.setShowDiffingLogs(...)`

### Feature Diffing State
- `perturbedFen` → `featureDiffing.perturbedFen`
- `setPerturbedFen(...)` → `actions.featureDiffing.setPerturbedFen(...)`
- `isComparingFens` → `featureDiffing.isComparingFens`
- `setIsComparingFens(...)` → `actions.featureDiffing.setIsComparingFens(...)`
- `inactiveNodes` → `featureDiffing.inactiveNodes`
- `setInactiveNodes(...)` → `actions.featureDiffing.setInactiveNodes(...)`
- `diffingLogs` → `featureDiffing.diffingLogs`
- `setDiffingLogs([...prev, new])` → `actions.featureDiffing.addDiffingLog(...)`
- `setDiffingLogs([])` → `actions.featureDiffing.clearDiffingLogs()`

### Position Mapping State
- `enablePositionMapping` → `positionMapping.enablePositionMapping`
- `setEnablePositionMapping(...)` → `actions.positionMapping.setEnablePositionMapping(...)`
- `positionMappingSelections` → `positionMapping.positionMappingSelections`
- `setPositionMappingSelections(...)` → `actions.positionMapping.setPositionMappingSelections(...)`
- `draftPositionMappingSelections` → `positionMapping.draftPositionMappingSelections`
- `setDraftPositionMappingSelections(...)` → `actions.positionMapping.setDraftPositionMappingSelections(...)`
- `positionMappingApplyNonce` → `positionMapping.positionMappingApplyNonce`
- `setPositionMappingApplyNonce(x => x + 1)` → `actions.positionMapping.incrementPositionMappingApplyNonce()`

### Dense State
- `denseNodes` → `dense.denseNodes`
- `setDenseNodes(...)` → `actions.dense.setDenseNodes(...)`
- `denseThreshold` → `dense.denseThreshold`
- `setDenseThreshold(...)` → `actions.dense.setDenseThreshold(...)`
- `checkingDenseFeatures` → `dense.checkingDenseFeatures`
- `setCheckingDenseFeatures(...)` → `actions.dense.setCheckingDenseFeatures(...)`

### Sync State
- `syncingToBackend` → `sync.syncingToBackend`
- `setSyncingToBackend(...)` → `actions.sync.setSyncingToBackend(...)`
- `syncingFromBackend` → `sync.syncingFromBackend`
- `setSyncingFromBackend(...)` → `actions.sync.setSyncingFromBackend(...)`

### Clerp State
- `editingClerp` → `clerp.editingClerp`
- `setEditingClerp(...)` → `actions.clerp.setEditingClerp(...)`
- `isSaving` → `clerp.isSaving`
- `setIsSaving(...)` → `actions.clerp.setIsSaving(...)`
- `updateCounter` → `clerp.updateCounter`
- `setUpdateCounter(x => x + 1)` → `actions.clerp.incrementUpdateCounter()`

### Steering State
- `steeringScale` → `steering.steeringScale`
- `setSteeringScale(...)` → `actions.steering.setSteeringScale(...)`
- `steeringScaleInput` → `steering.steeringScaleInput`
- `setSteeringScaleInput(...)` → `actions.steering.setSteeringScaleInput(...)`

### PosFeature State
- `posFeatureLayer` → `posFeature.posFeatureLayer`
- `setPosFeatureLayer(...)` → `actions.posFeature.setPosFeatureLayer(...)`
- `posFeaturePositions` → `posFeature.posFeaturePositions`
- `setPosFeaturePositions(...)` → `actions.posFeature.setPosFeaturePositions(...)`
- `posFeatureComponentType` → `posFeature.posFeatureComponentType`
- `setPosFeatureComponentType(...)` → `actions.posFeature.setPosFeatureComponentType(...)`

## Function Replacements

### Already Completed ✅
- `mergeGraphs` → `mergeCircuitGraphs` (from `graphMergeUtils.ts`)
- `extractFenFromPrompt` → `fenExtraction.extractFenFromPrompt`
- `extractFenFromCircuitJson` → `fenExtraction.extractFenFromCircuitJson`
- `extractOutputMove` → `fenExtraction.extractOutputMove`
- `extractOutputMoveFromCircuitJson` → `fenExtraction.extractOutputMoveFromCircuitJson`
- `getNodeActivationData` → `activationDataHook.getNodeActivationData`
- `getNodeActivationDataFromJson` → `activationDataHook.getNodeActivationDataFromJson`
- `getDictionaryName` → `dictionaryName.getDictionaryName`
- `getSaeNameForCircuit` → `dictionaryName.getSaeNameForCircuit`
- `normalizeZPattern` → Import from `activationUtils.ts` (need to add import)
- `parseNodeIdParts` → Use `parseNodeId` from `activationUtils.ts`

### Still Need Replacement
- `normalizeZPattern` function definition → Import from `activationUtils.ts`
- All references to `originalCircuitJson` → `file.originalCircuitJson`
- All references to `updateCounter` → `clerp.updateCounter`

## Chinese to English Translations

### Comments
- `// 不再使用全局状态，改为直接检查后端状态` → `// No longer using global state, directly checking backend status`
- `// 存储原始JSON数据` → `// Store original JSON data`
- `// 当前编辑的clerp` → `// Currently editing clerp`
- `// 保存状态` → `// Saving state`
- `// 原始文件名` → `// Original file name`
- `// 用于强制更新的计数器` → `// Counter for forcing updates`
- `// 是否有未保存的更改` → `// Whether there are unsaved changes`
- `// 保存历史记录` → `// Save history`
- `// Top Activation 数据` → `// Top Activation data`
- `// 加载状态` → `// Loading state`
- `// Token Predictions 数据` → `// Token Predictions data`
- `// steering 放大系数` → `// Steering scale factor`
- `// 文本输入，用于支持暂存 "-"` → `// Text input for supporting temporary "-"`
- `// Dense节点集合` → `// Dense nodes set`
- `// Dense阈值（空字符串表示无限大）` → `// Dense threshold (empty string means infinite)`
- `// 是否正在检查dense features` → `// Whether checking dense features`
- `// 是否正在同步到后端` → `// Whether syncing to backend`
- `// 是否正在从后端同步` → `// Whether syncing from backend`
- `// Graph Feature Diffing 相关状态` → `// Graph Feature Diffing related state`
- `// Perturbed FEN输入` → `// Perturbed FEN input`
- `// 是否正在比较` → `// Whether comparing`
- `// 未激活节点集合` → `// Inactive nodes set`
- `// 比较日志` → `// Comparison logs`
- `// 是否显示日志` → `// Whether to show logs`
- `// ===== Position 映射高亮（多图模式）=====` → `// ===== Position Mapping Highlight (Multi-graph Mode) =====`
- `// 每个 source graph 选择一个 position（0-63）。key=graphIndex` → `// Each source graph selects one position (0-63). key=graphIndex`
- `// 输入框草稿态：用户编辑时先写入这里，点"应用"后才真正生效` → `// Draft state for input: user edits are written here first, only take effect after clicking "Apply"`
- `// 用于强制刷新图（某些情况下 D3 渲染不会让用户立刻感知到变化）` → `// Used to force graph refresh (in some cases D3 rendering won't immediately show changes to users)`
- `// 子图功能相关状态` → `// Subgraph feature related state`
- `// 是否显示子图模式` → `// Whether to show subgraph mode`
- `// 子图数据` → `// Subgraph data`
- `// 子图根节点ID` → `// Subgraph root node ID`
- `// Feature 激活显示模式：单个位置 vs 所有位置` → `// Feature activation display mode: single position vs all positions`
- `// 是否显示所有位置的激活` → `// Whether to show activations for all positions`
- `// 所有位置的合并激活数据` → `// Merged activation data for all positions`
- `// 是否正在从后端加载所有位置数据` → `// Whether loading all positions data from backend`
- `// 多图模式的激活数据` → `// Activation data for multi-graph mode`
- `// 点击节点时，从后端实时计算/获取 z_pattern（不再信任 JSON 内保存的 z_pattern）` → `// When clicking a node, calculate/fetch z_pattern from backend in real-time (no longer trust z_pattern saved in JSON)`
- `// 仅用于"单位置模式"（showAllPositions=false）且 LoRSA 节点才会有 z_pattern` → `// Only used in "single position mode" (showAllPositions=false) and only LoRSA nodes have z_pattern`
- `// PosFeatureCard 相关状态` → `// PosFeatureCard related state`
- `// 层号` → `// Layer number`
- `// 位置输入（逗号分隔）` → `// Position input (comma-separated)`
- `// 组件类型` → `// Component type`
- `// 多图支持：存放多份原始 JSON 及其文件名` → `// Multi-graph support: store multiple original JSONs and their file names`

### UI Text
- `上传Clerp` → `Upload Clerp`
- `下载Clerp` → `Download Clerp`
- `判断Dense` → `Check Dense`
- `比较激活差异` → `Compare Activation Differences`
- `显示日志` → `Show Logs`
- `隐藏日志` → `Hide Logs`
- `单位置模式` → `Single Position Mode`
- `所有位置模式` → `All Positions Mode`
- `显示子图` → `Show Subgraph`
- `退出子图` → `Exit Subgraph`
- `保存子图` → `Save Subgraph`
- `应用映射` → `Apply Mapping`
- `撤销输入` → `Undo Input`
- `Position 映射高亮` → `Position Mapping Highlight`
- `为每个文件选一个 pos（0-63），高亮"不同文件的不同 pos 上但同一 (layer, feature) 的节点"` → `Select one pos (0-63) for each file, highlight nodes that are on different pos in different files but have the same (layer, feature)`
- `当前命中：` → `Current matches:`
- `个节点` → `nodes`
- `有未导出的更改` → `Unsaved changes`
- `导出` → `Export`
- `保存历史` → `Save History`
- `最近的更改:` → `Recent changes:`
- `上传新文件` → `Upload New File`
- `Perturb FEN:` → `Perturb FEN:`
- `输入扰动后的FEN...` → `Enter perturbed FEN...`
- `比较中...` → `Comparing...`
- `个未激活节点` → `inactive nodes`
- `Dense阈值:` → `Dense threshold:`
- `无限大` → `Infinite`
- `检查中...` → `Checking...`
- `个Dense节点` → `Dense nodes`
- `Circuit棋盘状态` → `Circuit Board State`
- `节点:` → `Node:`
- `输出移动:` → `Output Move:`
- `正在从后端加载所有位置的激活数据...` → `Loading activation data for all positions from backend...`
- `正在从后端计算 z_pattern...` → `Calculating z_pattern from backend...`
- `所有位置合并激活:` → `All positions merged activation:`
- `个非零激活` → `non-zero activations`
- `激活数据:` → `Activation data:`
- `个Z模式连接` → `Z-pattern connections`
- `位置 Feature 分析` → `Position Feature Analysis`
- `FEN:` → `FEN:`
- `层:` → `Layer:`
- `位置:` → `Positions:`
- `例如: 36 或 16,20,34` → `e.g., 36 or 16,20,34`
- `组件:` → `Component:`
- `FEN激活差异比较日志` → `FEN Activation Difference Comparison Logs`
- `清空日志` → `Clear Logs`
- `隐藏` → `Hide`
- `暂无日志...` → `No logs yet...`
- `比较中...` → `Comparing...`
- `Position 映射选择（每文件一个）` → `Position Mapping Selection (one per file)`
- `说明：先在下方输入 pos（草稿），再点击"应用映射"才会生效并刷新图（不会改变节点合并规则）` → `Note: First enter pos (draft) below, then click "Apply Mapping" to take effect and refresh the graph (won't change node merging rules)`
- `已应用命中：` → `Applied matches:`
- `pos` → `pos`
- `↦ 高亮` → `↦ Highlight`
- `高亮颜色：` → `Highlight color:`
- `选中节点:` → `Selected node:`
- `子图模式` → `Subgraph Mode`
- `根节点:` → `Root node:`
- `节点:` → `Nodes:`
- `边:` → `Links:`
- `Top Activation 棋盘` → `Top Activation Boards`
- `加载中...` → `Loading...`
- `正在获取 Top Activation 数据...` → `Fetching Top Activation data...`
- `Top #` → `Top #`
- `最大激活值:` → `Max activation value:`
- `未找到包含棋盘的激活样本` → `No activation samples with chess boards found`
- `Token Predictions` → `Token Predictions`
- `steering_scale:` → `steering_scale:`
- `开始分析` → `Start Analysis`
- `分析中...` → `Analyzing...`
- `正在运行特征干预分析...` → `Running feature intervention analysis...`
- `点击"开始分析"按钮以运行Token Predictions分析` → `Click "Start Analysis" button to run Token Predictions analysis`
- `请先在上方加载 TC/LoRSA 组合（SaeComboLoader）` → `Please load TC/LoRSA combo (SaeComboLoader) above first`
- `合法移动数:` → `Legal moves:`
- `平均概率差:` → `Avg prob diff:`
- `平均Logit差:` → `Avg logit diff:`
- `原始Value:` → `Original Value:`
- `Value变化:` → `Value change:`
- `概率差异最大（增加最多）Top 5` → `Top 5 Largest Probability Differences (Most Increased)`
- `排名:` → `Rank:`
- `概率差:` → `Prob diff:`
- `原始概率:` → `Original prob:`
- `修改后概率:` → `Modified prob:`
- `Logit差:` → `Logit diff:`
- `原始Logit:` → `Original logit:`
- `修改后Logit:` → `Modified logit:`
- `概率差异最小（减少最多）Top 5` → `Top 5 Smallest Probability Differences (Most Decreased)`
- `Feature Interpretation Editor` → `Feature Interpretation Editor`
- `Feature Interpretation (可编辑)` → `Feature Interpretation (Editable)`
- `(节点暂无interpretation字段，可新建)` → `(Node has no interpretation field, can create new)`
- `(当前为空，可编辑)` → `(Currently empty, editable)`
- `字符数:` → `Character count:`
- `输入或编辑节点的interpretation内容...` → `Enter or edit node interpretation content...`
- `重置` → `Reset`
- `保存并下载` → `Save and Download`
- `保存中...` → `Saving...`
- `⚠️ 内容已修改，请点击"保存并下载"以保存更改` → `⚠️ Content modified, please click "Save and Download" to save changes`
- `原始状态:` → `Original state:`
- `无interpretation字段` → `No interpretation field`
- `空字符串` → `Empty string`
- `有内容` → `Has content`
- `字符` → `characters`
- `当前编辑:` → `Current edit:`
- `空` → `Empty`
- `💡 文件更新工作流程:` → `💡 File Update Workflow:`
- `编辑interpretation内容后点击"保存并下载"` → `Edit interpretation content then click "Save and Download"`
- `更新后的文件会自动下载到Downloads文件夹` → `Updated file will be automatically downloaded to Downloads folder`
- `用新文件替换原文件，或重新拖拽到此页面` → `Replace original file with new file, or drag and drop again to this page`
- `文件名包含时间戳，避免意外覆盖` → `File name includes timestamp to avoid accidental overwrite`
- `提示:` → `Tip:`
- `由于浏览器安全限制，无法直接修改原文件，但下载的文件包含所有更改。` → `Due to browser security restrictions, cannot directly modify original file, but downloaded file contains all changes.`
- `Selected Feature Details` → `Selected Feature Details`
- `Connected features:` → `Connected features:`
- `查看L{layer} {type} #{index}` → `View L{layer} {type} #{index}`
- `No feature is available for this node` → `No feature is available for this node`

## Next Steps

1. Use find-and-replace in your IDE to replace all state variable references
2. Use find-and-replace to translate all Chinese comments
3. Use find-and-replace to translate all Chinese UI text
4. Test thoroughly after each batch of replacements
5. Split UI components into separate files as planned
