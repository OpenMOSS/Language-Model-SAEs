# Circuit Visualization 文件重构分析

## 文件概览
- **总行数**: 4268行
- **主要问题**: 代码过长，包含大量重复逻辑，可维护性差

## 主要内容模块

### 1. 状态管理 (约30+个useState)
- 文件上传相关: `isDragOver`, `originalCircuitJson`, `originalFileName`, `hasUnsavedChanges`, `saveHistory`
- 激活数据相关: `topActivations`, `loadingTopActivations`, `tokenPredictions`, `loadingTokenPredictions`
- 显示模式相关: `showAllPositions`, `allPositionsActivationData`, `loadingAllPositions`, `multiGraphActivationData`
- Graph Feature Diffing: `perturbedFen`, `isComparingFens`, `inactiveNodes`, `diffingLogs`, `showDiffingLogs`
- Position映射: `enablePositionMapping`, `positionMappingSelections`, `draftPositionMappingSelections`
- 子图功能: `showSubgraph`, `subgraphData`, `subgraphRootNodeId`
- Clerp编辑: `editingClerp`, `isSaving`, `updateCounter`
- Dense features: `denseNodes`, `denseThreshold`, `checkingDenseFeatures`
- 同步状态: `syncingToBackend`, `syncingFromBackend`
- Z-pattern: `loadingBackendZPattern`, `backendZPatternByNode`
- PosFeature: `posFeatureLayer`, `posFeaturePositions`, `posFeatureComponentType`
- Steering: `steeringScale`, `steeringScaleInput`

### 2. 重复代码识别

#### 2.1 颜色转换函数 (重复)
**位置**: 
- `circuit-visualization.tsx` 第169-261行 (mergeGraphs内部)
- `interaction-circuit/page.tsx` 第78-168行

**重复函数**:
- `hexToRgb`
- `rgbToHex`
- `rgbToHsl`
- `hslToRgb`
- `mixHexColorsVivid`

**建议**: 提取到 `ui/src/utils/colorUtils.ts`

#### 2.2 文件上传逻辑 (重复)
**位置**: 
- `handleSingleFileUpload` (第542-616行)
- `handleMultiFilesUpload` (第619-704行)

**重复代码**:
- JSON解析和验证
- analysis_name检查
- 状态重置逻辑
- 错误处理

**建议**: 提取公共逻辑到 `handleFileUpload` 函数

#### 2.3 FEN提取逻辑 (重复)
**位置**:
- `extractFenFromPrompt` (第743-777行)
- `extractFenFromCircuitJson` (第780-797行)

**重复代码**:
- FEN格式检测
- 正则表达式匹配
- 验证逻辑

**建议**: 提取到 `ui/src/utils/fenUtils.ts`

#### 2.4 激活数据获取 (重复)
**位置**:
- `getNodeActivationData` (第870-1034行, ~164行)
- `getNodeActivationDataFromJson` (第1036-1149行, ~113行)

**重复代码**:
- nodeId解析逻辑
- 节点搜索逻辑
- 记录匹配逻辑
- 模糊匹配逻辑

**建议**: 提取公共逻辑，使用参数区分数据源

#### 2.5 输出移动提取 (重复)
**位置**:
- `extractOutputMove` (第800-834行)
- `extractOutputMoveFromCircuitJson` (第837-867行)

**重复代码**:
- metadata读取逻辑
- prompt解析逻辑
- 正则匹配

**建议**: 合并为一个函数，接受数据源参数

### 3. 可提取的工具函数

#### 3.1 颜色工具 (`utils/colorUtils.ts`)
```typescript
export const hexToRgb = (hex: string): { r: number; g: number; b: number } | null
export const rgbToHex = (rgb: { r: number; g: number; b: number }): string
export const rgbToHsl = (rgb: { r: number; g: number; b: number }): { h: number; s: number; l: number }
export const hslToRgb = (hsl: { h: number; s: number; l: number }): { r: number; g: number; b: number }
export const mixHexColorsVivid = (hexColors: string[]): string | null
```

#### 3.2 FEN工具 (`utils/fenUtils.ts`)
```typescript
export const extractFenFromText = (text: string): string | null
export const validateFen = (fen: string): boolean
export const extractMoveFromText = (text: string): string | null
```

#### 3.3 激活数据工具 (`utils/activationUtils.ts`)
```typescript
export const parseNodeId = (nodeId: string): { rawLayer: number; featureOrHead: number; ctxIdx: number }
export const findNodeInJson = (json: any, nodeId: string): any | null
export const findActivationRecord = (json: any, parsed: ParsedNodeId, featureType?: string): any | null
```

#### 3.4 图合并工具 (`utils/graphMergeUtils.ts`)
```typescript
export const mergeCircuitGraphs = (jsons: CircuitJsonData[], fileNames?: string[]): LinkGraphData
export const getSubsetColor = (sourceIndices: number[], colorMap: Map<string, string>): string | null
```

### 4. 可拆分的UI组件

#### 4.1 文件上传组件 (`components/circuits/FileUploadZone.tsx`)
- 拖拽上传界面
- 文件选择逻辑
- 约50行代码

#### 4.2 控制栏组件 (`components/circuits/ControlBar.tsx`)
- Graph Feature Diffing控件
- Clerp同步控件
- Dense Feature检查控件
- Position映射控件
- 约200行代码

#### 4.3 棋盘显示组件 (`components/circuits/ChessBoardSection.tsx`)
- 单文件棋盘显示
- 多文件棋盘显示
- 激活数据展示
- 约300行代码

#### 4.4 Top Activation组件 (`components/circuits/TopActivationSection.tsx`)
- Top Activation数据获取
- 棋盘网格显示
- 约150行代码

#### 4.5 Token Predictions组件 (`components/circuits/TokenPredictionsSection.tsx`)
- Steering分析
- 概率差异显示
- 约200行代码

#### 4.6 Feature Editor组件 (`components/circuits/FeatureEditorSection.tsx`)
- Clerp编辑
- 保存逻辑
- 约150行代码

#### 4.7 子图控制组件 (`components/circuits/SubgraphControls.tsx`)
- 子图创建/退出
- 子图保存
- 约100行代码

### 5. 状态管理优化

#### 5.1 使用useReducer替代多个useState
```typescript
// 文件上传状态
const [fileState, dispatchFile] = useReducer(fileReducer, initialState)

// 激活数据状态
const [activationState, dispatchActivation] = useReducer(activationReducer, initialState)

// UI显示状态
const [uiState, dispatchUI] = useReducer(uiReducer, initialState)
```

#### 5.2 自定义Hooks提取
```typescript
// hooks/useCircuitFileUpload.ts
export const useCircuitFileUpload = () => {
  // 文件上传逻辑
}

// hooks/useActivationData.ts
export const useActivationData = (nodeId: string | null, jsonData: any) => {
  // 激活数据获取逻辑
}

// hooks/useGraphMerging.ts
export const useGraphMerging = () => {
  // 图合并逻辑
}

// hooks/useFeatureDiffing.ts
export const useFeatureDiffing = () => {
  // FEN差异比较逻辑
}
```

### 6. 重构后的文件结构

```
circuit-visualization.tsx (主组件, ~800行)
├── hooks/
│   ├── useCircuitFileUpload.ts (~150行)
│   ├── useActivationData.ts (~200行)
│   ├── useGraphMerging.ts (~100行)
│   ├── useFeatureDiffing.ts (~150行)
│   └── useSubgraph.ts (~100行)
├── components/
│   ├── FileUploadZone.tsx (~50行)
│   ├── ControlBar.tsx (~200行)
│   ├── ChessBoardSection.tsx (~300行)
│   ├── TopActivationSection.tsx (~150行)
│   ├── TokenPredictionsSection.tsx (~200行)
│   ├── FeatureEditorSection.tsx (~150行)
│   └── SubgraphControls.tsx (~100行)
└── utils/
    ├── colorUtils.ts (~100行)
    ├── fenUtils.ts (~80行)
    ├── activationUtils.ts (~150行)
    └── graphMergeUtils.ts (~200行)
```

### 7. 预期效果

- **主文件行数**: 从4268行减少到约800行 (减少81%)
- **代码复用**: 消除重复代码，提高可维护性
- **可测试性**: 提取的函数和组件更容易单元测试
- **可读性**: 每个文件职责单一，更容易理解

### 8. 重构优先级

1. **高优先级** (立即处理):
   - 提取颜色工具函数 (消除重复)
   - 提取FEN工具函数 (消除重复)
   - 拆分文件上传逻辑 (减少重复)

2. **中优先级** (后续处理):
   - 拆分UI组件
   - 提取自定义Hooks
   - 优化状态管理

3. **低优先级** (可选):
   - 使用useReducer替代useState
   - 进一步优化性能

### 9. 注意事项

- 保持现有功能不变
- 逐步重构，避免一次性大改
- 确保所有测试通过
- 保持API兼容性
