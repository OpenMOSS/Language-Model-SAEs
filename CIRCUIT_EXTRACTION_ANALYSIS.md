# Circuit Visualization - Remaining Extraction Opportunities

## 1. Duplicate Logic (Not Using Existing Utils)

### `getAllPositionsActivationDataSync` (~110 lines)
**Current:** Inline `parseFromNodeId`, `pushCandidateArrays`, `tryMatchRecord` - duplicates activationUtils.

**Fix:** Extend `activationUtils.ts`:
- Add `matchActivationRecordIgnorePosition(rec, parsed, featureType)` - same as matchActivationRecord but without `posOk` check
- Add `getAllPositionsActivationDataFromJson(jsonData, nodeId): NodeActivationData | null` - uses `findNodesArray`, `findCandidateRecords`, `matchActivationRecordIgnorePosition`

### `parseNodeIdParts` (3 lines)
**Current:** Wrapper that returns `{ rawLayer, featureOrHead, ctxIdx }` from parseNodeId.

**Fix:** Use `parseNodeId(nodeId)` directly and destructure where needed. Only used in `applyPositionMappingHighlights`.

### `formatProbability` (~12 lines)
**Current:** Inline in component.

**Fix:** Move to `utils/formatUtils.ts` or `utils/numberUtils.ts`.

---

## 2. Backend API Logic (~600 lines total)

These fetch/sync functions could move to a service or hook:

| Function | Lines | Suggestion |
|----------|-------|------------|
| `fetchAllPositionsFromBackend` | ~70 | `useCircuitBackend` or `circuitApi.ts` |
| `fetchAnalyzeFenFromBackend` | ~50 | same |
| `fetchZPatternForPosFromBackend` | ~45 | same |
| `getAllPositionsActivationData` | ~80 | `useActivationData` extension |
| `fetchTopActivations` | ~170 | `useCircuitBackend` |
| `checkSaeLoaded` | ~60 | `useCircuitBackend` or `utils/circuitApi.ts` |
| `fetchTokenPredictions` | ~80 | same |
| `syncClerpsToBackend` | ~75 | same |
| `syncClerpsFromBackend` | ~95 | same |
| `checkDenseFeatures` | ~115 | same |
| `compareFenActivations` | ~110 | same |

**New file:** `hooks/useCircuitBackend.ts` or `utils/circuitApi.ts` - consolidates all backend calls.

---

## 3. Graph Display Logic (~120 lines)

| Function | Lines | Suggestion |
|----------|-------|------------|
| `applyDenseNodeColors` | ~20 | `graphMergeUtils.ts` or new `graphDisplayUtils.ts` |
| `applyInactiveNodeColors` | ~20 | same |
| `applyPositionMappingHighlights` | ~65 | same |
| `findUpstreamNodes` | ~25 | `graphMergeUtils.ts` |
| `createSubgraph` | ~45 | same |

**New file:** `utils/graphDisplayUtils.ts` - `applyDenseNodeColors`, `applyInactiveNodeColors`, `applyPositionMappingHighlights`, `findUpstreamNodes`, `createSubgraph`.

---

## 4. UI Component Splits (per REFACTORING_PROGRESS.md)

| Component | Est. Lines | Content |
|-----------|------------|---------|
| **ControlBar** | ~250 | Header: Upload, Perturb FEN, Clerp sync, Dense check, Position mapping, Save history |
| **ChessBoardSection** | ~200 | Single-file + multi-file chess boards |
| **TopActivationsSection** | ~60 | Top activation grid |
| **TokenPredictionsSection** | ~160 | Steering analysis, probability display |
| **FeatureInterpretationEditor** | ~120 | Clerp textarea, save/reset |
| **SubgraphControls** | ~90 | Show subgraph, save, exit buttons |
| **PositionMappingControls** | ~90 | Multi-file position mapping UI |

---

## 5. File Handlers (~220 lines)

| Function | Lines | Suggestion |
|----------|-------|------------|
| `handleSaveClerp` | ~120 | Extract to `utils/clerpUtils.ts` or keep in component (tightly coupled to state) |
| `handleQuickExport` | ~25 | `utils/exportUtils.ts` |
| `handleSaveSubgraph` | ~70 | `utils/subgraphUtils.ts` |

---

## Priority Order

1. **High impact, low risk:** Add `getAllPositionsActivationDataFromJson` + `matchActivationRecordIgnorePosition` to activationUtils → remove ~100 lines
2. **High impact:** Create `utils/graphDisplayUtils.ts` → remove ~120 lines
3. **High impact:** Create `hooks/useCircuitBackend.ts` → remove ~600 lines
4. **Medium impact:** Create UI sub-components (ControlBar, etc.) → remove ~900 lines
5. **Low impact:** Move formatProbability, simplify parseNodeIdParts

**Estimated reduction:** From ~3384 to ~800-1000 lines after full extraction.
