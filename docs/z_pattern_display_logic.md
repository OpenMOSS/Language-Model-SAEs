# Feature samples 里 z_pattern 的展示逻辑与代码位置

## 为什么「以该格为 source 的边」会少于 64 条？

概念上 z_pattern 可以是 64×64（每个 source 对 64 个 target）。但**后端在写入时做了稀疏化**，只保存“显著”的边：

- **位置**: `src/lm_saes/analysis/post_analysis/lorsa.py`（LorsaPostAnalysisProcessor）
- **逻辑**（约 157–161 行）:
  - 对每个 (q_pos, k_pos) 的 z 值做阈值：`|z_pattern| < 1e-2 * feature_activation_at_q` 的置 0。
  - 然后 `z_pattern.to_sparse()`，**只把非零项**写入 `z_pattern_indices` / `z_pattern_values`。
- **公式**: 只有满足 `|z_pattern(q,k)| ≥ 0.01 × activation(q)` 的 (q, k) 才会被存下来。

所以 DB/API 里每条 sample 的 z_pattern **不是**完整 64×64，而是稀疏列表；对某个 source 来说，存下来的边可能只有几条到几十条。前端再按 source 过滤并取 top 8，所以悬停时看到少于 8 条是**数据里该 source 的边本来就少于 8 条**，没有额外的前端过滤。

---

## 数据从哪来（重要）

- **Feature 页看到的 samples** 来自后端 API：`GET /dictionaries/{dictionary}/features/{featureIndex}`。
- 后端在 **server/app.py** 里用 **DB 里存的 analysis 的 samplings**（`feature_acts_indices`、`feature_acts_values`、`z_pattern_indices`、`z_pattern_values`）拼成每个 sample，**原样**把 z_pattern 下发给前端，**没有任何过滤**。
- **Autointerp 不参与这条链路**：autointerp 只在内存里生成 `TokenizedSample`（带 `z_pattern_data`），用于生成 prompt（如 `display_max`），**不会**把这些 sample 写成 API/DB 里那种带 `z_pattern_indices` / `z_pattern_values` 的格式，也不会被 feature 页用到。所以「前端在显示 feature samples 时如何展示 z_pattern」和 autointerp 无关。

---

## 前端展示 z_pattern 的代码位置

### 1. Feature 页：从 API 拿到 sample，交给棋盘

- **文件**: `ui/src/routes/features/page.tsx`
- **流程**:
  - 用 `fetchFeature(dictionary, featureIndex)` 拉取 feature（含 `sample_groups[].samples`）。
  - 每个 sample 里可能有 `zPatternIndices`、`zPatternValues`（camelCase 由 camelcaseKeys 从后端的 `z_pattern_indices`、`z_pattern_values` 转来）。
  - 遍历 `group.samples`，从每个 sample 里**直接取** `(sample as any).zPatternIndices` 和 `(sample as any).zPatternValues`，**不做任何过滤**，只做格式兼容：
    - 若 `zPatternIndices[0]` 是长度为 2 的数组，当成「[[source, target], ...]」；
    - 否则当成「[sources[], targets[]]」。
  - 把 `zPatternIndices`、`zPatternValues` 放进 `chessSamples[]`，再传给 `ChessBoard`。

**相关行号**（约）:

- 478–483: 从 sample 读 `zPatternIndices` / `zPatternValues`，格式归一化，**无过滤**。
- 552–557: 把 `chessSample.zPatternIndices`、`chessSample.zPatternValues` 传给 `ChessBoard`。

```tsx
// 478-483: 只做格式判断，没有按数值过滤
if ((sample as any).zPatternIndices && (sample as any).zPatternValues) {
  const zpIdxRaw = (sample as any).zPatternIndices;
  zPatternIndices = Array.isArray(zpIdxRaw) && Array.isArray(zpIdxRaw[0]) ? zpIdxRaw : [zpIdxRaw];
  zPatternValues = (sample as any).zPatternValues;
}
// ...
<ChessBoard
  zPatternIndices={chessSample.zPatternIndices}
  zPatternValues={chessSample.zPatternValues}
  // ...
/>
```

### 2. 棋盘组件：唯一做「只显示较强连接」的地方

- **文件**: `ui/src/components/chess/chess-board.tsx`
- **函数**: `getZPatternTargets(sourceSquare, zPatternIndices, zPatternValues)`（约 108–149 行）

**逻辑**:

1. 根据当前 hover 的格子 `sourceSquare`，从 `zPatternIndices` + `zPatternValues` 里解析出所有以该格为 source 的 `(target, strength)`。
2. 支持两种格式（与 feature 页一致）：
   - `[[source, target], ...]` + `[value, ...]`
   - `[sources[], targets[]]` + `[value, ...]`
3. **这里才有过滤**：按 `|strength|` 从大到小排序，然后 **只取前 8 条**：`.sort(...).slice(0, 8)`。

**相关行号**:

- 108–149: `getZPatternTargets` 实现；**过滤逻辑在 146–149 行**。
- 487: 调用 `getZPatternTargets(hoveredSquare, zPatternIndices, zPatternValues)` 得到当前要画的连接。
- 543–544: 用 `zPatternTargets` 判断某格是否是「z_pattern 目标」并取 strength 上色。
- 781–795: 悬浮详情里只展示前 6 条（`.slice(0, 6)`），再多就显示 “… and N more”。

```ts
// 146-149: 唯一的「只显示比较大的几个」的逻辑
return targets
  .sort((a, b) => Math.abs(b.strength) - Math.abs(a.strength))
  .slice(0, 8);
```

---

## 是否存在过滤、在哪

| 位置 | 是否过滤 z_pattern | 说明 |
|------|--------------------|------|
| 后端 API (server/app.py) | 否 | 从 DB 的 sampling 里取 `z_pattern_indices` / `z_pattern_values`，原样放进 sample 返回。 |
| Feature 页 (features/page.tsx) | 否 | 只做格式兼容（pair list vs [sources, targets]），整份传给 ChessBoard。 |
| 棋盘 (chess-board.tsx) | **是** | 在 `getZPatternTargets` 里：按 \|strength\| 排序，只保留 **top 8**；悬浮文案再取前 6 条。 |

也就是说：**只有棋盘组件里有过滤**，且是「按强度排序 + 取前 8 条」，没有用类似 0.7×max 的阈值。

---

## Autointerp 和 feature 页 samples 的关系

- Autointerp 里会为每个 feature 生成 `activating_examples`（`TokenizedSample` 列表），里面通过 `add_z_pattern_data` 存了 `z_pattern_data`（按 segment 的 contributing indices + contributions）。
- 这些 sample 只用于：
  - 生成解释/检测/ fuzzing 的 **prompt**（例如 `display_max(threshold=0.7)` 会按 0.7×max_contribution 过滤，只影响**文本**）；
  - 没有在 autointerp 里被转成 `z_pattern_indices` / `z_pattern_values` 的数组格式，也没有写回 DB 或通过 feature API 返回。
- Feature 页拉的是 **analysis 的 samplings**（例如 lorsa 等 pipeline 写入的），不是 autointerp 的输出。所以 **autointerp 目前没有「把 z_pattern 推到 feature 页展示」这一步**。

如果希望 autointerp 生成的 sample 也在 feature 页用同样方式展示 z_pattern，需要在 autointerp 侧把 `TokenizedSample.z_pattern_data` 转成 API 期望的 `z_pattern_indices` / `z_pattern_values` 并写入/返回给前端（或单独接口），前端展示逻辑可以继续用上面同一套（feature 页 + ChessBoard + getZPatternTargets）。
