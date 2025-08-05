# 国际象棋棋盘组件

这个组件用于在国际象棋SAE特征分析系统中显示棋盘位置和最佳移动箭头。

## 功能特性

- 🎯 **自动解析longfen格式**：从棋类样本数据中自动提取棋盘状态和移动信息
- 🏹 **移动箭头显示**：用红色箭头清晰显示最佳移动路径
- 🎨 **美观的棋盘界面**：使用标准的国际象棋棋盘配色和棋子符号
- 📍 **坐标标注**：每个格子都显示标准国际象棋坐标
- 🔄 **支持所有移动类型**：包括普通移动、王车易位、升变等

## 文件结构

```
ui/src/components/chess/
├── simple-chess-board.tsx    # 简单棋盘组件（原生DOM）
├── chess-board.tsx          # React棋盘组件（可选）
├── index.ts                 # 导出文件
├── demo.html                # 独立演示页面
└── README.md               # 说明文档
```

## 使用方法

### 1. 在特征卡片中显示棋盘

当特征包含棋类样本时，系统会自动检测并显示棋盘：

```typescript
import { createChessBoardElement } from "../chess";

// 在FeatureCard组件中
const isChessFeature = useMemo(() => {
  return feature.sampleGroups.some(group => 
    group.samples.some(sample => 
      sample.text && sample.text.length > 77 && 
      /^[wb][rnbqkbnrpppppppp\.]{64}[KQkq\.]{4}[a-h][1-8]?\.{0,2}\d{1,3}\.\d{1,3}[a-h][1-8][a-h][1-8][qrbn]?0$/.test(sample.text)
    )
  );
}, [feature.sampleGroups]);
```

### 2. 手动创建棋盘

```typescript
import { createChessBoardElement } from "../chess";

// 创建棋盘元素
const chessBoardElement = createChessBoardElement({
  longfen: "wrnbqkbnrpppppppp................................PPPPPPPPRNBQKBNRKQkq..0..1..e2e40",
  size: 400,
  className: "mx-auto"
});

// 添加到DOM
container.appendChild(chessBoardElement);
```

### 3. 查看演示

打开 `demo.html` 文件可以在浏览器中查看完整的演示效果。

## longfen格式说明

longfen是扩展的FEN格式，包含以下部分：

| 位置 | 长度 | 说明 | 示例 |
|------|------|------|------|
| 1 | 1 | 当前走棋方 | w=白方, b=黑方 |
| 2-65 | 64 | 棋盘位置 | rnbqkbnrpppppppp... |
| 66-69 | 4 | 王车易位权利 | KQkq |
| 70-71 | 2 | 过路兵目标格 | .. |
| 72-74 | 3 | 半回合计数 | 0.. |
| 75-77 | 3 | 全回合计数 | 1.. |
| 78-81 | 4 | 移动(UCI格式) | e2e4 |
| 82 | 1 | 结束标记 | 0 |

## 移动箭头样式

- **箭头颜色**：红色 (#ff4444)
- **线条粗细**：3px
- **起点标记**：空心圆圈
- **终点标记**：实心圆圈
- **透明度**：0.8

## 支持的移动类型

1. **普通移动**：e2e4, d7d5
2. **王车易位**：e1g1 (王翼), e1c1 (后翼)
3. **升变移动**：e7e8q, e2e1n
4. **吃子移动**：e4d5, f3e4

## 技术实现

### 核心函数

- `parseLongfen(longfen: string)`: 解析longfen字符串为棋盘状态
- `extractMoveFromLongfen(longfen: string)`: 从longfen中提取移动信息
- `parseUciMove(uciMove: string)`: 解析UCI格式的移动
- `createChessBoardElement(props)`: 创建棋盘DOM元素

### 依赖

- 原生DOM API
- SVG绘图
- CSS样式

## 示例

### 标准开局 (e2e4)
```
longfen: wrnbqkbnrpppppppp................................PPPPPPPPRNBQKBNRKQkq..0..1..e2e40
```

### 王车易位 (O-O)
```
longfen: wrnbqkbnrpppppppp................................PPPPPPPPRNBQKBNRKQkq..0..1..e1g10
```

### 升变移动 (e7e8q)
```
longfen: wrnbqkbnrpppppppp................................PPPPPPPPRNBQKBNRKQkq..0..1..e7e8q0
```

## 注意事项

1. **longfen格式验证**：确保输入的longfen格式正确
2. **移动解析**：移动信息必须在longfen的最后5个字符中
3. **浏览器兼容性**：需要支持SVG的现代浏览器
4. **样式依赖**：需要Tailwind CSS类名支持

## 故障排除

### 常见问题

1. **棋盘不显示**：检查longfen格式是否正确
2. **箭头位置错误**：确认移动信息解析正确
3. **样式问题**：确保CSS类名正确加载

### 调试方法

1. 打开浏览器开发者工具
2. 检查控制台错误信息
3. 验证longfen格式
4. 测试移动解析函数

## 扩展功能

未来可以考虑添加的功能：

- [ ] 动画效果
- [ ] 交互式棋盘
- [ ] 移动历史显示
- [ ] 多种箭头样式
- [ ] 移动验证
- [ ] 棋局分析 