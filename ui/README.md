# Language Model SAEs UI

This is the React-based UI for the Language Model SAEs project, built with Vite, TypeScript, and Tailwind CSS.

## Features

### Circuit Visualization
- **Link Graph**: Interactive visualization of circuit connections between features
- **Node Connections**: Right-side panel showing input and output connections for selected features
- **Interactive Features**: Click nodes to see connections, Cmd/Ctrl+Click to pin nodes
- **Responsive Layout**: Two-column layout with the link graph on the left and node connections on the right

### Node Connections Component
The `NodeConnections` component replicates the functionality of the original JavaScript `init-cg-node-connections.js` file:

- **Feature Header**: Shows feature ID, type, and description with click interaction
- **Input Features**: Displays incoming connections, separated by positive/negative weights
- **Output Features**: Shows outgoing connections, separated by positive/negative weights
- **Interactive Elements**: Clickable feature rows with hover states and pinned indicators
- **Weight Display**: Shows connection weights and percentage contributions

## Usage

### Circuit Visualization
1. Upload a JSON file with circuit data
2. Click on nodes in the link graph to see their connections
3. Use Cmd/Ctrl+Click to pin important nodes
4. View detailed connection information in the right panel

### State Management
The circuit visualization uses React Context for state management:
- `clickedId`: Currently selected node
- `hoveredId`: Node being hovered over
- `pinnedIds`: Array of pinned node IDs
- `hiddenIds`: Array of hidden node IDs

## Development

### Prerequisites
- Node.js 18+
- Bun package manager

### Setup
```bash
cd ui
bun install
bun run dev
```

### Building
```bash
bun run build
```

### Testing
```bash
bun run test
```

## Component Structure

```
circuits/
├── circuit-visualization.tsx    # Main circuit visualization component
├── link-graph-container.tsx     # Container for the link graph
├── link-graph/                  # Link graph visualization components
├── node-connections.tsx         # Node connections panel (NEW)
└── ...
```

## Data Format

The UI expects circuit data in the following format:
```typescript
interface LinkGraphData {
  nodes: Node[];
  links: Link[];
  metadata: {
    prompt_tokens: string[];
    lorsa_analysis_name?: string;
    clt_analysis_name?: string;
  };
}
```

Each node should include:
- `nodeId`: Unique identifier
- `feature_type`: Type of feature (e.g., 'cross layer transcoder', 'transcoder')
- `localClerp`/`remoteClerp`: Human-readable feature descriptions
- `sourceLinks`/`targetLinks`: Connection information with weights and percentages
