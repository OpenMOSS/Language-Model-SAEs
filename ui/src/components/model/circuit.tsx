import { cn } from "@/lib/utils";
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  type Node,
  type Edge,
  applyNodeChanges,
  applyEdgeChanges,
  Position,
  Panel,
  NodeProps,
  Handle,
} from "@xyflow/react";
import { memo, useCallback, useEffect, useState } from "react";
import dagre from "dagre";
import { Tracing, TracingAction, TracingNode } from "@/types/model";
import { getAccentClassname } from "@/utils/style";
import { FeatureLinkWithPreview } from "../app/feature-preview";
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from "../ui/context-menu";

export type NodeData = {
  label: string;
  tracingNode: TracingNode;
  onTrace?: () => void;
  clearTracing?: () => void;
};

export type EdgeData = {
  attribution: number;
};

export type CircuitViewerProps = {
  className?: string;
  flowClassName?: string;
  tracings: Tracing[];
  disabled?: boolean;
  onTrace?: (node: TracingAction) => void;
  onTracingsChange?: (tracings: Tracing[]) => void;
};

const NodeInfo = ({ node }: { node: Node<NodeData> }) => {
  if (node.data.tracingNode.type === "logits") {
    return (
      <div className="rounded-lg border bg-card text-card-foreground shadow-sm p-4">
        <div className="grid grid-cols-2 gap-2">
          <div className="text-sm font-bold">Token ID:</div>
          <div className="text-sm">{node.data.tracingNode.tokenId}</div>
          <div className="text-sm font-bold">Position:</div>
          <div className="text-sm">{node.data.tracingNode.position}</div>
          <div className="text-sm font-bold">Logits:</div>
          <div className="text-sm">{node.data.tracingNode.activation.toFixed(3)}</div>
        </div>
      </div>
    );
  } else if (node.data.tracingNode.type === "attn-score") {
    return (
      <div className="rounded-lg border bg-card text-card-foreground shadow-sm p-4">
        <div className="grid grid-cols-2 gap-2">
          <div className="text-sm font-bold">Layer:</div>
          <div className="text-sm">{node.data.tracingNode.layer}</div>
          <div className="text-sm font-bold">Head:</div>
          <div className="text-sm">{node.data.tracingNode.head}</div>
          <div className="text-sm font-bold">Query:</div>
          <div className="text-sm">{node.data.tracingNode.query}</div>
          <div className="text-sm font-bold">Key:</div>
          <div className="text-sm">{node.data.tracingNode.key}</div>
          <div className="text-sm font-bold">Score:</div>
          <div className="text-sm">{node.data.tracingNode.activation.toFixed(3)}</div>
          <div className="text-sm font-bold">Pattern:</div>
          <div className={cn("text-sm", getAccentClassname(node.data.tracingNode.pattern, 1, "text"))}>
            {node.data.tracingNode.pattern.toFixed(3)}
          </div>
        </div>
      </div>
    );
  }
  return (
    <div className="rounded-lg border bg-card text-card-foreground shadow-sm p-4">
      <div className="grid grid-cols-2 gap-2">
        <div className="text-sm font-bold">SAE:</div>
        <div className="text-sm">{node.data.tracingNode.sae}</div>
        <div className="text-sm font-bold">Position:</div>
        <div className="text-sm">{node.data.tracingNode.position}</div>
        <div className="text-sm font-bold">Feature:</div>
        <div className="text-sm">
          <FeatureLinkWithPreview
            dictionaryName={node.data.tracingNode.sae}
            featureIndex={node.data.tracingNode.featureIndex}
          />
        </div>
        <div className="text-sm font-bold">Activation:</div>
        <div
          className={cn(
            "text-sm",
            getAccentClassname(node.data.tracingNode.activation, node.data.tracingNode.maxActivation, "text")
          )}
        >
          {node.data.tracingNode.activation.toFixed(3)}
        </div>
      </div>
    </div>
  );
};

const EdgeInfo = ({
  edge,
  source,
  target,
}: {
  edge: Edge<EdgeData>;
  source: Node<NodeData>;
  target: Node<NodeData>;
}) => {
  return (
    <div className="rounded-lg border bg-card text-card-foreground shadow-sm p-4">
      <div className="grid grid-cols-2 gap-2">
        <div className="text-sm font-bold">Source:</div>
        <div className="text-sm">{source.data.label}</div>
        <div className="text-sm font-bold">Target:</div>
        <div className="text-sm">{target.data.label}</div>
        <div className="text-sm font-bold">Attribution:</div>
        <div
          className={cn(
            "text-sm",
            getAccentClassname(edge.data!.attribution, target.data.tracingNode.activation, "text")
          )}
        >
          {edge.data!.attribution.toFixed(3)}
        </div>
      </div>
    </div>
  );
};

export const CircuitNode = ({
  data,
  isConnectable,
  targetPosition = Position.Top,
  sourcePosition = Position.Bottom,
}: NodeProps<Node<NodeData>>) => {
  
  // 根据节点类型获取颜色
  const getNodeColor = () => {
    const node = data?.tracingNode;
    if (!node) return "bg-gray-400";
    
    if (node.type === "feature") {
      const activation = node.activation;
      const maxActivation = node.maxActivation;
      const ratio = activation / maxActivation;
      
      if (ratio > 0.8) return "bg-red-500";
      if (ratio > 0.6) return "bg-orange-500";
      if (ratio > 0.4) return "bg-yellow-500";
      if (ratio > 0.2) return "bg-blue-500";
      return "bg-gray-400";
    } else if (node.type === "logits") {
      return "bg-purple-500";
    } else if (node.type === "attn-score") {
      return "bg-green-500";
    }
    return "bg-gray-400";
  };

  return (
    <>
      <Handle type="target" position={targetPosition} isConnectable={isConnectable} />
      
      {/* 圆形节点 */}
      <div
        className={cn(
          "w-6 h-6 rounded-full border border-white shadow-sm transition-all duration-200 cursor-pointer hover:scale-110 hover:shadow-md",
          getNodeColor()
        )}
        title={data?.label}
      />
      

      
      <Handle type="source" position={sourcePosition} isConnectable={isConnectable} />
    </>
  );
};

export const CircuitViewer = memo(
  ({ tracings, className, flowClassName, disabled, onTrace, onTracingsChange }: CircuitViewerProps) => {
    const [nodes, setNodes] = useState<Node<NodeData>[]>([]);
    const [edges, setEdges] = useState<Edge<EdgeData>[]>([]);
    const [selection, setSelection] = useState<{ nodes: Node<NodeData>[]; edges: Edge<EdgeData>[] }>({
      nodes: [],
      edges: [],
    });

    const getLayoutedElements = useCallback((nodes: Node<NodeData>[], edges: Edge<EdgeData>[]) => {
      const dagreGraph = new dagre.graphlib.Graph();
      dagreGraph.setDefaultEdgeLabel(() => ({}));
      dagreGraph.setGraph({ rankdir: "BT" });
      const nodeWidth = 24; // 更小的圆形节点宽度
      const nodeHeight = 24; // 更小的圆形节点高度

      nodes.forEach((node) => {
        dagreGraph.setNode(node.id, { width: nodeWidth, height: nodeHeight });
      });

      edges.forEach((edge) => {
        dagreGraph.setEdge(edge.source, edge.target);
      });

      dagre.layout(dagreGraph);

      const newNodes: Node<NodeData>[] = nodes.map((node) => {
        const nodeWithPosition = dagreGraph.node(node.id);
        
        // 检查这个节点是作为源节点还是目标节点
        const isSource = edges.some(edge => edge.source === node.id);
        const isTarget = edges.some(edge => edge.target === node.id);
        
        // 根据节点角色设置连接点
        let targetPosition = Position.Top;
        let sourcePosition = Position.Bottom;
        
        if (isSource && !isTarget) {
          // 纯源节点（只有出边，没有入边）- 从下方输出
          targetPosition = Position.Bottom;
          sourcePosition = Position.Bottom;
        } else if (!isSource && isTarget) {
          // 纯目标节点（只有入边，没有出边）- 从上方接收
          targetPosition = Position.Top;
          sourcePosition = Position.Top;
        } else {
          // 中间节点（既有入边又有出边）- 从上方接收，从下方输出
          targetPosition = Position.Top;
          sourcePosition = Position.Bottom;
        }
        
        const newNode: Node<NodeData> = {
          ...node,
          targetPosition,
          sourcePosition,
          // We are shifting the dagre node position (anchor=center center) to the top left
          // so it matches the React Flow node anchor point (top left).
          position: {
            x: nodeWithPosition.x - nodeWidth / 2,
            y: nodeWithPosition.y - nodeHeight / 2,
          },
        };

        return newNode;
      });

      return { nodes: newNodes, edges };
    }, []);

    const getNodeLabel = useCallback((node: TracingNode) => {
      if (node.type === "feature") {
        return `${node.position}.${node.sae}.${node.featureIndex}`;
      } else if (node.type === "attn-score") {
        return `L${node.layer}H${node.head}Q${node.query}K${node.key}`;
      } else {
        return `${node.position}.logits.${node.tokenId}`;
      }
    }, []);

    const getNodeClassNames = useCallback((node: TracingNode) => {
      if (node.type === "feature") {
        return cn(getAccentClassname(node.activation, node.maxActivation, "border"));
      } else if (node.type === "attn-score") {
        return cn(getAccentClassname(node.pattern, 1, "border"));
      }
      return "";
    }, []);

    const getEdgeBySourceAndTarget = useCallback(
      (sourceId: string, targetId: string) => {
        return edges.find((edge) => edge.source === sourceId && edge.target === targetId);
      },
      [edges]
    );

    const PanelContent = useCallback(() => {
      if (selection.nodes.length === 1) {
        return <NodeInfo node={selection.nodes[0]} />;
      } else if (selection.edges.length === 1) {
        return (
          <EdgeInfo
            edge={selection.edges[0]}
            source={nodes.find((n) => n.id === selection.edges[0].source)!}
            target={nodes.find((n) => n.id === selection.edges[0].target)!}
          />
        );
      } else if (selection.nodes.length === 2) {
        const edge =
          getEdgeBySourceAndTarget(selection.nodes[0].id, selection.nodes[1].id) ||
          getEdgeBySourceAndTarget(selection.nodes[1].id, selection.nodes[0].id);
        if (edge) {
          const source = nodes.find((n) => n.id === edge.source)!;
          const target = nodes.find((n) => n.id === edge.target)!;
          return <EdgeInfo edge={edge} source={source} target={target} />;
        }
      }
      return <></>;
    }, [selection, nodes, getEdgeBySourceAndTarget]);

    useEffect(() => {
      const nodes: Node<NodeData>[] = [];
      const edges: Edge<EdgeData>[] = [];

      tracings.forEach((tracing) => {
        const node: Node<NodeData> = {
          id: tracing.node.id,
          data: {
            label: getNodeLabel(tracing.node),
            tracingNode: tracing.node,
            onTrace: () => onTrace?.(tracing.node),
            clearTracing: () => {
              onTracingsChange?.(tracings.filter((t) => t.node.id !== tracing.node.id));
            },
          },
          className: cn("p-0", getNodeClassNames(tracing.node)),
          position: { x: 0, y: 0 },
        };

        nodes.push(node);

        tracing.contributors.forEach((contributor) => {
          const node: Node<NodeData> = {
            id: contributor.node.id,
            data: {
              label: getNodeLabel(contributor.node),
              tracingNode: contributor.node,
              onTrace: () => onTrace?.(contributor.node),
              clearTracing: () => {
                onTracingsChange?.(tracings.filter((t) => t.node.id !== contributor.node.id));
              },
            },
            className: cn("p-0", getNodeClassNames(contributor.node)),
            position: { x: 0, y: 0 },
          };

          nodes.push(node);

          const edge: Edge<EdgeData> = {
            id: `${contributor.node.id}-${tracing.node.id}`,
            source: contributor.node.id,
            target: tracing.node.id,
            data: {
              attribution: contributor.attribution,
            },
            className: cn(getAccentClassname(contributor.attribution, tracing.node.activation, "*:stroke")),
          };

          edges.push(edge);
        });
      });

      const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(nodes, edges);

      setNodes(layoutedNodes);
      setEdges(layoutedEdges);
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [tracings, getNodeLabel, getNodeClassNames, getLayoutedElements]);

    return (
      <div className={cn("w-full h-[500px]", className)}>
        <ReactFlow<Node<NodeData>, Edge<EdgeData>>
          className={cn("w-full h-full", flowClassName)}
          nodeTypes={{ default: CircuitNode }}
          nodes={nodes}
          edges={edges}
          onNodesChange={(changes) => {
            setNodes((nodes) => applyNodeChanges(changes, nodes));
          }}
          onEdgesChange={(changes) => {
            setEdges((edges) => applyEdgeChanges(changes, edges));
          }}
          onNodeDoubleClick={(_, node) => {
            !disabled && onTrace?.(node.data.tracingNode);
          }}
          onSelectionChange={setSelection as (selection: { nodes: Node[]; edges: Edge[] }) => void}
          fitView
          fitViewOptions={{ padding: 0.1 }}
        >
          <Controls />
          <MiniMap />
          <Panel position="top-right">
            <PanelContent />
          </Panel>
          <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
        </ReactFlow>
      </div>
    );
  }
);

CircuitViewer.displayName = "CircuitViewer";
