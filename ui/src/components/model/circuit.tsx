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
} from "@xyflow/react";
import { useCallback, useEffect, useState } from "react";
import dagre from "dagre";
import { Tracing, TracingAction, TracingNode } from "@/types/model";

export type NodeData = {
  label: string;
  tracingNode: TracingNode;
};

export type CircuitViewerProps = {
  className?: string;
  flowClassName?: string;
  tracings: Tracing[];
  onTrace?: (node: TracingAction) => void;
};

export const CircuitViewer = ({ tracings, className, flowClassName, onTrace }: CircuitViewerProps) => {
  const [nodes, setNodes] = useState<Node<NodeData>[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);

  const getLayoutedElements = useCallback((nodes: Node<NodeData>[], edges: Edge[]) => {
    const dagreGraph = new dagre.graphlib.Graph();
    dagreGraph.setDefaultEdgeLabel(() => ({}));
    dagreGraph.setGraph({ rankdir: "TB" });
    const nodeWidth = 172;
    const nodeHeight = 36;

    nodes.forEach((node) => {
      dagreGraph.setNode(node.id, { width: nodeWidth, height: nodeHeight });
    });

    edges.forEach((edge) => {
      dagreGraph.setEdge(edge.source, edge.target);
    });

    dagre.layout(dagreGraph);

    const newNodes: Node<NodeData>[] = nodes.map((node) => {
      const nodeWithPosition = dagreGraph.node(node.id);
      const newNode: Node<NodeData> = {
        ...node,
        targetPosition: Position.Top,
        sourcePosition: Position.Bottom,
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
    } else {
      return `${node.position}.logits.${node.tokenId}`;
    }
  }, []);

  useEffect(() => {
    const nodes: Node<NodeData>[] = [];
    const edges: Edge[] = [];

    tracings.forEach((tracing) => {
      const node: Node<NodeData> = {
        id: tracing.node.id,
        data: {
          label: getNodeLabel(tracing.node),
          tracingNode: tracing.node,
        },
        position: { x: 0, y: 0 },
      };

      nodes.push(node);

      tracing.contributors.forEach((contributor) => {
        const node: Node<NodeData> = {
          id: contributor.node.id,
          data: {
            label: getNodeLabel(contributor.node),
            tracingNode: contributor.node,
          },
          position: { x: 0, y: 0 },
        };

        nodes.push(node);

        const edge: Edge = {
          id: `${contributor.node.id}-${tracing.node.id}`,
          source: contributor.node.id,
          target: tracing.node.id,
        };

        edges.push(edge);
      });
    });

    const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(nodes, edges);

    setNodes(layoutedNodes);
    setEdges(layoutedEdges);
  }, [tracings]);

  return (
    <div className={cn("w-full h-[500px]", className)}>
      <ReactFlow
        className={flowClassName}
        nodes={nodes}
        edges={edges}
        onNodesChange={(changes) => {
          setNodes((nodes) => applyNodeChanges(changes, nodes));
        }}
        onEdgesChange={(changes) => {
          setEdges((edges) => applyEdgeChanges(changes, edges));
        }}
        onNodeDoubleClick={(_, node) => {
          onTrace?.(node.data.tracingNode);
        }}
      >
        <Controls />
        <MiniMap />
        <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
      </ReactFlow>
    </div>
  );
};
