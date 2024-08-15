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
import { useState } from "react";
import dagre from "dagre";

// const initialNodes: Node[] = [
//   { id: "1", position: { x: 0, y: 0 }, data: { label: "1" } },
//   { id: "12", position: { x: 300, y: 0 }, data: { label: "12" } },
//   { id: "2", position: { x: 0, y: 100 }, data: { label: "2" } },
// ];
// const initialEdges: Edge[] = [{ id: "e1-2", source: "1", target: "2" }];

export type CircuitViewerProps = {
  className?: string;
  flowClassName?: string;
  nodes: Node[];
  edges: Edge[];
  onNodesChange: React.Dispatch<React.SetStateAction<Node[]>>;
  onEdgesChange: React.Dispatch<React.SetStateAction<Edge[]>>;
};

export const CircuitViewer = ({
  className,
  flowClassName,
  nodes,
  edges,
  onNodesChange,
  onEdgesChange,
}: CircuitViewerProps) => {
  const [nodeIds, setNodeIds] = useState<string[]>([]);
  const [edgeIds, setEdgeIds] = useState<string[]>([]);
  const nodesHasChanged =
    !nodeIds.every((id) => nodes.some((node) => node.id === id)) || !nodes.every((node) => nodeIds.includes(node.id));
  const edgesHasChanged =
    !edgeIds.every((id) => edges.some((edge) => edge.id === id)) || !edges.every((edge) => edgeIds.includes(edge.id));

  const getLayoutedElements = (nodes: Node[], edges: Edge[]) => {
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

    const newNodes: Node[] = nodes.map((node) => {
      const nodeWithPosition = dagreGraph.node(node.id);
      const newNode: Node = {
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
  };

  if (nodesHasChanged || edgesHasChanged) {
    const layoutedElements = getLayoutedElements(nodes, edges);
    onNodesChange(layoutedElements.nodes);
    onEdgesChange(layoutedElements.edges);
    setNodeIds(layoutedElements.nodes.map((node) => node.id));
    setEdgeIds(layoutedElements.edges.map((edge) => edge.id));
  }

  return (
    <div className={cn("w-full h-[500px]", className)}>
      <ReactFlow
        className={flowClassName}
        nodes={nodes}
        edges={edges}
        onNodesChange={(changes) => {
          onNodesChange((nodes) => applyNodeChanges(changes, nodes));
        }}
        onEdgesChange={(changes) => {
          onEdgesChange((edges) => applyEdgeChanges(changes, edges));
        }}
      >
        <Controls />
        <MiniMap />
        <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
      </ReactFlow>
    </div>
  );
};
