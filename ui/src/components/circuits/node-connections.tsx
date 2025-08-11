import React, { useMemo } from 'react';
import { Node, Link, LinkGraphData } from './link-graph/types';

interface NodeConnectionsProps {
  data: LinkGraphData;
  clickedId: string | null;
  hoveredId: string | null;
  pinnedIds: string[];
  hiddenIds: string[];
  onFeatureClick: (node: Node, isMetaKey: boolean) => void;
}

interface ConnectionSection {
  title: string;
  nodes: Node[];
}

interface ConnectionType {
  id: 'input' | 'output';
  title: string;
  sections: ConnectionSection[];
}

export const NodeConnections: React.FC<NodeConnectionsProps> = ({
  data,
  clickedId,
  hoveredId,
  pinnedIds,
  hiddenIds,
  onFeatureClick,
}) => {  
  console.log('🔄 NodeConnections recomputed', { 
    dataNodesCount: data.nodes.length,
    dataLinksCount: data.links.length,
    clickedId,
    hoveredId,
    pinnedIdsCount: pinnedIds.length,
    hiddenIdsCount: hiddenIds.length
  });

  // Memoize the clicked node to avoid re-finding it on every render
  const clickedNode = useMemo(() => 
    data.nodes.find(node => node.nodeId === clickedId), 
    [data.nodes, clickedId]
  );

  // Memoize the connection types computation - this is expensive and should only recalculate when relevant data changes
  const connectionTypes = useMemo((): ConnectionType[] => {
    if (!clickedNode || !clickedNode.sourceLinks || !clickedNode.targetLinks) {
      return [];
    }
    
    // Input features: nodes that have links TO the clicked node
    // These are nodes where the clicked node is the TARGET of their links
    const inputNodes = data.nodes.filter(node => 
      node.nodeId !== clickedNode.nodeId && // Exclude the clicked node itself
      node.sourceLinks && // Ensure node has sourceLinks property
      node.sourceLinks.some(link => link.target === clickedNode.nodeId)
    );
    
    // Output features: nodes that the clicked node has links TO
    // These are nodes where the clicked node is the SOURCE of links to them
    const outputNodes = data.nodes.filter(node => 
      node.nodeId !== clickedNode.nodeId && // Exclude the clicked node itself
      clickedNode.sourceLinks && // Ensure clicked node has sourceLinks property
      clickedNode.sourceLinks.some(link => link.target === node.nodeId)
    );

    // Debug logging to verify the fix
    console.log('🔗 Connection analysis:', {
      clickedNodeId: clickedNode.nodeId,
      clickedNodeSourceLinksCount: clickedNode.sourceLinks?.length || 0,
      clickedNodeTargetLinksCount: clickedNode.targetLinks?.length || 0,
      inputNodesCount: inputNodes.length,
      outputNodesCount: outputNodes.length,
      inputNodeIds: inputNodes.map(n => n.nodeId),
      outputNodeIds: outputNodes.map(n => n.nodeId)
    });

    return [
      {
        id: 'input',
        title: 'Input Features',
        sections: ['Positive', 'Negative'].map(title => {
          const nodes = inputNodes.filter(node => {
            // Find the link from this node TO the clicked node
            const link = node.sourceLinks?.find(link => link.target === clickedNode.nodeId);
            if (!link || link.weight === undefined) return false;
            return title === 'Positive' ? link.weight > 0 : link.weight < 0;
          });
          
          // Sort by absolute weight (descending)
          nodes.sort((a, b) => {
            const linkA = a.sourceLinks?.find(link => link.target === clickedNode.nodeId);
            const linkB = b.sourceLinks?.find(link => link.target === clickedNode.nodeId);
            const weightA = Math.abs(linkA?.weight || 0);
            const weightB = Math.abs(linkB?.weight || 0);
            return weightB - weightA;
          });
          
          return { title, nodes };
        }),
      },
      {
        id: 'output',
        title: 'Output Features',
        sections: ['Positive', 'Negative'].map(title => {
          const nodes = outputNodes.filter(node => {
            // Find the link from the clicked node TO this node
            const link = clickedNode.sourceLinks?.find(link => link.target === node.nodeId);
            if (!link || link.weight === undefined) return false;
            return title === 'Positive' ? link.weight > 0 : link.weight < 0;
          });
          
          // Sort by absolute weight (descending)
          nodes.sort((a, b) => {
            const linkA = clickedNode.sourceLinks?.find(link => link.target === a.nodeId);
            const linkB = clickedNode.sourceLinks?.find(link => link.target === b.nodeId);
            const weightA = Math.abs(linkA?.weight || 0);
            const weightB = Math.abs(linkB?.weight || 0);
            return weightB - weightA;
          });
          
          return { title, nodes };
        }),
      },
    ];
  }, [data.nodes, clickedNode?.nodeId, clickedNode?.sourceLinks, clickedNode?.targetLinks]);

  // Memoize the formatFeatureId function to avoid recreating it on every render
  const formatFeatureId = useMemo(() => (node: Node): string => {
    if (node.feature_type === 'cross layer transcoder') {
      const layerIdx = Math.floor(node.layerIdx / 2) - 1;
      const featureId = node.id.split('_')[1];
      return `M${layerIdx}#${featureId}@${node.ctx_idx}`;
    } else if (node.feature_type === 'lorsa') {
      const layerIdx = Math.floor(node.layerIdx / 2);
      const featureId = node.id.split('_')[1];
      return `A${layerIdx}#${featureId}@${node.ctx_idx}`;
    } else if (node.feature_type === 'embedding') {
      return `Emb@${node.ctx_idx}`;
    } else if (node.feature_type === 'mlp reconstruction error') {
      return `M${Math.floor(node.layerIdx / 2) - 1}Error@${node.ctx_idx}`;
    } else if (node.feature_type === 'lorsa error') {
      return `A${Math.floor(node.layerIdx / 2)}Error@${node.ctx_idx}`;
    }
    return ' ';
  }, []);

  // Memoize the header click handler to prevent unnecessary re-renders
  const handleHeaderClick = useMemo(() => (event: React.MouseEvent) => {
    if (!clickedNode) return;
    const isMetaKey = event.metaKey || event.ctrlKey;
    onFeatureClick(clickedNode, isMetaKey);
  }, [clickedNode, onFeatureClick]);

  // Memoize the feature row renderer to avoid recreating the function on every render
  const renderFeatureRow = useMemo(() => (node: Node, type: 'input' | 'output') => {
    if (!clickedNode) return null;
    
    // Find the link in the correct direction
    const link = type === 'input' 
      ? node.sourceLinks?.find(link => link.target === clickedNode.nodeId)  // Link FROM this node TO clicked node
      : clickedNode.sourceLinks?.find(link => link.target === node.nodeId); // Link FROM clicked node TO this node
    
    if (!link || link.weight === undefined) return null;

    const weight = link.weight;
    const pctInput = link.pctInput || 0;
    const isPinned = pinnedIds.includes(node.nodeId);
    const isHidden = hiddenIds.includes(node.featureId);
    const isHovered = node.nodeId === hoveredId;
    const isClicked = node.nodeId === clickedId;

    return (
      <div
        key={node.nodeId}
        className={`feature-row p-1 border rounded cursor-pointer transition-colors ${
          isPinned ? 'bg-yellow-100 border-yellow-300' : 'bg-gray-50 border-gray-200'
        } ${isHidden ? 'opacity-50' : ''} ${isHovered ? 'ring-2 ring-blue-300' : ''} ${
          isClicked ? 'ring-2 ring-blue-500' : ''
        }`}
        onClick={() => onFeatureClick(node, false)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="text-sm font-mono text-gray-600">
              {formatFeatureId(node)}
            </span>
            <span className="text-sm font-medium">
              {node.localClerp || node.remoteClerp || ''}
            </span>
          </div>
          <div className="text-right">
            <div className="text-sm font-mono">
              {weight > 0 ? '+' : ''}{weight.toFixed(3)}
            </div>
            <div className="text-xs text-gray-500">
              {pctInput.toFixed(1)}%
            </div>
          </div>
        </div>
      </div>
    );
  }, [clickedNode, pinnedIds, hiddenIds, hoveredId, clickedId, onFeatureClick, formatFeatureId]);

  // Memoize the header styling to avoid recalculating classes on every render
  const headerClassName = useMemo(() => 
    `header-top-row section-title mb-3 cursor-pointer p-2 rounded-lg border ${
      clickedNode && pinnedIds.includes(clickedNode.nodeId)
        ? 'bg-yellow-50 border-yellow-200 text-yellow-800' 
        : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
    }`, 
    [pinnedIds, clickedNode?.nodeId]
  );

  // Early return after all hooks have been called
  if (!clickedNode) {
    return (
      <div className="node-connections flex flex-col h-full overflow-y-auto">
        <div className="header-top-row section-title mb-3">
          Click a feature on the left for details
        </div>
      </div>
    );
  }

  return (
    <div className="node-connections flex flex-col h-full overflow-y-auto">
      {/* Header */}
      <div 
        className={headerClassName}
        onClick={handleHeaderClick}
      >
        <span className="inline-block mr-2 font-mono tabular-nums w-20 text-sm">
          {formatFeatureId(clickedNode)}
        </span>
        <span className="feature-title font-medium text-sm">
          {clickedNode.localClerp || clickedNode.remoteClerp || ''}
        </span>
      </div>

      {/* Connections */}
      <div className="connections flex-1 flex overflow-hidden gap-5">
        {connectionTypes.map(type => (
          <div
            key={type.id}
            className={`features flex-1 ${type.id === 'output' ? 'output' : 'input'}`}
          >
            <div className="section-title text-lg font-semibold mb-2 text-gray-800">
              {type.title}
            </div>
            
            <div className="effects space-y-2">
              {type.sections.map(section => (
                <div key={section.title} className="section">
                  <h4 className={`text-sm font-medium mb-1 px-2 py-1 rounded ${
                    section.title === 'Positive' 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-red-100 text-red-800'
                  }`}>
                    {section.title}
                  </h4>
                  <div className="space-y-1">
                    {section.nodes.map(node => renderFeatureRow(node, type.id))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}; 