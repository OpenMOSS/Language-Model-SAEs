import React, { useCallback, useEffect, useRef, useState } from "react";
import * as d3 from "d3";

interface RawAtlasNodeNew {
  node_id: string;
  autointerp?: string;
}

interface RawAtlasNodeOld {
  [nodeId: string]: string;
}

type RawAtlasNode = RawAtlasNodeNew | RawAtlasNodeOld;

interface RawAtlasLink {
  source: string;
  target: string;
  weight: number;
}

interface AtlasGraphData {
  nodes: RawAtlasNode[];
  links: RawAtlasLink[];
}

interface ParsedNodeId {
  layer: number;
  type: "lorsa" | "plt";
  feature: number;
}

interface AtlasNode {
  id: string;
  autointerp?: string;
  index: number;
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
}

interface AtlasLink {
  source: AtlasNode;
  target: AtlasNode;
  weight: number;
}

const parseNodeId = (nodeId: string): ParsedNodeId | null => {
  const match = nodeId.match(/^L(\d+)([AM])#(\d+)$/);
  if (!match) {
    return null;
  }
  const layer = Number.parseInt(match[1], 10);
  const type = match[2] === "A" ? "lorsa" : "plt";
  const feature = Number.parseInt(match[3], 10);
  return { layer, type, feature };
};

const generateIframeUrl = (
  layer: number,
  type: "lorsa" | "plt",
  feature: number,
): string => {
  const saeType = type === "lorsa" ? "lorsa" : "plt";
  const origin =
    typeof window !== "undefined" ? window.location.origin : "http://localhost:3000";
  return `${origin}/embed/dictionaries/qwen3-1.7b-${saeType}-32x-topk128-layer${layer}/features/${feature}`;
};

export const AtlasVisualization: React.FC = () => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const simulationRef = useRef<any>(null);
  const allLinksRef = useRef<AtlasLink[]>([]);

  const [graphData, setGraphData] = useState<AtlasGraphData | null>(null);
  const [selectedAtlas, setSelectedAtlas] = useState<string>(
    "circuits/atlases/[Thr]Opp. One-move mate threaten/atlas_0.json",
  );
  const [status, setStatus] = useState<string>("");
  const [sliderValue, setSliderValue] = useState<number>(0);
  const [strengthThreshold, setStrengthThreshold] = useState<number>(0);
  const [maxEdgeWeight, setMaxEdgeWeight] = useState<number>(0);

  const [iframeUrl, setIframeUrl] = useState<string | null>(null);
  const [iframeTitle, setIframeTitle] = useState<string | null>(null);

  const showNodeIframe = useCallback((nodeId: string): void => {
    const parsed = parseNodeId(nodeId);
    if (!parsed) {
      // eslint-disable-next-line no-console
      console.warn("Invalid node ID format:", nodeId);
      return;
    }
    const url = generateIframeUrl(parsed.layer, parsed.type, parsed.feature);
    setIframeUrl(url);
    setIframeTitle(
      `Feature: ${nodeId} (Layer ${parsed.layer}, ${parsed.type}, Feature ${parsed.feature})`,
    );
  }, []);

  const loadAtlasFile = useCallback(async (filePath: string): Promise<void> => {
    if (!filePath) {
      return;
    }
    try {
      setStatus("Loading atlas...");
      const baseUrl = import.meta.env.VITE_BACKEND_URL as string | undefined;
      const url =
        filePath.startsWith("http://") || filePath.startsWith("https://")
          ? filePath
          : baseUrl
            ? `${baseUrl.replace(/\/$/, "")}/${filePath.replace(/^\//, "")}`
            : filePath;
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error("File not found or empty");
      }
      const data = (await response.json()) as AtlasGraphData;
      if (!data || Object.keys(data).length === 0) {
        setStatus("File is empty. Please generate the graph data first.");
        return;
      }
      setGraphData(data);
      allLinksRef.current = [];
      setSliderValue(0);
      setStrengthThreshold(0);
      setMaxEdgeWeight(0);
      setIframeUrl(null);
      setIframeTitle(null);
      setStatus("Graph loaded successfully!");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setStatus(`Error: ${message}`);
    }
  }, []);

  const visualizeGraph = useCallback((): void => {
    if (!graphData || !graphData.nodes || !graphData.links || !svgRef.current) {
      if (!graphData) {
        return;
      }
      setStatus("Invalid graph data format");
      return;
    }

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const containerWidth = svgRef.current.clientWidth;
    const width =
      Number(svgRef.current.getAttribute("width")) || containerWidth || 2000;
    const height = Number(svgRef.current.getAttribute("height")) || 1600;

    const g = svg.append("g").attr("class", "graph-container");

    const zoom = d3
      .zoom()
      .scaleExtent([0.1, 4])
      .on("zoom", (event: any) => {
        g.attr("transform", event.transform.toString());
      });

    svg.call(zoom as any);

    const nodes: AtlasNode[] = graphData.nodes.map((node, index) => {
      let nodeId: string;
      let autointerp = "";
      if ("node_id" in node) {
        nodeId = node.node_id;
        autointerp = node.autointerp ?? "";
      } else {
        const key = Object.keys(node)[0];
        nodeId = key;
      }
      return {
        id: nodeId,
        autointerp,
        index,
      };
    });

    let links: AtlasLink[] = [];
    if (allLinksRef.current.length === 0) {
      links = graphData.links
        .map((link) => {
          const source = nodes.find((n) => n.id === link.source);
          const target = nodes.find((n) => n.id === link.target);
          if (!source || !target) {
            return null;
          }
          return {
            source,
            target,
            weight: link.weight * 100,
          };
        })
        .filter((l): l is AtlasLink => l !== null);

      allLinksRef.current = links;
      const maxWeight =
        allLinksRef.current.length > 0
          ? Math.max(...allLinksRef.current.map((link) => Math.abs(link.weight)))
          : 0;
      setMaxEdgeWeight(maxWeight);
    } else {
      links = allLinksRef.current;
    }

    const filteredLinks = links.filter(
      (link) => Math.abs(link.weight) >= strengthThreshold,
    );

    const connectedNodeIds = new Set<string>();
    filteredLinks.forEach((link) => {
      connectedNodeIds.add(link.source.id);
      connectedNodeIds.add(link.target.id);
    });

    const connectedNodes = nodes.filter((node) =>
      connectedNodeIds.has(node.id),
    );

    const nodeMap = new Map<string, AtlasNode>(
      connectedNodes.map((node) => [node.id, node]),
    );
    const processedLinks: AtlasLink[] = filteredLinks
      .map((link) => {
        const source = nodeMap.get(link.source.id);
        const target = nodeMap.get(link.target.id);
        if (!source || !target) {
          return null;
        }
        return {
          source,
          target,
          weight: link.weight,
        };
      })
      .filter((l): l is AtlasLink => l !== null);

    if (simulationRef.current) {
      simulationRef.current.stop();
    }

    const simulation = d3
      .forceSimulation(connectedNodes)
      .force(
        "link",
        d3
          .forceLink(processedLinks)
          .id((d: AtlasNode) => d.id)
          .distance(100),
      )
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(20));

    simulationRef.current = simulation;

    const link = g
      .append("g")
      .attr("class", "links")
      .selectAll("line")
      .data(processedLinks)
      .enter()
      .append("line")
      .attr("class", "link")
      .attr("stroke", "#999")
      .attr("stroke-opacity", 0.6)
      .attr("stroke-width", (d: AtlasLink) => Math.sqrt(Math.abs(d.weight)) * 2);

    const linkLabels = g
      .append("g")
      .attr("class", "link-labels")
      .selectAll("text")
      .data(processedLinks.filter((d: AtlasLink) => Math.abs(d.weight) > 0.1))
      .enter()
      .append("text")
      .attr("class", "link-label")
      .attr("font-size", 10)
      .attr("fill", "#666")
      .text((d: AtlasLink) => d.weight.toFixed(2));

    const node = g
      .append("g")
      .attr("class", "nodes")
      .selectAll("g")
      .data(connectedNodes)
      .enter()
      .append("g")
      .attr("class", (d: AtlasNode) => {
        const parsed = parseNodeId(d.id);
        if (parsed) {
          return parsed.type === "lorsa" ? "node lorsa" : "node transcoder";
        }
        return "node";
      })
      .call(
        d3
          .drag()
          .on("start", (event: any, d: AtlasNode) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x ?? null;
            d.fy = d.y ?? null;
          })
          .on("drag", (event: any, d: AtlasNode) => {
            const transform = d3.zoomTransform(svgRef.current as SVGSVGElement);
            d.fx = (event.x - transform.x) / transform.k;
            d.fy = (event.y - transform.y) / transform.k;
          })
          .on("end", (event: any, d: AtlasNode) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          }),
      )
      .on("click", (_event: any, d: AtlasNode) => {
        showNodeIframe(d.id);
      });

    node
      .append("circle")
      .attr("r", 8)
      .attr("stroke", "#fff")
      .attr("stroke-width", 2)
      .attr("fill", (d: AtlasNode) => {
        const parsed = parseNodeId(d.id);
        if (!parsed) return "#999";
        return parsed.type === "lorsa" ? "#4a90e2" : "#ff8c00";
      });

    const nodeLabels = node
      .append("text")
      .attr("class", "node-label")
      .attr("font-size", 12)
      .attr("dx", 12)
      .attr("dy", -4);

    nodeLabels.each((d: AtlasNode, i: number, nodes: any[]) => {
      const text = d3.select(nodes[i]);
      const autointerp = d.autointerp ?? "";
      if (autointerp) {
        text
          .append("tspan")
          .attr("x", 12)
          .attr("dy", "0em")
          .text(autointerp);
        text
          .append("tspan")
          .attr("x", 12)
          .attr("dy", "1.2em")
          .text(d.id);
      } else {
        text.text(d.id);
      }
    });

    simulation.on("tick", () => {
      link
        .attr("x1", (d: AtlasLink) => d.source.x ?? 0)
        .attr("y1", (d: AtlasLink) => d.source.y ?? 0)
        .attr("x2", (d: AtlasLink) => d.target.x ?? 0)
        .attr("y2", (d: AtlasLink) => d.target.y ?? 0);

      linkLabels
        .attr(
          "x",
          (d: AtlasLink) => ((d.source.x ?? 0) + (d.target.x ?? 0)) / 2,
        )
        .attr(
          "y",
          (d: AtlasLink) => ((d.source.y ?? 0) + (d.target.y ?? 0)) / 2,
        );

      node.attr(
        "transform",
        (d: AtlasNode) => `translate(${d.x ?? 0},${d.y ?? 0})`,
      );
    });
  }, [graphData, strengthThreshold, showNodeIframe]);

  useEffect(() => {
    loadAtlasFile(selectedAtlas);
  }, [selectedAtlas, loadAtlasFile]);

  useEffect(() => {
    if (graphData) {
      visualizeGraph();
    }
  }, [graphData, visualizeGraph]);

  useEffect(() => {
    return () => {
      if (simulationRef.current) {
        simulationRef.current.stop();
      }
    };
  }, []);

  const handleSliderChange = (
    event: React.ChangeEvent<HTMLInputElement>,
  ): void => {
    const value = Number.parseFloat(event.target.value);
    setSliderValue(value);
  };

  const applyFilter = (): void => {
    setStrengthThreshold(sliderValue);
  };

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ): Promise<void> => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    setStatus(`Loading local file: ${file.name}...`);
    try {
      const text = await file.text();
      const data = JSON.parse(text) as AtlasGraphData;
      if (!data || !Array.isArray(data.nodes) || !Array.isArray(data.links)) {
        setStatus("Invalid graph data format in local file");
        return;
      }
      setGraphData(data);
      allLinksRef.current = [];
      setSliderValue(0);
      setStrengthThreshold(0);
      setMaxEdgeWeight(0);
      setIframeUrl(null);
      setIframeTitle(null);
      setStatus(`Loaded local file: ${file.name}`);
      // 清空 input，方便重复选择同一个文件时也能触发 change
      event.target.value = "";
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setStatus(`Error loading local file: ${message}`);
    }
  };

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-xl font-semibold">Atlas Graph Visualization</h2>
        <p className="text-sm text-muted-foreground">
          Explore atlas graphs of features and inspect individual feature
          details.
        </p>
      </div>

      <div className="space-y-3">
        <div className="flex flex-wrap items-center gap-3">
          <select
            className="border rounded px-3 py-2 text-sm bg-background"
            value={selectedAtlas}
            onChange={(event) => setSelectedAtlas(event.target.value)}
          >
            <option value="circuits/atlases/[Thr]Opp. One-move mate threaten/atlas_0.json">
              Opp. One-move threaten 0
            </option>
            <option value="circuits/atlases/[Thr]Opp. One-move mate threaten/atlas_1.json">
              Opp. One-move threaten 1
            </option>
          </select>
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">
              or upload local JSON:
            </span>
            <input
              type="file"
              accept=".json,application/json"
              onChange={handleFileUpload}
              className="text-xs"
            />
          </div>
          <span className="text-xs text-muted-foreground flex-1 min-w-0 truncate">
            {status}
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <label htmlFor="strengthSlider" className="text-sm">
            Min Edge Strength:
          </label>
          <input
            id="strengthSlider"
            type="range"
            min={0}
            max={maxEdgeWeight || 100}
            value={sliderValue}
            step={0.1}
            onChange={handleSliderChange}
            className="w-[300px]"
          />
          <span className="text-sm tabular-nums text-muted-foreground min-w-[60px]">
            {sliderValue.toFixed(1)}
          </span>
          <button
            type="button"
            onClick={applyFilter}
            className="px-3 py-2 text-sm rounded bg-primary text-primary-foreground hover:opacity-90"
          >
            Apply Filter
          </button>
        </div>
      </div>

      <div className="flex gap-4 items-start">
        <div className="flex-1 min-w-0">
          <div className="border rounded bg-background overflow-hidden">
            <svg
              ref={svgRef}
              className="w-full"
              width={2000}
              height={1600}
            />
          </div>
        </div>

        <div className="w-[420px] min-w-[420px] sticky top-6">
          <div
            className={`border rounded bg-background p-3 ${
              iframeUrl ? "block" : "hidden"
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-base font-semibold">
                {iframeTitle ?? "Feature Details"}
              </h3>
              <button
                type="button"
                onClick={() => setIframeUrl(null)}
                className="text-xs text-muted-foreground hover:underline"
              >
                Close
              </button>
            </div>
            <iframe
              title="feature-details"
              src={iframeUrl ?? ""}
              className="w-full border-0 rounded"
              style={{ height: "calc(100vh - 220px)", minHeight: 800 }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};