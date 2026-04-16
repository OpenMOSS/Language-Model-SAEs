import React, { useCallback, useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { Link } from "react-router-dom";
import { ChessBoard } from "@/components/chess/chess-board";
import { FeatureSchema } from "@/types/feature";

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

interface TopActivationSample {
  fen: string;
  activationStrength: number;
  activations?: number[];
  zPatternIndices?: number[];
  zPatternValues?: number[];
  contextId?: number;
  sampleIndex?: number;
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
  // This helper is no longer used to build an iframe URL.
  // Kept for backward compatibility; actual feature details are fetched
  // via the dictionaries backend API instead of embedding a separate page.
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

  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [topActivations, setTopActivations] = useState<TopActivationSample[]>([]);
  const [loadingTopActivations, setLoadingTopActivations] = useState<boolean>(false);
  const [topActivationError, setTopActivationError] = useState<string | null>(null);
  const [nodeInterpretations, setNodeInterpretations] = useState<Record<string, string>>({});
  const [refreshingInterpretations, setRefreshingInterpretations] = useState<boolean>(false);

  const buildDictionaryName = (layerIdx: number, isLorsa: boolean): string => {
    const LORSA_ANALYSIS_NAME = "BT4_lorsa_k30_e16";
    const TC_ANALYSIS_NAME = "BT4_tc_k30_e16";

    if (isLorsa) {
      const suffix = LORSA_ANALYSIS_NAME.replace("BT4_lorsa", "").replace(/^_/, "");
      return suffix ? `BT4_lorsa_L${layerIdx}A_${suffix}` : `BT4_lorsa_L${layerIdx}A`;
    }

    const suffix = TC_ANALYSIS_NAME.replace("BT4_tc", "").replace(/^_/, "");
    return suffix ? `BT4_tc_L${layerIdx}M_${suffix}` : `BT4_tc_L${layerIdx}M`;
  };

  const fetchTopActivations = useCallback(
    async (nodeId: string): Promise<void> => {
      const parsed = parseNodeId(nodeId);
      if (!parsed) {
        // eslint-disable-next-line no-console
        console.warn("Invalid node ID format:", nodeId);
        setTopActivations([]);
        return;
      }

      const backendBase = import.meta.env.VITE_BACKEND_URL ?? "";
      if (!backendBase) {
        // eslint-disable-next-line no-console
        console.warn("VITE_BACKEND_URL is not set; cannot fetch top activations.");
        setTopActivations([]);
        return;
      }

      const layerIdx = parsed.layer;
      const featureIndex = parsed.feature;
      const isLorsa = parsed.type === "lorsa";
      const dictionary = buildDictionaryName(layerIdx, isLorsa);

      setLoadingTopActivations(true);
      setTopActivationError(null);
      try {
        const response = await fetch(
          `${backendBase}/dictionaries/${encodeURIComponent(
            dictionary,
          )}/features/${featureIndex}`,
          {
            method: "GET",
            headers: {
              Accept: "application/x-msgpack",
            },
          },
        );

        if (!response.ok) {
          const text = await response.text();
          throw new Error(`HTTP ${response.status}: ${text}`);
        }

        const arrayBuffer = await response.arrayBuffer();
        const msgpackModule = await import("@msgpack/msgpack");
        const decoded = msgpackModule.decode(new Uint8Array(arrayBuffer)) as any;

        const camelcaseKeysModule = await import("camelcase-keys");
        const camelcaseKeys = camelcaseKeysModule.default;

        const camelData = camelcaseKeys(decoded as Record<string, unknown>, {
          deep: true,
          stopPaths: ["sample_groups.samples.context"],
        }) as any;

        const feature = FeatureSchema.parse(camelData);
        const interpretationText: string =
          typeof feature.interpretation?.text === "string"
            ? feature.interpretation.text
            : "";

        const sampleGroups = feature.sampleGroups || [];
        const allSamples: any[] = [];

        for (const group of sampleGroups) {
          if (group.samples && Array.isArray(group.samples)) {
            allSamples.push(...group.samples);
          }
        }

        const chessSamples: TopActivationSample[] = [];

        for (const sample of allSamples) {
          if (!sample.text) continue;
          const lines = String(sample.text).split("\n");

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.includes("/")) continue;

            const parts = trimmed.split(/\s+/);
            if (parts.length < 6) continue;

            const [boardPart, activeColor] = parts;
            const boardRows = boardPart.split("/");
            if (boardRows.length !== 8 || !/^[wb]$/.test(activeColor)) {
              continue;
            }

            let isValidBoard = true;
            let totalSquares = 0;

            for (const row of boardRows) {
              if (!/^[rnbqkpRNBQKP1-8]+$/.test(row)) {
                isValidBoard = false;
                break;
              }
              let rowSquares = 0;
              for (const ch of row) {
                if (/\d/.test(ch)) {
                  rowSquares += Number.parseInt(ch, 10);
                } else {
                  rowSquares += 1;
                }
              }
              totalSquares += rowSquares;
            }

            if (!isValidBoard || totalSquares !== 64) {
              continue;
            }

            let activationsArray: number[] | undefined;
            let maxActivation = 0;

            if (
              Array.isArray(sample.featureActsIndices) &&
              Array.isArray(sample.featureActsValues)
            ) {
              activationsArray = new Array(64).fill(0);
              for (
                let i = 0;
                i < Math.min(sample.featureActsIndices.length, sample.featureActsValues.length);
                i += 1
              ) {
                const idx = sample.featureActsIndices[i];
                const value = sample.featureActsValues[i];
                if (idx >= 0 && idx < 64) {
                  activationsArray[idx] = value;
                  if (Math.abs(value) > Math.abs(maxActivation)) {
                    maxActivation = value;
                  }
                }
              }
            }

            chessSamples.push({
              fen: trimmed,
              activationStrength: maxActivation,
              activations: activationsArray,
              zPatternIndices: sample.zPatternIndices,
              zPatternValues: sample.zPatternValues,
              contextId: sample.contextIdx ?? sample.context_idx,
              sampleIndex: sample.sampleIndex ?? 0,
            });

            break;
          }
        }

        const topSamples = chessSamples
          .sort((a, b) => Math.abs(b.activationStrength) - Math.abs(a.activationStrength))
          .slice(0, 4);

        setTopActivations(topSamples);
        if (interpretationText) {
          setNodeInterpretations((prev) => ({
            ...prev,
            [nodeId]: interpretationText,
          }));
        }
      } catch (error) {
        // eslint-disable-next-line no-console
        console.error("Failed to fetch top activation data for atlas node:", error);
        setTopActivations([]);
        setTopActivationError(
          error instanceof Error ? error.message : "Failed to fetch top activations",
        );
      } finally {
        setLoadingTopActivations(false);
      }
    },
    [],
  );

  const showNodeDetails = useCallback(
    (nodeId: string): void => {
      const parsed = parseNodeId(nodeId);
      if (!parsed) {
        // eslint-disable-next-line no-console
        console.warn("Invalid node ID format:", nodeId);
        return;
      }
      setSelectedNodeId(nodeId);
      setTopActivations([]);
      setTopActivationError(null);
      fetchTopActivations(nodeId);
    },
    [fetchTopActivations],
  );

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
      setSelectedNodeId(null);
      setTopActivations([]);
      setTopActivationError(null);
      setNodeInterpretations({});
      setStatus("Graph loaded successfully!");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setStatus(`Error: ${message}`);
    }
  }, []);

  const refreshAllInterpretations = useCallback(async (): Promise<void> => {
    if (!graphData) return;

    const backendBase = import.meta.env.VITE_BACKEND_URL ?? "";
    if (!backendBase) {
      // eslint-disable-next-line no-console
      console.warn("VITE_BACKEND_URL is not set; cannot refresh interpretations.");
      return;
    }

    setRefreshingInterpretations(true);
    try {
      const updates: Record<string, string> = {};

      for (const rawNode of graphData.nodes) {
        let nodeId: string;
        if ("node_id" in rawNode) {
          nodeId = rawNode.node_id;
        } else {
          const key = Object.keys(rawNode)[0];
          nodeId = key;
        }

        const parsed = parseNodeId(nodeId);
        if (!parsed) {
          continue;
        }

        const layerIdx = parsed.layer;
        const featureIndex = parsed.feature;
        const isLorsa = parsed.type === "lorsa";
        const dictionary = buildDictionaryName(layerIdx, isLorsa);

        try {
          const response = await fetch(
            `${backendBase}/dictionaries/${encodeURIComponent(
              dictionary,
            )}/features/${featureIndex}`,
            {
              method: "GET",
              headers: {
                Accept: "application/x-msgpack",
              },
            },
          );

          if (!response.ok) {
            continue;
          }

          const arrayBuffer = await response.arrayBuffer();
          const msgpackModule = await import("@msgpack/msgpack");
          const decoded = msgpackModule.decode(new Uint8Array(arrayBuffer)) as any;

          const camelcaseKeysModule = await import("camelcase-keys");
          const camelcaseKeys = camelcaseKeysModule.default;

          const camelData = camelcaseKeys(decoded as Record<string, unknown>, {
            deep: true,
            stopPaths: ["sample_groups.samples.context"],
          }) as any;

          const feature = FeatureSchema.parse(camelData);
          const interpretationText: string =
            typeof feature.interpretation?.text === "string"
              ? feature.interpretation.text
              : "";

          if (interpretationText && interpretationText.trim().length > 0) {
            updates[nodeId] = interpretationText;
          }
        } catch {
          // ignore errors for individual nodes
        }
      }

      if (Object.keys(updates).length > 0) {
        setNodeInterpretations((prev) => ({
          ...prev,
          ...updates,
        }));
      }
    } finally {
      setRefreshingInterpretations(false);
    }
  }, [graphData]);

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
        const maybeInterp = (node as RawAtlasNodeOld)[key];
        if (typeof maybeInterp === "string") {
          autointerp = maybeInterp;
        }
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

    const highlightedNodeIds = new Set<string>();
    if (selectedNodeId) {
      highlightedNodeIds.add(selectedNodeId);
      processedLinks.forEach((link) => {
        if (link.source.id === selectedNodeId) {
          highlightedNodeIds.add(link.target.id);
        }
        if (link.target.id === selectedNodeId) {
          highlightedNodeIds.add(link.source.id);
        }
      });
    }

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
      .attr("stroke-opacity", (d: AtlasLink) => {
        const absW = Math.abs(d.weight);
        const maxW = maxEdgeWeight || absW || 1;
        const norm = Math.min(1, absW / maxW);
        const baseOpacity = 0.1 + Math.sqrt(norm) * 0.7; // 0.1 ~ 0.8
        if (!selectedNodeId) {
          return baseOpacity;
        }
        const isIncident =
          d.source.id === selectedNodeId || d.target.id === selectedNodeId;
        return isIncident ? baseOpacity : 0.03;
      })
      .attr("stroke-width", (d: AtlasLink) => {
        const absW = Math.abs(d.weight);
        const maxW = maxEdgeWeight || absW || 1;
        const norm = Math.min(1, absW / maxW);
        const minWidth = 0.8;
        const maxWidth = 4.0;
        return minWidth + norm * (maxWidth - minWidth);
      });

    const linkLabels = g
      .append("g")
      .attr("class", "link-labels")
      .selectAll("text")
      .data(processedLinks.filter((d: AtlasLink) => Math.abs(d.weight) > 500))
      .enter()
      .append("text")
      .attr("class", "link-label")
      .attr("font-size", 10)
      .attr("fill", "#666")
      .attr("text-anchor", "middle")
      .attr("dy", -2)
      .style("pointer-events", "none")
      .style("opacity", (d: AtlasLink) => {
        if (!selectedNodeId) return 0.7;
        const isIncident =
          d.source.id === selectedNodeId || d.target.id === selectedNodeId;
        return isIncident ? 0.8 : 0.05;
      })
      .text((d: AtlasLink) => d.weight.toFixed(1));

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
        showNodeDetails(d.id);
      })
      .style("opacity", (d: AtlasNode) => {
        if (!selectedNodeId) {
          return 1;
        }
        return highlightedNodeIds.has(d.id) ? 1 : 0.1;
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
      const interpFromState = nodeInterpretations[d.id];
      const rawInterp = interpFromState ?? d.autointerp ?? "";
      const trimmedInterp =
        typeof rawInterp === "string" ? rawInterp.trim() : "";

      if (trimmedInterp.length > 0) {
        text
          .append("tspan")
          .attr("x", 12)
          .attr("dy", "0em")
          .text(trimmedInterp);

        text
          .append("tspan")
          .attr("x", 12)
          .attr("dy", "1.2em")
          .text(d.id);
      } else {
        text
          .append("tspan")
          .attr("x", 12)
          .attr("dy", "0em")
          .text(d.id);
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
  }, [graphData, strengthThreshold, showNodeDetails, nodeInterpretations, selectedNodeId, maxEdgeWeight]);

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
      setSelectedNodeId(null);
      setTopActivations([]);
      setTopActivationError(null);
      setNodeInterpretations({});
      setStatus(`Loaded local file: ${file.name}`);
      event.target.value = "";
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setStatus(`Error loading local file: ${message}`);
    }
  };

  return (
    <div className="space-y-4">

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
          <input
            type="number"
            min={0}
            max={maxEdgeWeight || 0}
            value={Number.isFinite(sliderValue) ? sliderValue : 0}
            step={0.1}
            onChange={(event) => {
              const raw = Number.parseFloat(event.target.value);
              const max = maxEdgeWeight || 0;
              if (!Number.isFinite(raw)) {
                setSliderValue(0);
                return;
              }
              const clamped = Math.max(0, Math.min(raw, max));
              setSliderValue(clamped);
            }}
            className="w-20 px-2 py-1 border rounded text-sm"
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
          <button
            type="button"
            onClick={refreshAllInterpretations}
            disabled={refreshingInterpretations || !graphData}
            className="px-3 py-2 text-sm rounded border border-muted text-muted-foreground hover:bg-muted disabled:opacity-50"
          >
            {refreshingInterpretations ? "Refreshing Interpretations..." : "Refresh Interpretations"}
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
              selectedNodeId ? "block" : "hidden"
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-base font-semibold">
                {selectedNodeId ? `Feature: ${selectedNodeId}` : "Feature Details"}
              </h3>
              <div className="flex items-center gap-2">
                {selectedNodeId &&
                  (() => {
                    const parsed = parseNodeId(selectedNodeId);
                    if (!parsed) {
                      return null;
                    }
                    const layerIdx = parsed.layer;
                    const featureIndex = parsed.feature;
                    const isLorsa = parsed.type === "lorsa";
                    const dictionary = buildDictionaryName(layerIdx, isLorsa);
                    const nodeTypeDisplay = isLorsa ? "LORSA" : "SAE";
                    return (
                      <Link
                        to={`/features?dictionary=${encodeURIComponent(
                          dictionary,
                        )}&featureIndex=${featureIndex}`}
                        className="inline-flex items-center px-2 py-1 text-xs bg-primary text-primary-foreground rounded hover:opacity-90"
                        title={`Go to L${layerIdx} ${nodeTypeDisplay} Feature #${featureIndex}`}
                      >
                        View in Feature Page
                      </Link>
                    );
                  })()}
                <button
                  type="button"
                  onClick={() => {
                    setSelectedNodeId(null);
                    setTopActivations([]);
                    setTopActivationError(null);
                  }}
                  className="text-xs text-muted-foreground hover:underline"
                >
                  Close
                </button>
              </div>
            </div>
            {loadingTopActivations ? (
              <div className="flex items-center justify-center py-4">
                <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary" />
                  <span>Loading top activation samples...</span>
                </div>
              </div>
            ) : topActivationError ? (
              <div className="text-sm text-red-600">
                Failed to load top activations: {topActivationError}
              </div>
            ) : topActivations.length === 0 ? (
              <div className="text-sm text-muted-foreground">
                No activation samples containing chessboard were found for this feature.
              </div>
            ) : (
              <div className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  Showing top {topActivations.length} activation samples (by max activation).
                </p>
                <div className="grid grid-cols-1 gap-3">
                  {topActivations.map((sample, index) => (
                    <div
                      key={`${sample.contextId ?? 0}-${sample.sampleIndex ?? index}`}
                      className="bg-background border rounded p-2"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs text-muted-foreground">
                          Sample #{index + 1} • Max act:{" "}
                          {sample.activationStrength.toFixed(3)}
                        </span>
                        {sample.contextId !== undefined && (
                          <span className="text-xs text-muted-foreground">
                            Context {sample.contextId}
                          </span>
                        )}
                      </div>
                      <ChessBoard
                        fen={sample.fen}
                        size="small"
                        showCoordinates
                        activations={sample.activations}
                        zPatternIndices={sample.zPatternIndices}
                        zPatternValues={sample.zPatternValues}
                        sampleIndex={sample.sampleIndex}
                        analysisName="Atlas Feature"
                        flip_activation={Boolean(
                          sample.fen && sample.fen.split(" ")[1] === "b",
                        )}
                        autoFlipWhenBlack
                      />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};