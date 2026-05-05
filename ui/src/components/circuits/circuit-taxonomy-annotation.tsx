import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Feature } from "@/types/feature";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { AppPagination } from "@/components/ui/pagination";
import { ChessBoard } from "@/components/chess/chess-board";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { LinkGraphContainer } from "./link-graph-container";
import { transformCircuitData } from "./link-graph/utils";
import {
  annotateCircuitTaxonomyFeature,
  CircuitTaxonomyCircuitDetail,
  CircuitTaxonomyCircuitSummary,
  CircuitTaxonomyDirectoryOption,
  CircuitTaxonomyFeatureRef,
  fetchCircuitTaxonomyCircuit,
  fetchCircuitTaxonomyCircuits,
  fetchCircuitTaxonomyDirectories,
  fetchFeatureByDictionaryName,
} from "@/utils/api";
import { normalizeZPattern } from "@/utils/activationUtils";
import { extractFenFromText, validateFen } from "@/utils/fenUtils";

const TAXONOMY_PREFIX_RE = /^\[(Det|Src|Tgt|Val|Cap|Pro|Mov|Tac|Spa|Uninterpretable)\]\s*/;
const TAXONOMY_TOP_ACTIVATION_PAGE_SIZE = 6;

const getFeatureCacheKey = (featureRef: CircuitTaxonomyFeatureRef) =>
  `${featureRef.dictionary_name}:${featureRef.feature_index}`;

const extractTaxonomyPrefix = (text?: string | null) => {
  if (!text) {
    return "";
  }
  const match = text.match(TAXONOMY_PREFIX_RE);
  return match?.[1] ?? "";
};

type ChessTopActivationSample = {
  fen: string;
  activationStrength: number;
  activations: number[] | undefined;
  zPatternIndices?: number[][];
  zPatternValues?: number[];
  sampleIndex: number;
};

const extractChessTopActivationSamples = (
  sampleGroup: Feature["sampleGroups"][0] | null | undefined,
): ChessTopActivationSample[] => {
  if (!sampleGroup) {
    return [];
  }

  const chessSamples: ChessTopActivationSample[] = [];

  sampleGroup.samples.forEach((sample, sampleIndex) => {
      const fen = extractFenFromText(sample.text ?? "");
      if (!fen || !validateFen(fen)) {
        return;
      }

      let activations: number[] | undefined;
      let activationStrength = 0;

      if (Array.isArray(sample.featureActsIndices) && Array.isArray(sample.featureActsValues)) {
        activations = new Array(64).fill(0);
        for (
          let idx = 0;
          idx < Math.min(sample.featureActsIndices.length, sample.featureActsValues.length);
          idx += 1
        ) {
          const boardIndex = sample.featureActsIndices[idx];
          const value = sample.featureActsValues[idx];

          if (boardIndex >= 0 && boardIndex < 64) {
            activations[boardIndex] = value;
            if (Math.abs(value) > Math.abs(activationStrength)) {
              activationStrength = value;
            }
          }
        }
      }

      chessSamples.push({
        fen,
        activationStrength,
        activations,
        ...normalizeZPattern(sample.zPatternIndices, sample.zPatternValues),
        sampleIndex,
      });
    });

  return chessSamples.sort((a, b) => Math.abs(b.activationStrength) - Math.abs(a.activationStrength));
};

const CircuitTaxonomyTopActivationBoards = ({
  sampleGroup,
}: {
  sampleGroup: Feature["sampleGroups"][0];
}) => {
  const [page, setPage] = useState(1);

  const chessSamples = useMemo(() => extractChessTopActivationSamples(sampleGroup), [sampleGroup]);
  const maxPage = Math.max(1, Math.ceil(chessSamples.length / TAXONOMY_TOP_ACTIVATION_PAGE_SIZE));
  const currentSamples = useMemo(
    () =>
      chessSamples.slice(
        (page - 1) * TAXONOMY_TOP_ACTIVATION_PAGE_SIZE,
        page * TAXONOMY_TOP_ACTIVATION_PAGE_SIZE,
      ),
    [chessSamples, page],
  );

  useEffect(() => {
    setPage(1);
  }, [sampleGroup]);

  if (chessSamples.length === 0) {
    return (
      <div className="text-sm text-muted-foreground">
        No activation samples containing chessboard were found for this feature.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        Showing {chessSamples.length} top activation samples as chess boards.
      </p>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        {currentSamples.map((sample, index) => (
          <div
            key={`${sample.sampleIndex}-${sample.fen}`}
            className="rounded-lg border bg-background p-3"
          >
            <div className="mb-2 flex items-center justify-between">
              <span className="text-xs text-muted-foreground">
                Sample #{(page - 1) * TAXONOMY_TOP_ACTIVATION_PAGE_SIZE + index + 1}
              </span>
              <span className="text-xs text-muted-foreground">
                Max act: {sample.activationStrength.toFixed(3)}
              </span>
            </div>

            <ChessBoard
              fen={sample.fen}
              size="small"
              showCoordinates
              activations={sample.activations}
              zPatternIndices={sample.zPatternIndices}
              zPatternValues={sample.zPatternValues}
              sampleIndex={sample.sampleIndex}
              analysisName="Taxonomy Top Activation"
              flip_activation={sample.fen.includes(" b ")}
              autoFlipWhenBlack
            />
          </div>
        ))}
      </div>

      {maxPage > 1 && <AppPagination page={page} setPage={setPage} maxPage={maxPage} />}
    </div>
  );
};

export const CircuitTaxonomyAnnotation = () => {
  const [directories, setDirectories] = useState<CircuitTaxonomyDirectoryOption[]>([]);
  const [taxonomyLabels, setTaxonomyLabels] = useState<string[]>([]);
  const [selectedDirectoryId, setSelectedDirectoryId] = useState<string>("");
  const [circuits, setCircuits] = useState<CircuitTaxonomyCircuitSummary[]>([]);
  const [selectedCircuitFile, setSelectedCircuitFile] = useState<string>("");
  const [circuitDetail, setCircuitDetail] = useState<CircuitTaxonomyCircuitDetail | null>(null);
  const [currentFeatureIndex, setCurrentFeatureIndex] = useState(0);
  const [selectedTaxonomy, setSelectedTaxonomy] = useState("");
  const [featureCache, setFeatureCache] = useState<Record<string, Feature>>({});
  const [clickedId, setClickedId] = useState<string | null>(null);
  const [loadingDirectories, setLoadingDirectories] = useState(false);
  const [loadingCircuits, setLoadingCircuits] = useState(false);
  const [loadingCircuitDetail, setLoadingCircuitDetail] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const pendingFeatureLoads = useRef<Map<string, Promise<Feature | null>>>(new Map());

  useEffect(() => {
    let cancelled = false;
    const loadDirectories = async () => {
      setLoadingDirectories(true);
      setError(null);
      try {
        const response = await fetchCircuitTaxonomyDirectories();
        if (cancelled) {
          return;
        }
        setDirectories(response.directories);
        setTaxonomyLabels(response.taxonomy_labels);
        if (!selectedDirectoryId && response.directories.length > 0) {
          setSelectedDirectoryId(response.directories[0].id);
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : "Failed to load circuit directories");
        }
      } finally {
        if (!cancelled) {
          setLoadingDirectories(false);
        }
      }
    };

    loadDirectories();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedDirectoryId) {
      return;
    }

    let cancelled = false;
    const loadCircuits = async () => {
      setLoadingCircuits(true);
      setError(null);
      setCircuitDetail(null);
      setFeatureCache({});
      setSelectedCircuitFile("");
      setCurrentFeatureIndex(0);
      try {
        const response = await fetchCircuitTaxonomyCircuits(selectedDirectoryId);
        if (cancelled) {
          return;
        }
        setCircuits(response.circuits);
        if (response.circuits.length > 0) {
          setSelectedCircuitFile(response.circuits[0].file_name);
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : "Failed to load circuit list");
          setCircuits([]);
        }
      } finally {
        if (!cancelled) {
          setLoadingCircuits(false);
        }
      }
    };

    loadCircuits();
    return () => {
      cancelled = true;
    };
  }, [selectedDirectoryId]);

  useEffect(() => {
    if (!selectedDirectoryId || !selectedCircuitFile) {
      return;
    }

    let cancelled = false;
    const loadCircuitDetail = async () => {
      setLoadingCircuitDetail(true);
      setError(null);
      setCircuitDetail(null);
      setFeatureCache({});
      setCurrentFeatureIndex(0);
      try {
        const detail = await fetchCircuitTaxonomyCircuit(selectedDirectoryId, selectedCircuitFile);
        if (cancelled) {
          return;
        }
        setCircuitDetail(detail);
        setClickedId(detail.features[0]?.node_id ?? null);
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : "Failed to load circuit detail");
        }
      } finally {
        if (!cancelled) {
          setLoadingCircuitDetail(false);
        }
      }
    };

    loadCircuitDetail();
    return () => {
      cancelled = true;
    };
  }, [selectedDirectoryId, selectedCircuitFile]);

  const currentFeatureRef = useMemo(
    () => circuitDetail?.features[currentFeatureIndex] ?? null,
    [circuitDetail, currentFeatureIndex],
  );

  const ensureFeatureLoaded = useCallback(async (featureRef: CircuitTaxonomyFeatureRef | null) => {
    if (!featureRef) {
      return null;
    }
    const cacheKey = getFeatureCacheKey(featureRef);
    if (featureCache[cacheKey]) {
      return featureCache[cacheKey];
    }
    if (pendingFeatureLoads.current.has(cacheKey)) {
      return pendingFeatureLoads.current.get(cacheKey) ?? null;
    }

    const request = fetchFeatureByDictionaryName(featureRef.dictionary_name, featureRef.feature_index)
      .then((feature) => {
        if (feature) {
          setFeatureCache((prev) => ({
            ...prev,
            [cacheKey]: feature,
          }));
        }
        return feature;
      })
      .finally(() => {
        pendingFeatureLoads.current.delete(cacheKey);
      });

    pendingFeatureLoads.current.set(cacheKey, request);
    return request;
  }, [featureCache]);

  const refreshFeature = useCallback(async (featureRef: CircuitTaxonomyFeatureRef | null) => {
    if (!featureRef) {
      return null;
    }
    const refreshed = await fetchFeatureByDictionaryName(featureRef.dictionary_name, featureRef.feature_index);
    if (refreshed) {
      setFeatureCache((prev) => ({
        ...prev,
        [getFeatureCacheKey(featureRef)]: refreshed,
      }));
    }
    return refreshed;
  }, []);

  useEffect(() => {
    const prefetch = async () => {
      if (!circuitDetail) {
        return;
      }
      for (let offset = 0; offset < 3; offset += 1) {
        const featureRef = circuitDetail.features[currentFeatureIndex + offset];
        if (!featureRef) {
          continue;
        }
        void ensureFeatureLoaded(featureRef);
      }
    };
    void prefetch();
  }, [circuitDetail, currentFeatureIndex, ensureFeatureLoaded]);

  const currentFeature = currentFeatureRef ? featureCache[getFeatureCacheKey(currentFeatureRef)] ?? null : null;

  useEffect(() => {
    setSelectedTaxonomy(extractTaxonomyPrefix(currentFeature?.interpretation?.text));
  }, [currentFeature?.dictionaryName, currentFeature?.featureIndex, currentFeature?.interpretation?.text]);

  useEffect(() => {
    setClickedId(currentFeatureRef?.node_id ?? null);
  }, [currentFeatureRef?.node_id]);

  const graphData = useMemo(() => {
    if (!circuitDetail) {
      return null;
    }
    return transformCircuitData(circuitDetail.graph_data as any);
  }, [circuitDetail]);

  const topActivationGroup = useMemo(() => {
    if (!currentFeature) {
      return null;
    }
    return (
      currentFeature.sampleGroups.find((group) => group.analysisName === "top_activations") ??
      currentFeature.sampleGroups[0] ??
      null
    );
  }, [currentFeature]);

  const goToFeatureIndex = useCallback((nextIndex: number) => {
    if (!circuitDetail) {
      return;
    }
    const clamped = Math.min(Math.max(nextIndex, 0), Math.max(circuitDetail.total_features - 1, 0));
    setCurrentFeatureIndex(clamped);
  }, [circuitDetail]);

  const goToNextFeature = useCallback(() => {
    if (!circuitDetail) {
      return;
    }
    if (currentFeatureIndex < circuitDetail.total_features - 1) {
      setCurrentFeatureIndex((prev) => prev + 1);
      return;
    }
    const nextCircuit = circuits[circuitDetail.circuit_index + 1];
    if (nextCircuit) {
      setSelectedCircuitFile(nextCircuit.file_name);
    }
  }, [circuitDetail, circuits, currentFeatureIndex]);

  const handleGraphFeatureSelect = useCallback((feature: Feature | null) => {
    if (!feature || !circuitDetail) {
      return;
    }

    const matchedIndex = circuitDetail.features.findIndex(
      (item) =>
        item.dictionary_name === feature.dictionaryName && item.feature_index === feature.featureIndex,
    );
    if (matchedIndex >= 0) {
      setFeatureCache((prev) => ({
        ...prev,
        [`${feature.dictionaryName}:${feature.featureIndex}`]: feature,
      }));
      setCurrentFeatureIndex(matchedIndex);
    }
  }, [circuitDetail]);

  const handleSaveAndNext = useCallback(async () => {
    if (!currentFeatureRef) {
      return;
    }
    if (!selectedTaxonomy) {
      window.alert("Please select a taxonomy label first.");
      return;
    }

    setSaving(true);
    setError(null);
    try {
      let response = await annotateCircuitTaxonomyFeature(
        currentFeatureRef.dictionary_name,
        currentFeatureRef.feature_index,
        selectedTaxonomy,
      );

      if (response.status === "conflict") {
        const confirmed = window.confirm(
          `Current interpretation starts with [${response.existing_taxonomy}]. Replace it with [${selectedTaxonomy}]?`,
        );
        if (!confirmed) {
          return;
        }
        response = await annotateCircuitTaxonomyFeature(
          currentFeatureRef.dictionary_name,
          currentFeatureRef.feature_index,
          selectedTaxonomy,
          true,
        );
      }

      if (response.status === "updated") {
        await refreshFeature(currentFeatureRef);
      }

      goToNextFeature();
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Failed to save taxonomy label");
    } finally {
      setSaving(false);
    }
  }, [currentFeatureRef, goToNextFeature, refreshFeature, selectedTaxonomy]);

  const directoryLabel = directories.find((item) => item.id === selectedDirectoryId)?.label ?? selectedDirectoryId;
  const featureProgressValue = circuitDetail ? currentFeatureIndex + 1 : 0;
  const featureProgressMax = circuitDetail?.total_features ?? 1;
  const circuitProgressValue = circuitDetail ? circuitDetail.circuit_index + 1 : 0;
  const circuitProgressMax = circuitDetail?.total_circuits ?? 1;

  return (
    <div className="flex flex-col gap-6">
      <Card>
        <CardHeader>
          <CardTitle>Taxonomy Annotation</CardTitle>
          <CardDescription>
            Load a saved circuit, browse its feature list in layer order, and write taxonomy labels back to MongoDB interpretations.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-2">
          <div className="flex flex-col gap-2">
            <span className="text-sm font-medium">Circuit Directory</span>
            <Select value={selectedDirectoryId} onValueChange={setSelectedDirectoryId} disabled={loadingDirectories}>
              <SelectTrigger>
                <SelectValue placeholder="Select a circuit directory" />
              </SelectTrigger>
              <SelectContent>
                {directories.map((directory) => (
                  <SelectItem key={directory.id} value={directory.id}>
                    {directory.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex flex-col gap-2">
            <span className="text-sm font-medium">Circuit File</span>
            <Select value={selectedCircuitFile} onValueChange={setSelectedCircuitFile} disabled={loadingCircuits || circuits.length === 0}>
              <SelectTrigger>
                <SelectValue placeholder="Select a circuit file" />
              </SelectTrigger>
              <SelectContent>
                {circuits.map((circuit) => (
                  <SelectItem key={circuit.file_name} value={circuit.file_name}>
                    {`${circuit.index + 1}. ${circuit.file_name}`}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {error && (
        <Card className="border-red-300">
          <CardContent className="pt-6 text-sm text-red-600">{error}</CardContent>
        </Card>
      )}

      {circuitDetail && (
        <Card>
          <CardContent className="grid gap-4 pt-6 md:grid-cols-2">
            <div className="flex flex-col gap-2">
              <div className="text-sm font-medium">{directoryLabel}</div>
              <div className="text-sm text-muted-foreground break-all">{circuitDetail.file_name}</div>
              <div className="text-sm text-muted-foreground">
                Prompt: {String(circuitDetail.metadata.prompt ?? "")}
              </div>
              <div className="text-sm text-muted-foreground">
                Target Move: {String(circuitDetail.metadata.target_move ?? "-")}
              </div>
            </div>

            <div className="grid gap-4">
              <div className="grid gap-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium">Circuit Progress</span>
                  <span>{circuitProgressValue} / {circuitProgressMax}</span>
                </div>
                <Progress value={circuitProgressValue} max={circuitProgressMax} />
              </div>

              <div className="grid gap-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium">Feature Progress</span>
                  <span>{featureProgressValue} / {featureProgressMax}</span>
                </div>
                <Progress value={featureProgressValue} max={featureProgressMax} />
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid gap-6 xl:grid-cols-[1.3fr_1fr]">
        <Card className="min-h-[720px]">
          <CardHeader>
            <CardTitle>Circuit Graph</CardTitle>
            <CardDescription>
              Click a node to jump to that feature. Embedding and logit nodes stay hidden in this annotation view.
            </CardDescription>
          </CardHeader>
          <CardContent className="h-[720px]">
            {graphData ? (
              <LinkGraphContainer
                data={graphData}
                clickedId={clickedId}
                onNodeClick={(node) => setClickedId(node.nodeId || null)}
                onFeatureSelect={handleGraphFeatureSelect}
                hideEmbLogit
              />
            ) : (
              <div className="text-sm text-muted-foreground">
                {loadingCircuitDetail ? "Loading circuit graph..." : "Select a circuit to start."}
              </div>
            )}
          </CardContent>
        </Card>

        <div className="flex flex-col gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Current Feature</CardTitle>
              <CardDescription>
                {currentFeatureRef
                  ? `${currentFeatureRef.label} | dictionary ${currentFeatureRef.dictionary_name}`
                  : "No feature selected"}
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              {currentFeatureRef && (
                <>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      onClick={() => goToFeatureIndex(currentFeatureIndex - 1)}
                      disabled={currentFeatureIndex <= 0}
                    >
                      Previous
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => goToNextFeature()}
                      disabled={!circuitDetail}
                    >
                      Skip
                    </Button>
                  </div>

                  <div className="grid gap-2">
                    <div className="text-sm font-medium">Taxonomy Label</div>
                    <div className="grid grid-cols-2 gap-2">
                      {taxonomyLabels.map((label) => (
                        <Button
                          key={label}
                          type="button"
                          variant={selectedTaxonomy === label ? "default" : "outline"}
                          onClick={() => setSelectedTaxonomy(label)}
                        >
                          {label}
                        </Button>
                      ))}
                    </div>
                  </div>

                  <div className="grid gap-2 text-sm">
                    <div>
                      <span className="font-medium">Current Prefix:</span>{" "}
                      {extractTaxonomyPrefix(currentFeature?.interpretation?.text) || "None"}
                    </div>
                    <div className="whitespace-pre-wrap rounded-md border bg-slate-50 p-3">
                      {currentFeature
                        ? (currentFeature.interpretation?.text || "No interpretation available.")
                        : "Loading feature interpretation and top activation samples..."}
                    </div>
                  </div>

                  <Button onClick={handleSaveAndNext} disabled={saving || !currentFeatureRef}>
                    {saving ? "Saving..." : "Confirm And Next"}
                  </Button>
                </>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Top Activation Samples</CardTitle>
              <CardDescription>
                The current feature is loaded first, and the next few features are prefetched in the background for faster annotation.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {currentFeature && topActivationGroup ? (
                <CircuitTaxonomyTopActivationBoards sampleGroup={topActivationGroup} />
              ) : (
                <div className="text-sm text-muted-foreground">
                  {currentFeatureRef ? "Loading top activation samples..." : "Select a feature to inspect samples."}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};
