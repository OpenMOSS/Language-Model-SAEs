import { AppNavbar } from "@/components/app/navbar";
import { SectionNavigator } from "@/components/app/section-navigator";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { FeatureSchema } from "@/types/feature";
import { decode } from "@msgpack/msgpack";
import camelcaseKeys from "camelcase-keys";
import { useEffect, useState, useMemo, useCallback, Suspense, lazy } from "react";
import { useSearchParams } from "react-router-dom";
import { useAsyncFn, useMount, useDebounce } from "react-use";
import { z } from "zod";

const FeatureCard = lazy(() => import("@/components/feature/feature-card").then(module => ({ default: module.FeatureCard })));

export const FeaturesPage = () => {
  const [searchParams, setSearchParams] = useSearchParams();

  const [dictionariesState, fetchDictionaries] = useAsyncFn(async () => {
    return await fetch(`${import.meta.env.VITE_BACKEND_URL}/dictionaries`)
      .then(async (res) => await res.json())
      .then((res) => z.array(z.string()).parse(res));
  });

  const [selectedDictionary, setSelectedDictionary] = useState<string | null>(null);

  const [analysesState, fetchAnalyses] = useAsyncFn(async (dictionary: string) => {
    if (!dictionary) return [];

    return await fetch(`${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}/analyses`)
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(await res.text());
        }
        return res;
      })
      .then(async (res) => await res.json())
      .then((res) => z.array(z.string()).parse(res));
  });

  const [selectedAnalysis, setSelectedAnalysis] = useState<string | null>(null);

  // Metric filtering state
  const [metricsState, fetchMetrics] = useAsyncFn(async (dictionary: string) => {
    if (!dictionary) return [];

    return await fetch(`${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}/metrics`)
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(await res.text());
        }
        return res;
      })
      .then(async (res) => await res.json())
      .then((res) => z.object({ metrics: z.array(z.string()) }).parse(res).metrics);
  });

  const [metricFilters, setMetricFilters] = useState<Record<string, { min?: number; max?: number }>>({});
  const [featureCount, setFeatureCount] = useState<number | null>(null);

  const [featureIndex, setFeatureIndex] = useState<number>(0);
  const [inputValue, setInputValue] = useState<string>("0");
  const [loadingRandomFeature, setLoadingRandomFeature] = useState<boolean>(false);

  // Debounce the input value to avoid excessive updates
  useDebounce(
    () => {
      const parsed = parseInt(inputValue);
      if (!isNaN(parsed) && parsed !== featureIndex) {
        setFeatureIndex(parsed);
      }
    },
    300,
    [inputValue]
  );

  const handleFeatureIndexChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  }, []);

  // Function to count features matching filters
  const [countState, countFeatures] = useAsyncFn(
    async (
      dictionary: string | null,
      analysisName: string | null = null,
      metricFilters?: Record<string, { min?: number; max?: number }>
    ) => {
      if (!dictionary) {
        return 0;
      }

      // Build query parameters
      const params = new URLSearchParams();
      if (analysisName) {
        params.append("feature_analysis_name", analysisName);
      }
      
      // Add metric filters if provided
      if (metricFilters) {
        const mongoFilters: Record<string, Record<string, number>> = {};
        
        for (const [metricName, filter] of Object.entries(metricFilters)) {
          const mongoFilter: Record<string, number> = {};
          if (filter.min !== undefined) {
            mongoFilter["$gte"] = filter.min;
          }
          if (filter.max !== undefined) {
            mongoFilter["$lte"] = filter.max;
          }
          
          if (Object.keys(mongoFilter).length > 0) {
            mongoFilters[metricName] = mongoFilter;
          }
        }
        
        if (Object.keys(mongoFilters).length > 0) {
          params.append("metric_filters", JSON.stringify(mongoFilters));
        }
      }

      const queryString = params.toString();
      const url = `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}/features/count${queryString ? `?${queryString}` : ""}`;

      const response = await fetch(url, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const data = await response.json();
      const count = z.object({ count: z.number() }).parse(data).count;
      setFeatureCount(count);
      return count;
    }
  );

  const [featureState, fetchFeature] = useAsyncFn(
    async (
      dictionary: string | null,
      featureIndex: number | string = "random",
      analysisName: string | null = null,
      metricFilters?: Record<string, { min?: number; max?: number }>
    ) => {
      if (!dictionary) {
        alert("Please select a dictionary first");
        return;
      }

      setLoadingRandomFeature(featureIndex === "random");

      // Build query parameters
      const params = new URLSearchParams();
      if (analysisName) {
        params.append("feature_analysis_name", analysisName);
      }
      
      // Add metric filters if provided and we're fetching a random feature
      if (metricFilters && featureIndex === "random") {
        const mongoFilters: Record<string, Record<string, number>> = {};
        
        for (const [metricName, filter] of Object.entries(metricFilters)) {
          const mongoFilter: Record<string, number> = {};
          if (filter.min !== undefined) {
            mongoFilter["$gte"] = filter.min;
          }
          if (filter.max !== undefined) {
            mongoFilter["$lte"] = filter.max;
          }
          
          if (Object.keys(mongoFilter).length > 0) {
            mongoFilters[metricName] = mongoFilter;
          }
        }
        
        if (Object.keys(mongoFilters).length > 0) {
          params.append("metric_filters", JSON.stringify(mongoFilters));
        }
      }

      const queryString = params.toString();
      const url = `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}/features/${featureIndex}${queryString ? `?${queryString}` : ""}`;

      const feature = await fetch(url, {
        method: "GET",
        headers: {
          Accept: "application/x-msgpack",
        },
      })
        .then(async (res) => {
          if (!res.ok) {
            throw new Error(await res.text());
          }
          return res;
        })
        .then(async (res) => await res.arrayBuffer())
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        .then((res) => decode(new Uint8Array(res)) as any)
        .then((res) =>
          camelcaseKeys(res, {
            deep: true,
            stopPaths: ["sample_groups.samples.context"],
          })
        )
        .then((res) => FeatureSchema.parse(res));
      setFeatureIndex(feature.featureIndex);
      setSelectedAnalysis(feature.analysisName);
      setSearchParams({
        dictionary,
        featureIndex: feature.featureIndex.toString(),
        analysis: feature.analysisName,
      });
      return feature;
    }
  );

  useMount(async () => {
    await fetchDictionaries();
    if (searchParams.get("dictionary")) {
      const dict = searchParams.get("dictionary")!;
      const analysisParam = searchParams.get("analysis");
      setSelectedDictionary(dict);

      fetchAnalyses(dict).then((analyses) => {
        if (analyses.length > 0) {
          setSelectedAnalysis(analysisParam || analyses[0]);
        }
      });

      fetchMetrics(dict);

      if (searchParams.get("featureIndex")) {
        setFeatureIndex(parseInt(searchParams.get("featureIndex")!));
        fetchFeature(dict, searchParams.get("featureIndex")!, analysisParam || null);
      }
    }
  });

  useEffect(() => {
    if (dictionariesState.value && selectedDictionary === null) {
      setSelectedDictionary(dictionariesState.value[0]);
      fetchAnalyses(dictionariesState.value[0]).then((analyses) => {
        if (analyses.length > 0) {
          setSelectedAnalysis(analyses[0]);
        }
      });

      fetchMetrics(dictionariesState.value[0]);
      fetchFeature(dictionariesState.value[0], "random", selectedAnalysis);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dictionariesState.value]);

  useEffect(() => {
    if (selectedDictionary) {
      fetchAnalyses(selectedDictionary);
      fetchMetrics(selectedDictionary);
      setSelectedAnalysis(null);
      setMetricFilters({});
      setFeatureCount(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDictionary]);

  // Handle metric filter changes
  const handleMetricFilterChange = useCallback((metricName: string, type: 'min' | 'max', value: string) => {
    setMetricFilters(prev => ({
      ...prev,
      [metricName]: {
        ...prev[metricName],
        [type]: value === '' ? undefined : parseFloat(value)
      }
    }));
    // Clear count when filters change
    setFeatureCount(null);
  }, []);

  // Handle clear filters
  const handleClearFilters = useCallback(() => {
    setMetricFilters({});
    setFeatureCount(null);
  }, []);

  // Memoize sections calculation
  const sections = useMemo(() => [
    {
      title: "Histogram",
      id: "Histogram",
    },
    {
      title: "Decoder Norms",
      id: "DecoderNorms",
    },
    {
      title: "Similarity Matrix",
      id: "DecoderSimilarityMatrix",
    },
    {
      title: "Inner Product Matrix",
      id: "DecoderInnerProductMatrix",
    },
    {
      title: "Logits",
      id: "Logits",
    },
    {
      title: "Top Activation",
      id: "Activation",
    },
  ].filter((section) => (featureState.value && featureState.value.logits != null) || section.id !== "Logits"), [featureState.value]);

  return (
    <div id="Top">
      <AppNavbar />
      <div className="pt-4 pb-20 px-20 flex flex-col items-center gap-12">
        <div className="container grid grid-cols-[auto_600px_auto_auto] justify-center items-center gap-4">
          <span className="font-bold justify-self-end">Select dictionary:</span>
          <Select
            disabled={dictionariesState.loading || featureState.loading}
            value={selectedDictionary || undefined}
            onValueChange={(value) => {
              setSelectedDictionary(value);
            }}
          >
            <SelectTrigger className="bg-white">
              <SelectValue placeholder="Select a dictionary" />
            </SelectTrigger>
            <SelectContent>
              {dictionariesState.value?.map((dictionary, i) => (
                <SelectItem key={i} value={dictionary}>
                  {dictionary}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            disabled={dictionariesState.loading || featureState.loading}
            onClick={async () => {
              await fetchFeature(selectedDictionary, "random", selectedAnalysis);
            }}
          >
            Go
          </Button>
          <span className="font-bold"></span>

          <span className="font-bold justify-self-end">Select analysis:</span>
          <Select
            disabled={analysesState.loading || !selectedDictionary || featureState.loading}
            value={selectedAnalysis || undefined}
            onValueChange={setSelectedAnalysis}
          >
            <SelectTrigger className="bg-white">
              <SelectValue placeholder="Select an analysis" />
            </SelectTrigger>
            <SelectContent>
              {analysesState.value?.map((analysis, i) => (
                <SelectItem key={i} value={analysis}>
                  {analysis}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            disabled={analysesState.loading || !selectedDictionary || featureState.loading}
            onClick={async () => {
              await fetchFeature(selectedDictionary, featureIndex, selectedAnalysis);
            }}
          >
            Apply
          </Button>
          <span className="font-bold"></span>

          <span className="font-bold justify-self-end">Choose a specific feature:</span>
          <Input
            disabled={dictionariesState.loading || selectedDictionary === null || featureState.loading}
            id="feature-input"
            className="bg-white"
            type="number"
            value={inputValue}
            onChange={handleFeatureIndexChange}
          />
          <Button
            disabled={dictionariesState.loading || selectedDictionary === null || featureState.loading}
            onClick={async () => await fetchFeature(selectedDictionary, featureIndex, selectedAnalysis)}
          >
            Go
          </Button>
          <Button
            disabled={dictionariesState.loading || selectedDictionary === null || featureState.loading}
            onClick={async () => {
              await fetchFeature(selectedDictionary, "random", selectedAnalysis, metricFilters);
            }}
          >
            Show Random Feature
          </Button>

          {/* Metric filters section */}
          {metricsState.value && metricsState.value.length > 0 && (
            <>
              <span className="font-bold justify-self-end">Metric filters:</span>
              <div className="bg-white p-4 rounded-lg border grid grid-cols-2 gap-4 col-span-2">
                {metricsState.value.map((metric) => (
                  <div key={metric} className="flex flex-col gap-2">
                    <label className="text-sm font-medium">{metric}</label>
                    <div className="flex gap-2">
                      <Input
                        placeholder="Min"
                        type="number"
                        step="any"
                        value={metricFilters[metric]?.min?.toString() || ''}
                        onChange={(e) => handleMetricFilterChange(metric, 'min', e.target.value)}
                        className="text-xs"
                      />
                      <Input
                        placeholder="Max"
                        type="number"
                        step="any"
                        value={metricFilters[metric]?.max?.toString() || ''}
                        onChange={(e) => handleMetricFilterChange(metric, 'max', e.target.value)}
                        className="text-xs"
                      />
                    </div>
                  </div>
                ))}
              </div>
              <span className="font-bold"></span>

              <span className="font-bold justify-self-end">Filter actions:</span>
              <div className="flex gap-2 items-center">
                <Button
                  disabled={dictionariesState.loading || selectedDictionary === null || countState.loading}
                  onClick={async () => {
                    await countFeatures(selectedDictionary, selectedAnalysis, metricFilters);
                  }}
                >
                  Count Features
                </Button>
                <Button
                  variant="outline"
                  disabled={dictionariesState.loading || selectedDictionary === null}
                  onClick={handleClearFilters}
                >
                  Clear Filters
                </Button>
                {featureCount !== null && (
                  <span className="text-sm font-medium ml-2">
                    {countState.loading ? "Counting..." : `Found ${featureCount} features`}
                  </span>
                )}
              </div>
              <span className="font-bold"></span>
              <span className="font-bold"></span>
            </>
          )}
        </div>

        {featureState.loading && !loadingRandomFeature && (
          <div>
            Loading Feature <span className="font-bold">#{featureIndex}</span>...
          </div>
        )}
        {featureState.loading && loadingRandomFeature && <div>Loading Random Living Feature...</div>}
        {featureState.error && <div className="text-red-500 font-bold">Error: {featureState.error.message}</div>}
        {!featureState.loading && featureState.value && (
          <div className="flex gap-12 w-full">
            <Suspense fallback={<div>Loading Feature Card...</div>}>
              <FeatureCard feature={featureState.value} />
            </Suspense>
            <SectionNavigator sections={sections} />
          </div>
        )}
      </div>
    </div>
  );
};
