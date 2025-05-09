import { AppNavbar } from "@/components/app/navbar";
import { FeatureCard } from "@/components/feature/feature-card";
import { SectionNavigator } from "@/components/app/section-navigator";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { FeatureSchema } from "@/types/feature";
import { decode } from "@msgpack/msgpack";
import camelcaseKeys from "camelcase-keys";
import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useAsyncFn, useMount } from "react-use";
import { z } from "zod";

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

  const [selectedAnalysis, setSelectedAnalysis] = useState<string>("default");

  const [featureIndex, setFeatureIndex] = useState<number>(0);
  const [loadingRandomFeature, setLoadingRandomFeature] = useState<boolean>(false);

  const [featureState, fetchFeature] = useAsyncFn(
    async (dictionary: string | null, featureIndex: number | string = "random", analysisName: string = "default") => {
      if (!dictionary) {
        alert("Please select a dictionary first");
        return;
      }

      setLoadingRandomFeature(featureIndex === "random");

      const feature = await fetch(
        `${
          import.meta.env.VITE_BACKEND_URL
        }/dictionaries/${dictionary}/features/${featureIndex}?feature_analysis_name=${analysisName}`,
        {
          method: "GET",
          headers: {
            Accept: "application/x-msgpack",
          },
        }
      )
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
      setSearchParams({
        dictionary,
        featureIndex: feature.featureIndex.toString(),
        analysis: analysisName,
      });
      return feature;
    }
  );

  useMount(async () => {
    await fetchDictionaries();
    if (searchParams.get("dictionary")) {
      const dict = searchParams.get("dictionary")!;
      setSelectedDictionary(dict);

      await fetchAnalyses(dict);

      const analysisParam = searchParams.get("analysis");
      if (analysisParam) {
        setSelectedAnalysis(analysisParam);
      }

      if (searchParams.get("featureIndex")) {
        setFeatureIndex(parseInt(searchParams.get("featureIndex")!));
        fetchFeature(dict, searchParams.get("featureIndex")!, analysisParam || "default");
      }
    }
  });

  useEffect(() => {
    if (dictionariesState.value && selectedDictionary === null) {
      setSelectedDictionary(dictionariesState.value[0]);
      fetchAnalyses(dictionariesState.value[0]);
      fetchFeature(dictionariesState.value[0], "random", "default");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dictionariesState.value]);

  useEffect(() => {
    if (selectedDictionary) {
      fetchAnalyses(selectedDictionary);
      setSelectedAnalysis("default");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDictionary]);

  const sections = [
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
  ].filter((section) => (featureState.value && featureState.value.logits != null) || section.id !== "Logits");

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
            value={selectedAnalysis}
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
            value={featureIndex.toString()}
            onChange={(e) => setFeatureIndex(parseInt(e.target.value))}
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
              await fetchFeature(selectedDictionary, "random", selectedAnalysis);
            }}
          >
            Show Random Feature
          </Button>
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
            <FeatureCard feature={featureState.value} />
            <SectionNavigator sections={sections} />
          </div>
        )}
      </div>
    </div>
  );
};
