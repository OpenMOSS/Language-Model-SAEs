import { AppNavbar } from "@/components/app/navbar";
import { DictionaryCard } from "@/components/dictionary/dictionary-card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Combobox } from "@/components/ui/combobox";
import { DictionarySchema } from "@/types/dictionary";
import { decode } from "@msgpack/msgpack";
import camelcaseKeys from "camelcase-keys";
import { useState, useEffect, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import { useAsyncFn, useMount } from "react-use";
import { z } from "zod";

export const DictionaryPage = () => {
  const [searchParams, setSearchParams] = useSearchParams();

  const [dictionariesState, fetchDictionaries] = useAsyncFn(async () => {
    return await fetch(`${import.meta.env.VITE_BACKEND_URL}/dictionaries`)
      .then(async (res) => await res.json())
      .then((res) => z.array(z.string()).parse(res));
  });

  const [selectedDictionary, setSelectedDictionary] = useState<string | null>(null);
  const [selectedAnalysis, setSelectedAnalysis] = useState<string | null>(null);

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

  const [dictionaryState, fetchDictionary] = useAsyncFn(async (dictionary: string | null) => {
    if (!dictionary) {
      alert("Please select a dictionary first");
      return;
    }

    const feature = await fetch(`${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}`, {
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
        })
      )
      .then((res) => DictionarySchema.parse(res));

    setSearchParams({
      dictionary,
      ...(selectedAnalysis ? { analysis: selectedAnalysis } : {}),
    });

    return feature;
  });

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
      fetchDictionary(dict);
    }
  });

  useEffect(() => {
    if (dictionariesState.value && dictionariesState.value.length > 0 && selectedDictionary === null) {
      setSelectedDictionary(dictionariesState.value[0]);
      fetchAnalyses(dictionariesState.value[0]).then((analyses) => {
        if (analyses.length > 0) {
          setSelectedAnalysis(analyses[0]);
        }
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dictionariesState.value]);

  useEffect(() => {
    if (selectedDictionary) {
      fetchAnalyses(selectedDictionary);
      setSelectedAnalysis(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDictionary]);

  useEffect(() => {
    if (analysesState.value && analysesState.value.length > 0 && selectedAnalysis === null) {
      setSelectedAnalysis(analysesState.value[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysesState.value]);

  // Memoize dictionary options for Combobox
  const dictionaryOptions = useMemo(() => {
    if (!dictionariesState.value) return [];
    return dictionariesState.value.map((dict) => ({
      value: dict,
      label: dict,
    }));
  }, [dictionariesState.value]);

  return (
    <div>
      <AppNavbar />
      <div className="pt-4 pb-20 px-20 flex flex-col items-center gap-12">
        <div className="container grid grid-cols-[auto_600px_auto_auto_300px] justify-center items-center gap-4">
          <span className="font-bold justify-self-end">Select dictionary:</span>
          <Combobox
            disabled={dictionariesState.loading || dictionaryState.loading}
            value={selectedDictionary || null}
            onChange={(value) => {
              setSelectedDictionary(value);
            }}
            options={dictionaryOptions}
            placeholder="Select dictionary..."
            commandPlaceholder="Search dictionaries..."
            emptyIndicator="No matching dictionaries found"
            className="w-full"
          />
          <span className="font-bold justify-self-end">Select analysis:</span>
          <Select
            disabled={analysesState.loading || !selectedDictionary || dictionaryState.loading}
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
            disabled={dictionariesState.loading || dictionaryState.loading}
            onClick={async () => {
              await fetchDictionary(selectedDictionary);
            }}
          >
            Go
          </Button>
        </div>
        {dictionaryState.loading && (
          <div>
            Loading Dictionary <span className="font-bold">{selectedDictionary}</span>...
          </div>
        )}
        {dictionaryState.error && <div className="text-red-500 font-bold">Error: {dictionaryState.error.message}</div>}
        {!dictionaryState.loading && dictionaryState.value && <DictionaryCard dictionary={dictionaryState.value} />}
      </div>
    </div>
  );
};
