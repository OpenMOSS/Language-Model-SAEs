import { FeatureActivationSample } from "@/components/feature/sample";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { FeatureSchema } from "@/types/feature";
import { decode } from "@msgpack/msgpack";
import camelcaseKeys from "camelcase-keys";
import { useEffect, useState } from "react";
import { useAsyncFn, useMount } from "react-use";
import { z } from "zod";

export const FeaturesPage = () => {
  const [dictionariesState, fetchDictionaries] = useAsyncFn(async () => {
    return await fetch(`${import.meta.env.VITE_BACKEND_URL}/dictionaries`)
      .then(async (res) => await res.json())
      .then((res) => z.array(z.string()).parse(res));
  });

  const [selectedDictionary, setSelectedDictionary] = useState<string | null>(
    null
  );

  const [featureIndex, setFeatureIndex] = useState<number>(0);
  const [loadingRandomFeature, setLoadingRandomFeature] = useState<boolean>(false);

  const [featureState, fetchFeature] = useAsyncFn(
    async (dictionary: string | null, featureIndex: number | string = "random") => {
      if (!dictionary) {
        alert("Please select a dictionary first");
        return;
      }
      setLoadingRandomFeature(featureIndex === "random");
      const feature = await fetch(
        `${
          import.meta.env.VITE_BACKEND_URL
        }/dictionaries/${dictionary}/features/${featureIndex}`,
        {
          method: "GET",
          headers: {
            "Accept": "application/x-msgpack",
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
          camelcaseKeys(res, { deep: true, stopPaths: ["samples.context"] })
        )
        .then((res) => FeatureSchema.parse(res));
      setFeatureIndex(feature.featureIndex);
      return feature;
    }
  );

  useMount(async () => {
    await fetchDictionaries();
  });

  useEffect(() => {
    if (dictionariesState.value && selectedDictionary === null) {
      setSelectedDictionary(dictionariesState.value[0]);
      fetchFeature(dictionariesState.value[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dictionariesState.value]);

  return (
    <div className="p-20 flex flex-col items-center gap-12">
      <div className="container grid grid-cols-[auto_600px_auto_auto] justify-center items-center gap-4">
        <span className="font-bold justify-self-end">Select dictionary:</span>
        <Select
          value={selectedDictionary || undefined}
          onValueChange={setSelectedDictionary}
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
          onClick={async () => {
            await fetchFeature(selectedDictionary);
          }}
        >
          Go
        </Button>
        <span className="font-bold"></span>
        <span className="font-bold justify-self-end">
          Choose a specific feature:
        </span>
        <Input
          id="feature-input"
          className="bg-white"
          type="number"
          value={featureIndex.toString()}
          onChange={(e) => setFeatureIndex(parseInt(e.target.value))}
        />
        <Button
          onClick={async () =>
            await fetchFeature(selectedDictionary, featureIndex)
          }
        >
          Go
        </Button>
        <Button
          onClick={async () => {
            await fetchFeature(selectedDictionary);
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
      {featureState.loading && loadingRandomFeature && (
        <div>
          Loading Random Living Feature...
        </div>
      )}
      {featureState.error && <div>Error: {featureState.error.message}</div>}
      {!featureState.loading && featureState.value && (
        <Card className="container">
          <CardHeader>
            <CardTitle>#{featureState.value!.featureIndex}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col gap-4">
              <div className="flex flex-col w-full gap-4">
                <h2 className="text-xl font-bold py-1">
                  Top Activations (Max ={" "}
                  {featureState.value!.maxFeatureAct.toFixed(3)})
                </h2>
                {featureState.value!.samples.slice(0, 20).map((sample, i) => (
                  <FeatureActivationSample
                    key={i}
                    sample={sample}
                    sampleIndex={i}
                    maxFeatureAct={featureState.value!.maxFeatureAct}
                  />
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
