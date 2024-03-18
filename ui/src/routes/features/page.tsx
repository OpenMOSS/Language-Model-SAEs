import { FeatureActivationSample } from "@/components/feature/sample";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { FeatureSchema } from "@/types/feature";
import { decode } from "@msgpack/msgpack";
import camelcaseKeys from "camelcase-keys";
import { useState } from "react";
import { useAsyncFn, useMount } from "react-use";

export const FeaturesPage = () => {
  const [featureIndex, setFeatureIndex] = useState<number>(
    Math.floor(Math.random() * 24576)
  );

  const [state, fetchFeature] = useAsyncFn(async (featureIndex: number) => {
    return await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/features/${featureIndex}`,
      {
        method: "GET",
        headers: {
          "Content-Type": "application/x-msgpack",
        },
      }
    )
      .then(async (res) => await res.arrayBuffer())
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      .then((res) => decode(new Uint8Array(res)) as any)
      .then((res) =>
        camelcaseKeys(res, { deep: true, stopPaths: ["samples.context"] })
      )
      .then((res) => FeatureSchema.parse(res));
  });

  useMount(async () => {
    await fetchFeature(featureIndex);
  });

  return (
    <div className="p-20 flex flex-col items-center gap-12">
      <div className="container flex justify-center items-center gap-4">
        <span className="font-bold">Choose a specific feature: </span>
        <Input
          id="feature-input"
          className="bg-white w-[600px]"
          type="number"
          value={featureIndex.toString()}
          onChange={(e) => setFeatureIndex(parseInt(e.target.value))}
        />
        <Button onClick={async () => await fetchFeature(featureIndex)}>
          Go
        </Button>
        <Button
          onClick={async () => {
            const featureIndex = Math.floor(Math.random() * 24576);
            setFeatureIndex(featureIndex);
            await fetchFeature(featureIndex);
          }}
        >
          Show Random Feature
        </Button>
      </div>
      {state.loading && (
        <div>
          Loading Feature <span className="font-bold">#{featureIndex}</span>...
        </div>
      )}
      {state.error && <div>Error: {state.error.message}</div>}
      {!state.loading && state.value && (
        <Card className="container">
          <CardHeader>
            <CardTitle>#{state.value!.featureIndex}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col gap-4">
              <div className="flex flex-col w-full gap-4">
                <h2 className="text-xl font-bold py-1">
                  Top Activations (Max = {state.value!.maxFeatureAct.toFixed(3)}
                  )
                </h2>
                {state.value!.samples.slice(0, 20).map((sample, i) => (
                  <FeatureActivationSample
                    key={i}
                    sample={sample}
                    sampleIndex={i}
                    maxFeatureAct={state.value!.maxFeatureAct}
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
