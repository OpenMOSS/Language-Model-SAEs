import { Feature, FeatureSampleCompactSchema } from "@/types/feature";
import { decode } from "@msgpack/msgpack";
import camelcaseKeys from "camelcase-keys";
import { useState, useEffect, useCallback } from "react";
import Plot from "react-plotly.js";
import { useAsyncFn } from "react-use";
import { Button } from "../ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";
import { Textarea } from "../ui/textarea";
import { FeatureInterpretation } from "./interpret";
import { FeatureActivationSample, FeatureSampleGroup } from "./sample";

const FeatureCustomInputArea = ({ feature }: { feature: Feature }) => {
  const [customInput, setCustomInput] = useState<string>("");
  const [state, submit] = useAsyncFn(async () => {
    if (!customInput) {
      alert("Please enter your input.");
      return;
    }
    return await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${feature.dictionaryName}/features/${
        feature.featureIndex
      }/custom?input_text=${encodeURIComponent(customInput)}`,
      {
        method: "POST",
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
          stopPaths: ["context"],
        })
      )
      .then((res) => FeatureSampleCompactSchema.parse(res));
  }, [customInput]);

  return (
    <div className="flex flex-col gap-4">
      <p className="font-bold">Custom Input</p>
      <Textarea
        placeholder="Type your custom input here."
        value={customInput}
        onChange={(e) => setCustomInput(e.target.value)}
      />
      <Button onClick={submit} disabled={state.loading}>
        Submit
      </Button>
      {state.error && <p className="text-red-500">{state.error.message}</p>}
      {state.value && (
        <>
          <FeatureActivationSample
            sample={state.value}
            sampleName="Custom Input"
            maxFeatureAct={feature.maxFeatureAct}
          />
          <p className="font-bold">Custom Input Max Activation: {Math.max(...state.value.featureActsValues.flat()).toFixed(3)}</p>
        </>
      )}
    </div>
  );
};

const FeatureBookmarkButton = ({ feature }: { feature: Feature }) => {
  const [isBookmarked, setIsBookmarked] = useState<boolean>(feature.isBookmarked || false);

  const toggleBookmark = useCallback(async () => {
    const method = isBookmarked ? "DELETE" : "POST";
    const response = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${feature.dictionaryName}/features/${
        feature.featureIndex
      }/bookmark`,
      {
        method,
      }
    );
    if (response.ok) {
      setIsBookmarked(!isBookmarked);
      return !isBookmarked;
    } else {
      throw new Error(await response.text());
    }
  }, [isBookmarked, feature.dictionaryName, feature.featureIndex]);

  const [toggleState, executeToggle] = useAsyncFn(toggleBookmark, [toggleBookmark]);

  useEffect(() => {
    setIsBookmarked(feature.isBookmarked || false);
  }, [feature.isBookmarked]);

  return (
    <Button
      onClick={executeToggle}
      disabled={toggleState.loading}
      variant={isBookmarked ? "default" : "outline"}
    >
      {toggleState.loading ? "..." : isBookmarked ? "★ Bookmarked" : "☆ Bookmark"}
    </Button>
  );
};

export const FeatureCard = ({ feature }: { feature: Feature }) => {
  console.log('🔄 FeatureCard recomputed', { 
    featureId: feature.featureIndex,
    dictionaryName: feature.dictionaryName,
    analysisName: feature.analysisName,
    actTimes: feature.actTimes,
    maxFeatureAct: feature.maxFeatureAct
  });

  const analysisNameMap = (analysisName: string) => {
    if (analysisName === "top_activations") {
      return "Top Activations";
    } else if (/^subsample-/.test(analysisName)) {
      const [, proportion] = analysisName.split("-");
      const percentage = parseFloat(proportion) * 100;
      return `Subsample ${percentage}%`;
    } else {
      return analysisName;
    }
  };

  const [showCustomInput, setShowCustomInput] = useState<boolean>(false);

  const activationTimesSpan = feature.nAnalyzedTokens ? (
    <span className="font-medium">
      (Activation Times ={" "}
      <span className="font-bold">
        {feature.actTimes}
        {feature.actTimesModalities &&
          ` = ${Object.entries(feature.actTimesModalities)
            .map(([modality, actTime]) => `${actTime} (${modality})`)
            .join(" + ")}`}
        )
      </span>
    </span>
  ) : (
    <span className="font-medium">
      (Activation Frequency ={" "}
      <span className="font-bold">
        {(feature.actTimes / feature.nAnalyzedTokens!).toFixed(3)}
        {feature.actTimesModalities &&
          ` = ${Object.entries(feature.actTimesModalities)
            .map(([modality, actTime]) => `${(actTime / feature.nAnalyzedTokens!).toFixed(3)} (${modality})`)
            .join(" + ")}`}
        )
      </span>
    </span>
  );

  const maxActivationSpan = (
    <span className="font-medium">
      (Max Activation ={" "}
      <span className="font-bold">
        {feature.maxFeatureAct}
        {feature.maxFeatureActsModalities &&
          ` = max(${Object.entries(feature.maxFeatureActsModalities)
            .map(([modality, maxFeatureAct]) => `${maxFeatureAct} (${modality})`)
            .join(", ")})`}
        )
      </span>
    </span>
  );

  return (
    <Card id="Interp." className="container">
      <CardHeader>
        <CardTitle className="flex justify-between items-center text-xl">
          <span>
            #{feature.featureIndex} {activationTimesSpan}
            {maxActivationSpan}
          </span>
          <div className="flex gap-2">
            <FeatureBookmarkButton feature={feature} />
            <Button onClick={() => setShowCustomInput((prev) => !prev)}>
              {showCustomInput ? "Hide Custom Input" : "Try Custom Input"}
            </Button>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col gap-4">
          {showCustomInput && <FeatureCustomInputArea feature={feature} />}

          <FeatureInterpretation feature={feature} />

          {feature.decoderNorms && (
            <div id="DecoderNorms" className="flex flex-col w-full gap-4">
              <p className="font-bold">Decoder Norms</p>
              <Plot
                data={[
                  {
                    x: Array.from({ length: feature.decoderNorms.length }, (_, i) => i),
                    y: feature.decoderNorms,
                    type: "bar",
                    marker: { color: "#636EFA" },
                    hovertemplate: "Index: %{x}<br>Norm: %{y}<extra></extra>",
                  },
                ]}
                layout={{
                  xaxis: { title: "Output Feature Index" },
                  yaxis: { title: "Norm" },
                  bargap: 0.2,
                  margin: { t: 0, b: 40 },
                  showlegend: false,
                  height: 300,
                }}
                config={{ responsive: true }}
              />
            </div>
          )}

          {(feature.decoderSimilarityMatrix || feature.decoderInnerProductMatrix) && (
            <div className="flex flex-col w-full gap-4">
              <div className="flex justify-between gap-4">
                {feature.decoderSimilarityMatrix && (
                  <div id="DecoderSimilarityMatrix" className="flex flex-col w-1/2 gap-2">
                    <p className="font-bold">Decoder Similarity Matrix</p>
                    <Plot
                      data={[
                        {
                          z: feature.decoderSimilarityMatrix,
                          type: "heatmap",
                          colorscale: "Viridis",
                          hovertemplate: "Row: %{y}<br>Column: %{x}<br>Value: %{z}<extra></extra>",
                        },
                      ]}
                      layout={{
                        xaxis: {
                          title: "Head Index",
                          scaleanchor: "y",
                          scaleratio: 1,
                          constrain: "domain",
                        },
                        yaxis: {
                          title: "Head Index",
                          constrain: "domain",
                        },
                        margin: { t: 10, b: 50, l: 60, r: 10 },
                        height: 400,
                        width: 400,
                      }}
                      config={{ responsive: true }}
                    />
                  </div>
                )}

                {feature.decoderInnerProductMatrix && (
                  <div id="DecoderInnerProductMatrix" className="flex flex-col w-1/2 gap-2">
                    <p className="font-bold">Decoder Inner Product Matrix</p>
                    <Plot
                      data={[
                        {
                          z: feature.decoderInnerProductMatrix,
                          type: "heatmap",
                          colorscale: "Viridis",
                          hovertemplate: "Row: %{y}<br>Column: %{x}<br>Value: %{z}<extra></extra>",
                        },
                      ]}
                      layout={{
                        xaxis: {
                          title: "Head Index",
                          scaleanchor: "y",
                          scaleratio: 1,
                          constrain: "domain",
                        },
                        yaxis: {
                          title: "Head Index",
                          constrain: "domain",
                        },
                        margin: { t: 10, b: 50, l: 60, r: 10 },
                        height: 400,
                        width: 400,
                      }}
                      config={{ responsive: true }}
                    />
                  </div>
                )}
              </div>
            </div>
          )}

          {feature.featureActivationHistogram && (
            <div id="Histogram" className="flex flex-col w-full gap-4">
              <p className="font-bold">Activation Histogram</p>
              <Plot
                data={feature.featureActivationHistogram}
                layout={{
                  xaxis: { title: "Activation" },
                  yaxis: { title: "Count" },
                  bargap: 0.2,
                  margin: { t: 0, b: 40 },
                  showlegend: false,
                }}
              />
            </div>
          )}

          {feature.logits && (
            <div id="Logits" className="flex flex-col w-full gap-4">
              <p className="font-bold">Logits</p>
              <div className="flex gap-4">
                <div className="flex flex-col w-1/2 gap-4">
                  <p className="font-bold">Top Positive</p>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Token</TableHead>
                        <TableHead>Logit</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {feature.logits.topPositive.map((token) => (
                        <TableRow key={token.token}>
                          <TableCell className="underline whitespace-pre-wrap decoration-slate-400 decoration-1 decoration-dotted underline-offset-[6px]">
                            {token.token}
                          </TableCell>
                          <TableCell>{token.logit.toFixed(3)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
                <div className="flex flex-col w-1/2 gap-4">
                  <p className="font-bold">Top Negative</p>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Token</TableHead>
                        <TableHead>Logit</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {feature.logits.topNegative.map((token) => (
                        <TableRow key={token.token}>
                          <TableCell className="underline whitespace-pre-wrap decoration-slate-400 decoration-1 decoration-dotted underline-offset-[6px]">
                            {token.token}
                          </TableCell>
                          <TableCell>{token.logit.toFixed(3)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </div>
              <Plot
                data={feature.logits.histogram}
                layout={{
                  bargap: 0.2,
                  margin: { t: 0, b: 40 },
                  showlegend: false,
                }}
              />
            </div>
          )}

          <div id="Activation" className="flex flex-col w-full gap-4">
            <Tabs defaultValue="top_activations">
              <TabsList className="font-bold">
                {feature.sampleGroups.slice(0, feature.sampleGroups.length / 2).map((sampleGroup) => (
                  <TabsTrigger key={`tab-trigger-${sampleGroup.analysisName}`} value={sampleGroup.analysisName}>
                    {analysisNameMap(sampleGroup.analysisName)}
                  </TabsTrigger>
                ))}
              </TabsList>
              <TabsList className="font-bold">
                {feature.sampleGroups
                  .slice(feature.sampleGroups.length / 2, feature.sampleGroups.length)
                  .map((sampleGroup) => (
                    <TabsTrigger key={`tab-trigger-${sampleGroup.analysisName}`} value={sampleGroup.analysisName}>
                      {analysisNameMap(sampleGroup.analysisName)}
                    </TabsTrigger>
                  ))}
              </TabsList>
              {feature.sampleGroups.map((sampleGroup) => (
                <TabsContent
                  key={`tab-content-${sampleGroup.analysisName}`}
                  value={sampleGroup.analysisName}
                  className="mt-0"
                >
                  <FeatureSampleGroup feature={feature} sampleGroup={sampleGroup} />
                </TabsContent>
              ))}
            </Tabs>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
