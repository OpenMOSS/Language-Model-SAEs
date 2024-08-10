import { Fragment, useEffect, useMemo, useState } from "react";
import { Button } from "../ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Textarea } from "../ui/textarea";
import { useAsyncFn, useMount } from "react-use";
import camelcaseKeys from "camelcase-keys";
import snakecaseKeys from "snakecase-keys";
import { decode } from "@msgpack/msgpack";
import { ModelGeneration, ModelGenerationSchema } from "@/types/model";
import { Sample } from "../app/sample";
import { cn } from "@/lib/utils";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Label, LabelList, ResponsiveContainer } from "recharts";
import { zip } from "@/utils/array";
import { Input } from "../ui/input";
import { countTokenGroupPositions, groupToken, hex } from "@/utils/token";
import { getAccentClassname } from "@/utils/style";
import { Separator } from "../ui/separator";
import MultipleSelector from "../ui/multiple-selector";
import { z } from "zod";
import { Combobox } from "../ui/combobox";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { Ban, MoreHorizontal, Plus, Trash2, Wrench, X } from "lucide-react";
import { ColumnDef } from "@tanstack/react-table";
import { FeatureLinkWithPreview } from "../app/feature-preview";
import { DataTable } from "../ui/data-table";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "../ui/dropdown-menu";
import { useNavigate } from "react-router-dom";
import { Switch } from "../ui/switch";
import { Label as SLabel } from "../ui/label";
import { Toggle } from "../ui/toggle";

const SAEInfo = ({
  saeInfo,
  saeSettings,
  onSteerFeature,
  setSAESettings,
}: {
  saeInfo: {
    name: string;
    featureActs: {
      featureActIndex: number;
      featureAct: number;
      maxFeatureAct: number;
    }[];
  };
  saeSettings: { sortedBySum: boolean };
  onSteerFeature?: (name: string, featureIndex: number) => void;
  setSAESettings: (settings: { sortedBySum: boolean }) => void;
}) => {
  const navigate = useNavigate();

  const columns: ColumnDef<{
    featureActIndex: number;
    featureAct: number;
    maxFeatureAct: number;
  }>[] = [
    {
      accessorKey: "featureActIndex",
      header: () => (
        <div>
          <span className="font-bold">Feature</span>
        </div>
      ),
      cell: ({ row }) => (
        <div>
          <FeatureLinkWithPreview dictionaryName={saeInfo.name} featureIndex={row.original.featureActIndex} />
        </div>
      ),
    },
    {
      accessorKey: "featureAct",
      header: () => (
        <div>
          <span className="font-bold">Activation</span>
        </div>
      ),
      cell: ({ row }) => (
        <div className={cn(getAccentClassname(row.original.featureAct, row.original.maxFeatureAct, "text"))}>
          {row.original.featureAct.toFixed(3)}
        </div>
      ),
    },
    {
      id: "actions",
      enableHiding: false,
      cell: ({ row }) => (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className="h-8 w-8 p-0">
              <span className="sr-only">Open menu</span>
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuLabel>Actions</DropdownMenuLabel>
            <DropdownMenuItem onClick={() => onSteerFeature?.(saeInfo.name, row.original.featureActIndex)}>
              Steer feature
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              onClick={() => {
                navigate(
                  `/features?dictionary=${encodeURIComponent(saeInfo.name)}&featureIndex=${
                    row.original.featureActIndex
                  }`
                );
              }}
            >
              View feature
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      ),
      meta: {
        headerClassName: "w-16",
        cellClassName: "py-0",
      },
    },
  ];

  return (
    <div className="flex flex-col gap-4">
      <div className="flex justify-between items-center">
        <div className="text-sm font-bold">Features from {saeInfo.name}:</div>
        <Toggle
          className="-my-2"
          variant="outline"
          size="sm"
          pressed={saeSettings.sortedBySum}
          onPressedChange={(pressed) => setSAESettings({ ...saeSettings, sortedBySum: pressed })}
        >
          <span className="text-sm font-bold">Sort by sum</span>
        </Toggle>
      </div>
      <DataTable
        columns={columns}
        data={saeInfo.featureActs}
        pageSize={5}
        key={saeSettings.sortedBySum ? "sum" : "act"}
      />
    </div>
  );
};

const LogitsInfo = ({
  logits,
}: {
  logits: {
    logits: number;
    token: Uint8Array;
  }[];
}) => {
  const maxLogits = Math.max(...logits.map((logit) => logit.logits));
  const columns: ColumnDef<{
    logits: number;
    token: Uint8Array;
  }>[] = [
    {
      accessorKey: "token",
      header: () => (
        <div>
          <span className="font-bold">Token</span>
        </div>
      ),
      cell: ({ row }) => (
        <div>
          <span className="underline whitespace-pre-wrap">{hex(row.original)}</span>
        </div>
      ),
    },
    {
      accessorKey: "logits",
      header: () => (
        <div>
          <span className="font-bold">Logits</span>
        </div>
      ),
      cell: ({ row }) => (
        <div className={cn(getAccentClassname(row.original.logits, maxLogits, "text"))}>
          {row.original.logits.toFixed(3)}
        </div>
      ),
    },
  ];

  return (
    <div className="flex flex-col gap-4">
      <div className="text-sm font-bold">Logits:</div>
      <DataTable columns={columns} data={logits} pageSize={5} />
    </div>
  );
};

const LogitsBarChart = ({
  tokens,
}: {
  tokens: {
    logits: {
      logits: number;
      token: Uint8Array;
    }[];
    token: Uint8Array;
  }[];
}) => {
  const data = tokens.map((token) =>
    Object.assign(
      {},
      ...token.logits.map((logit, j) => ({
        [`logits-${j}`]: logit.logits,
        [`logits-token-${j}`]: hex(logit),
      })),
      {
        name: hex(token),
      }
    )
  );

  const colors = ["#8884d8", "#82ca9d", "#ffc658", "#ff7300", "#d6d6d6"];

  return (
    <ResponsiveContainer height={300}>
      <BarChart data={data} margin={{ top: 50, right: 50, left: 50, bottom: 15 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name">
          <Label value="Tokens" offset={0} position="bottom" />
        </XAxis>
        <YAxis label={{ value: "Logits", angle: -90, position: "left", textAnchor: "middle" }} />
        {tokens[0].logits.slice(0, 5).map((_, i) => (
          <Bar key={i} dataKey={`logits-${i}`} fill={colors[i]}>
            <LabelList dataKey={`logits-token-${i}`} position="top" />
          </Bar>
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
};

const ModelSample = ({
  sample,
  onSteerFeature,
}: {
  sample: ModelGeneration;
  onSteerFeature?: (name: string, featureIndex: number) => void;
}) => {
  const [selectedTokenGroupIndices, setSelectedTokenGroupIndices] = useState<number[]>([]);
  const toggleSelectedTokenGroupIndex = (tokenGroupIndex: number) => {
    setSelectedTokenGroupIndices((prev) =>
      prev.includes(tokenGroupIndex)
        ? prev.filter((t) => t !== tokenGroupIndex)
        : [...prev, tokenGroupIndex].sort((a, b) => a - b)
    );
  };

  const [saeSettings, setSAESettings] = useState<{ [name: string]: { sortedBySum: boolean } }>({});
  const getSAESettings = (name: string) => saeSettings[name] || { sortedBySum: false };

  useEffect(() => {
    setSelectedTokenGroupIndices([]);
  }, [sample]);

  const tokens = useMemo(() => {
    const saeInfo =
      sample.saeInfo.length > 0
        ? zip(
            ...sample.saeInfo.map((sae) =>
              zip(sae.featureActsIndices, sae.featureActs, sae.maxFeatureActs).map(
                ([featureActIndex, featureAct, maxFeatureAct]) => ({
                  name: sae.name,
                  featureActs: zip(featureActIndex, featureAct, maxFeatureAct).map(
                    ([featureActIndex, featureAct, maxFeatureAct]) => ({
                      featureActIndex,
                      featureAct,
                      maxFeatureAct,
                    })
                  ),
                })
              )
            )
          )
        : sample.context.map(() => []);

    return zip(sample.context, sample.inputMask, sample.logits, sample.logitsTokens, saeInfo).map(
      ([token, inputMask, logits, logitsTokens, saeInfo]) => ({
        token,
        inputMask,
        logits: zip(logits, logitsTokens).map(([logits, token]) => ({
          logits,
          token,
        })),
        saeInfo,
      })
    );
  }, [sample]);

  type Token = (typeof tokens)[0];

  const sortTokenInfo = (tokens: Token[]) => {
    const featureActSum = tokens.reduce((acc, token) => {
      token.saeInfo.forEach((saeInfo) => {
        saeInfo.featureActs.forEach((featureAct) => {
          acc[saeInfo.name] = acc[saeInfo.name] || {};
          acc[saeInfo.name][featureAct.featureActIndex.toString()] =
            acc[saeInfo.name][featureAct.featureActIndex.toString()] || 0;
          acc[saeInfo.name][featureAct.featureActIndex.toString()] += featureAct.featureAct;
        });
      });
      return acc;
    }, {} as { [name: string]: { [featureIndex: string]: number } });

    return tokens.map((token) => ({
      ...token,
      logits: token.logits.sort((a, b) => b.logits - a.logits),
      saeInfo: token.saeInfo.map((saeInfo) => ({
        ...saeInfo,
        featureActs: getSAESettings(saeInfo.name).sortedBySum
          ? saeInfo.featureActs.sort(
              (a, b) =>
                featureActSum[saeInfo.name][b.featureActIndex.toString()] -
                featureActSum[saeInfo.name][a.featureActIndex.toString()]
            )
          : saeInfo.featureActs.sort((a, b) => b.featureAct - a.featureAct),
      })),
    }));
  };

  const tokenGroups = groupToken(tokens);
  const tokenGroupPositions = countTokenGroupPositions(tokenGroups);
  const selectedTokenGroups = selectedTokenGroupIndices.map((i) => tokenGroups[i]);
  const selectedTokens = sortTokenInfo(selectedTokenGroups.flatMap((t) => t));
  const selectedTokenGroupPositions = selectedTokenGroupIndices.map((i) => tokenGroupPositions[i]);
  const selectedTokenPositions = selectedTokenGroups.flatMap((t, i) =>
    t.map((_, j) => selectedTokenGroupPositions[i] + j)
  );

  return (
    <div className="flex flex-col gap-4">
      <Sample
        sampleName={`Generation`}
        tokenGroups={tokenGroups}
        tokenGroupClassName={(_, tokenIndex) =>
          cn(
            "hover:shadow-lg hover:text-gray-600 cursor-pointer",
            selectedTokenGroupIndices.some((t) => t === tokenIndex) && "bg-orange-500"
          )
        }
        tokenGroupProps={(_, i) => ({
          onClick: () => toggleSelectedTokenGroupIndex(i),
        })}
      />

      {selectedTokens.length > 0 && (
        <p className="font-bold">
          Detail of {selectedTokens.length} Selected Token{selectedTokens.length > 1 ? "s" : ""}:
        </p>
      )}

      {selectedTokens.map((token, i) => (
        <Fragment key={selectedTokenPositions[i]}>
          <div className="flex flex-col gap-4">
            <div className="grid grid-cols-2 gap-x-8 gap-y-4">
              <div className="text-sm font-bold">Token:</div>
              <div className="text-sm underline whitespace-pre-wrap">{hex(token)}</div>
              <div className="text-sm font-bold">Position:</div>
              <div className="text-sm">{selectedTokenPositions[i]}</div>
            </div>
          </div>
          <div key={i} className="grid grid-cols-2 gap-8">
            <LogitsInfo logits={token.logits} />
            {token.saeInfo.map((saeInfo, j) => (
              <SAEInfo
                key={j}
                saeInfo={saeInfo}
                saeSettings={getSAESettings(saeInfo.name)}
                onSteerFeature={onSteerFeature}
                setSAESettings={(settings) => setSAESettings({ ...saeSettings, [saeInfo.name]: settings })}
              />
            ))}
          </div>
          {i < selectedTokens.length - 1 && <Separator />}
        </Fragment>
      ))}

      {selectedTokens.length > 0 && <LogitsBarChart tokens={selectedTokens} />}
    </div>
  );
};

const ModelCustomInputArea = () => {
  const [customInput, setCustomInput] = useState<string>("");
  const [doGenerate, setDoGenerate] = useState<boolean>(true);
  const [maxNewTokens, setMaxNewTokens] = useState<number>(128);
  const [topK, setTopK] = useState<number>(50);
  const [topP, setTopP] = useState<number>(0.95);
  const [selectedDictionaries, setSelectedDictionaries] = useState<string[]>([]);
  const [steerings, setSteerings] = useState<
    {
      sae: string | null;
      featureIndex: number;
      steeringType: "times" | "ablate" | "add" | "set";
      steeringValue: number | null;
    }[]
  >([{ sae: null, featureIndex: 0, steeringType: "times", steeringValue: 1 }]);

  const [sample, setSample] = useState<ModelGeneration | null>(null);

  const [dictionariesState, fetchDictionaries] = useAsyncFn(async () => {
    return await fetch(`${import.meta.env.VITE_BACKEND_URL}/dictionaries`)
      .then(async (res) => await res.json())
      .then((res) => z.array(z.string()).parse(res));
  });

  useMount(async () => {
    await fetchDictionaries();
  });

  const [state, submit] = useAsyncFn(async () => {
    if (!customInput) {
      alert("Please enter your input.");
      return;
    }
    const sample = await fetch(`${import.meta.env.VITE_BACKEND_URL}/model/generate`, {
      method: "POST",
      body: JSON.stringify(
        snakecaseKeys({
          inputText: customInput,
          maxNewTokens: doGenerate ? maxNewTokens : 0,
          topK,
          topP,
          saes: selectedDictionaries,
          steerings: steerings.filter((s) => s.sae !== null),
        })
      ),
      headers: {
        Accept: "application/x-msgpack",
        "Content-Type": "application/json",
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
          stopPaths: ["context", "logits_tokens"],
        })
      )
      .then((res) => ModelGenerationSchema.parse(res));
    setSample(sample);
  }, [doGenerate, customInput, maxNewTokens, topK, topP, selectedDictionaries]);

  return (
    <div className="flex flex-col gap-4">
      <p className="font-bold">Generation</p>
      <div className="container grid grid-cols-4 justify-center items-center gap-4 px-20">
        <span className="font-bold justify-self-end">Do generate:</span>
        <div className="col-span-3 flex items-center space-x-2">
          <Switch id="do-generate" checked={doGenerate} onCheckedChange={setDoGenerate} disabled={state.loading} />
          <SLabel htmlFor="do-generate" className="text-sm text-muted-foreground">
            {doGenerate
              ? "Model will generate new contents based on the following configuration."
              : "Model will only perceive the given input."}
          </SLabel>
        </div>
        <span className="font-bold justify-self-end">Max new tokens:</span>
        <Input
          disabled={state.loading || !doGenerate}
          className="bg-white"
          type="number"
          value={maxNewTokens.toString()}
          onChange={(e) => setMaxNewTokens(parseInt(e.target.value))}
        />
        <span className="font-bold justify-self-end">Top K:</span>
        <Input
          disabled={state.loading || !doGenerate}
          className="bg-white"
          type="number"
          value={topK.toString()}
          onChange={(e) => setTopK(parseInt(e.target.value))}
        />
        <span className="font-bold justify-self-end">Top P:</span>
        <Input
          disabled={state.loading || !doGenerate}
          className="bg-white"
          type="number"
          value={topP.toString()}
          onChange={(e) => setTopP(parseFloat(e.target.value))}
        />
        <Separator className="col-span-4" />
        <span className="font-bold justify-self-end">SAEs:</span>
        <MultipleSelector
          className="bg-white"
          disabled={state.loading}
          options={dictionariesState.value?.map((name) => ({ value: name, label: name })) || []}
          commandProps={{
            className: "col-span-3",
          }}
          hidePlaceholderWhenSelected
          placeholder="Bind SAEs to the language model to see features activated in the generation."
          value={selectedDictionaries.map((name) => ({ value: name, label: name }))}
          onChange={(value) => setSelectedDictionaries(value.map((v) => v.value))}
          emptyIndicator={
            <p className="w-full text-center text-lg leading-10 text-muted-foreground">No dictionaries found.</p>
          }
        />
        {selectedDictionaries.length > 0 &&
          steerings.map((steering, i) => (
            <Fragment key={i}>
              <span className="font-bold justify-self-end">Steering {i + 1}:</span>
              <div className="relative group col-span-3">
                <div className="w-full flex items-center gap-2">
                  <Combobox
                    className="min-w-[350px]"
                    value={steering.sae}
                    onChange={(value) => {
                      setSteerings((prev) => prev.map((s, j) => (i === j ? { ...s, sae: value } : s)));
                    }}
                    options={selectedDictionaries.map((name) => ({ value: name, label: name }))}
                    disabled={state.loading}
                    placeholder="Select a dictionary to steer the generation."
                    commandPlaceholder="Search for a selected dictionary..."
                    emptyIndicator="No options found"
                  />
                  <span className="text-muted-foreground">#</span>
                  <Input
                    disabled={state.loading}
                    className="bg-white"
                    type="number"
                    value={steering.featureIndex.toString()}
                    onChange={(e) => {
                      setSteerings((prev) =>
                        prev.map((s, j) => (i === j ? { ...s, featureIndex: parseInt(e.target.value) } : s))
                      );
                    }}
                  />
                  <Select
                    disabled={state.loading}
                    value={steering.steeringType}
                    onValueChange={(value) => {
                      if (value === "times" || value === "add" || value === "set")
                        setSteerings((prev) =>
                          prev.map((s, j) => (i === j ? { ...s, steeringType: value, steeringValue: 1 } : s))
                        );
                      else if (value === "ablate")
                        setSteerings((prev) =>
                          prev.map((s, j) => (i === j ? { ...s, steeringType: value, steeringValue: null } : s))
                        );
                    }}
                  >
                    <SelectTrigger className="bg-white w-16">
                      <SelectValue placeholder="Select a dictionary" />
                    </SelectTrigger>
                    <SelectContent className="w-16">
                      <SelectItem value="times" className="items-center">
                        <X className="h-4 w-4" />
                      </SelectItem>
                      <SelectItem value="ablate" className="items-center">
                        <Ban className="h-4 w-4" />
                      </SelectItem>
                      <SelectItem value="add" className="items-center">
                        <Plus className="h-4 w-4" />
                      </SelectItem>
                      <SelectItem value="set" className="items-center">
                        <Wrench className="h-4 w-4" />
                      </SelectItem>
                    </SelectContent>
                  </Select>
                  <Input
                    disabled={state.loading || steering.steeringType === "ablate"}
                    className="bg-white"
                    type={steering.steeringType === "ablate" ? "text" : "number"}
                    value={steering.steeringValue?.toString() || ""}
                    onChange={(e) => {
                      setSteerings((prev) =>
                        prev.map((s, j) => (i === j ? { ...s, steeringValue: parseFloat(e.target.value) } : s))
                      );
                    }}
                  />
                </div>
                <div
                  className={cn(
                    "absolute -right-10 top-1/2 transform -translate-y-1/2 w-8 h-8 flex items-center justify-center rounded-full bg-white hover:bg-red-500 hover:text-white cursor-pointer opacity-0 group-hover:opacity-100 transition-opacity select-none"
                  )}
                  onClick={() =>
                    steerings.length > 1
                      ? setSteerings((prev) => prev.filter((_, j) => i !== j))
                      : setSteerings([{ sae: null, featureIndex: 0, steeringType: "times", steeringValue: 1 }])
                  }
                >
                  <Trash2 className="h-4 w-4" />
                </div>
                <div
                  className={cn(
                    "absolute -right-20 top-1/2 transform -translate-y-1/2 w-8 h-8 flex items-center justify-center rounded-full bg-white hover:bg-green-500 hover:text-white cursor-pointer opacity-0 group-hover:opacity-100 transition-opacity select-none"
                  )}
                  onClick={() =>
                    setSteerings((prev) => [
                      ...prev.slice(0, i + 1),
                      { sae: null, featureIndex: 0, steeringType: "times", steeringValue: 1 },
                      ...prev.slice(i + 1),
                    ])
                  }
                >
                  <Plus className="h-4 w-4" />
                </div>
              </div>
            </Fragment>
          ))}
      </div>
      <Textarea
        placeholder="Type your custom input here."
        value={customInput}
        onChange={(e) => setCustomInput(e.target.value)}
      />
      <Button onClick={submit} disabled={state.loading}>
        Submit
      </Button>
      {state.error && <p className="text-red-500">{state.error.message}</p>}
      {sample && (
        <ModelSample
          sample={sample}
          onSteerFeature={(name, featureIndex) => {
            setSteerings((prev) => {
              const steerings = prev[prev.length - 1].sae ? [...prev] : prev.slice(0, -1);
              return [...steerings, { sae: name, featureIndex, steeringType: "times", steeringValue: 1 }];
            });
          }}
        />
      )}
    </div>
  );
};

export const ModelCard = () => {
  return (
    <Card className="container">
      <CardHeader>
        <CardTitle className="flex justify-between items-center text-xl">
          <span>Model</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col gap-4">
          <ModelCustomInputArea />
        </div>
      </CardContent>
    </Card>
  );
};
