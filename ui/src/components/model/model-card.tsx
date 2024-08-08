import { useState } from "react";
import { Button } from "../ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Textarea } from "../ui/textarea";
import { useAsyncFn } from "react-use";
import camelcaseKeys from "camelcase-keys";
import { decode } from "@msgpack/msgpack";
import { ModelGeneration, ModelGenerationSchema } from "@/types/model";
import { Sample } from "../app/sample";
import { cn } from "@/lib/utils";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Label, LabelList, ResponsiveContainer } from "recharts";
import { zip } from "@/utils/array";
import { Input } from "../ui/input";
import { groupToken } from "@/utils/token";

const ModelSample = ({ sample }: { sample: ModelGeneration }) => {
  const [selectedTokenGroupIndices, setSelectedTokenGroupIndices] = useState<number[]>([]);
  const toggleSelectedTokenGroupIndex = (tokenGroupIndex: number) => {
    setSelectedTokenGroupIndices((prev) =>
      prev.includes(tokenGroupIndex) ? prev.filter((t) => t !== tokenGroupIndex) : [...prev, tokenGroupIndex]
    );
  };

  const tokens = zip(sample.context, sample.inputMask, sample.logits, sample.logitsTokens).map(
    ([token, inputMask, logits, logitsTokens]) => ({
      token,
      inputMask,
      logits: zip(logits, logitsTokens).map(([logits, logitsTokens]) => ({
        logits,
        logitsTokens,
      })),
    })
  );
  const tokenGroups = groupToken(tokens);
  const selectedTokenGroups = selectedTokenGroupIndices.map((i) => tokenGroups[i]);
  const selectedTokens = selectedTokenGroups.flatMap((t) => t);

  const data = selectedTokens.map((token) =>
    Object.assign(
      {},
      ...token.logits.map((logits, j) => ({
        [`logits-${j}`]: logits.logits,
        [`logits-token-${j}`]: logits.logitsTokens.reduce(
          (acc, b) =>
            b < 32 || b > 126 ? `${acc}\\x${b.toString(16).padStart(2, "0")}` : `${acc}${String.fromCharCode(b)}`,
          ""
        ),
      })),
      {
        name: token.token.reduce(
          (acc, b) =>
            b < 32 || b > 126 ? `${acc}\\x${b.toString(16).padStart(2, "0")}` : `${acc}${String.fromCharCode(b)}`,
          ""
        ),
      }
    )
  );

  const colors = ["#8884d8", "#82ca9d", "#ffc658", "#ff7300", "#d6d6d6"];

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
        <ResponsiveContainer height={300}>
          <BarChart data={data} margin={{ top: 50, right: 50, left: 50, bottom: 15 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name">
              <Label value="Tokens" offset={0} position="bottom" />
            </XAxis>
            <YAxis label={{ value: "Logits", angle: -90, position: "left", textAnchor: "middle" }} />
            {selectedTokens[0].logits.map((_, i) => (
              <Bar key={i} dataKey={`logits-${i}`} fill={colors[i]}>
                <LabelList dataKey={`logits-token-${i}`} position="top" />
              </Bar>
            ))}
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
};

const ModelCustomInputArea = () => {
  const [customInput, setCustomInput] = useState<string>("");
  const [maxNewTokens, setMaxNewTokens] = useState<number>(128);
  const [topK, setTopK] = useState<number>(50);
  const [topP, setTopP] = useState<number>(0.95);
  const [logitTopK, setLogitTopK] = useState<number>(5);
  const [sample, setSample] = useState<ModelGeneration | null>(null);
  const [state, submit] = useAsyncFn(async () => {
    if (!customInput) {
      alert("Please enter your input.");
      return;
    }
    const sample = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/model/generate?input_text=${encodeURIComponent(
        customInput
      )}&max_new_tokens=${encodeURIComponent(maxNewTokens.toString())}&top_k=${encodeURIComponent(
        topK.toString()
      )}&top_p=${encodeURIComponent(topP.toString())}&return_logits_top_k=${encodeURIComponent(logitTopK.toString())}`,
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
          stopPaths: ["context", "logits_tokens"],
        })
      )
      .then((res) => ModelGenerationSchema.parse(res));
    setSample(sample);
  }, [customInput]);

  return (
    <div className="flex flex-col gap-4">
      <p className="font-bold">Generation</p>
      <div className="container grid grid-cols-4 justify-center items-center gap-4 px-20">
        <span className="font-bold justify-self-end">Max new tokens:</span>
        <Input
          disabled={state.loading}
          className="bg-white"
          type="number"
          value={maxNewTokens.toString()}
          onChange={(e) => setMaxNewTokens(parseInt(e.target.value))}
        />
        <span className="font-bold justify-self-end">Top K:</span>
        <Input
          disabled={state.loading}
          className="bg-white"
          type="number"
          value={topK.toString()}
          onChange={(e) => setTopK(parseInt(e.target.value))}
        />
        <span className="font-bold justify-self-end">Top P:</span>
        <Input
          disabled={state.loading}
          className="bg-white"
          type="number"
          value={topP.toString()}
          onChange={(e) => setTopP(parseFloat(e.target.value))}
        />
        <span className="font-bold justify-self-end">Logit Top K:</span>
        <Input
          disabled={state.loading}
          className="bg-white"
          type="number"
          value={logitTopK.toString()}
          onChange={(e) => setLogitTopK(parseInt(e.target.value))}
        />
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
      {sample && <ModelSample sample={sample} />}
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
