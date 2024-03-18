import { Feature, Token } from "@/types/feature";
import { SuperToken } from "./token";
import { mergeUint8Arrays } from "@/utils/array";

export type FeatureActivationSampleProps = {
  sample: Feature["samples"][0];
  sampleIndex: number;
  maxFeatureAct: number;
};

export const FeatureActivationSample = ({
  sample,
  sampleIndex,
  maxFeatureAct,
}: FeatureActivationSampleProps) => {
  const sampleMaxFeatureAct = Math.max(...sample.featureActs);

  const decoder = new TextDecoder("utf-8", { fatal: true });

  // Find max feature activation, and pick the 20 tokens surrounding it
  // const maxFeatureActIndex = sample.featureActs.indexOf(sampleMaxFeatureAct);
  const start = Math.max(0);
  const end = Math.min(sample.context.length);
  const tokens = sample.context.slice(start, end).map((token, i) => ({
    token,
    featureAct: sample.featureActs[start + i],
  }));

  const [tokenGroups, _] = tokens.reduce<[Token[][], Token[]]>(
    ([groups, currentGroup], token) => {
      const newGroup = [...currentGroup, token];
      try {
        decoder.decode(mergeUint8Arrays(newGroup.map((t) => t.token)));
        return [[...groups, newGroup], []];
      } catch {
        return [groups, newGroup];
      }
    },
    [[], []]
  );
  return (
    <p className="flex flex-wrap whitespace-pre">
      <span className="text-gray-700 font-bold">
        Sample {sampleIndex + 1}:{" "}
      </span>
      {tokenGroups.map((tokens, i) => (
        <SuperToken
          key={i}
          tokens={tokens}
          maxFeatureAct={maxFeatureAct}
          sampleMaxFeatureAct={sampleMaxFeatureAct}
        />
      ))}
    </p>
  );
};
