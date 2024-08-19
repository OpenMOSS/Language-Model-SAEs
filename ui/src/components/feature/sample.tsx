import { Feature, FeatureSampleCompact } from "@/types/feature";
import { useState } from "react";
import { AppPagination } from "../ui/pagination";
import { countTokenGroupPositions, groupToken, hex } from "@/utils/token";
import { zip } from "@/utils/array";
import { getAccentClassname } from "@/utils/style";
import { cn } from "@/lib/utils";
import { Sample } from "../app/sample";

export const FeatureSampleGroup = ({
  feature,
  sampleGroup,
}: {
  feature: Feature;
  sampleGroup: Feature["sampleGroups"][0];
}) => {
  const [page, setPage] = useState<number>(1);
  const maxPage = Math.ceil(sampleGroup.samples.length / 10);

  return (
    <div className="flex flex-col gap-4 mt-4">
      <p className="font-bold">Max Activation: {Math.max(...sampleGroup.samples[0].featureActs).toFixed(3)}</p>
      {sampleGroup.samples.slice((page - 1) * 10, page * 10).map((sample, i) => (
        <FeatureActivationSample
          key={(page - 1) * 10 + i}
          sample={sample}
          sampleName={`Sample ${(page - 1) * 10 + i + 1}`}
          maxFeatureAct={feature.maxFeatureAct}
        />
      ))}
      <AppPagination page={page} setPage={setPage} maxPage={maxPage} />
    </div>
  );
};

export type TokenInfoProps = {
  token: { token: Uint8Array; featureAct: number };
  maxFeatureAct: number;
  position: number;
};

export const TokenInfo = ({ token, maxFeatureAct, position }: TokenInfoProps) => {
  return (
    <div className="grid grid-cols-2 gap-2">
      <div className="text-sm font-bold">Token:</div>
      <div className="text-sm underline whitespace-pre-wrap">{hex(token)}</div>
      <div className="text-sm font-bold">Position:</div>
      <div className="text-sm">{position}</div>
      <div className="text-sm font-bold">Activation:</div>
      <div className={cn("text-sm", getAccentClassname(token.featureAct, maxFeatureAct, "text"))}>
        {token.featureAct.toFixed(3)}
      </div>
    </div>
  );
};

export type FeatureActivationSampleProps = {
  sample: FeatureSampleCompact;
  sampleName: string;
  maxFeatureAct: number;
};

export const FeatureActivationSample = ({ sample, sampleName, maxFeatureAct }: FeatureActivationSampleProps) => {
  const sampleMaxFeatureAct = Math.max(...sample.featureActs);

  const tokens = zip(sample.context, sample.featureActs).map(([token, featureAct]) => ({
    token,
    featureAct,
  }));

  const tokenGroups = groupToken(tokens);
  const tokenGroupPositions = countTokenGroupPositions(tokenGroups);

  const featureActs = tokenGroups.map((group) => Math.max(...group.map((token) => token.featureAct)));
  const start = Math.max(featureActs.findIndex((act) => act === sampleMaxFeatureAct) - 60, 0);

  return (
    <Sample
      tokenGroups={tokenGroups}
      sampleName={sampleName}
      tokenInfoContent={(_, i) => (token, j) =>
        <TokenInfo token={token} maxFeatureAct={maxFeatureAct} position={tokenGroupPositions[i] + j} />}
      tokenGroupClassName={(tokenGroup) => {
        const tokenGroupMaxFeatureAct = Math.max(...tokenGroup.map((t) => t.featureAct));
        return cn(
          tokenGroupMaxFeatureAct > 0 && "hover:underline cursor-pointer",
          sampleMaxFeatureAct > 0 && tokenGroupMaxFeatureAct == sampleMaxFeatureAct && "font-bold",
          getAccentClassname(tokenGroupMaxFeatureAct, maxFeatureAct, "bg")
        );
      }}
      foldedStart={start}
    />
  );
};
