import { Feature, Sample, Token } from "@/types/feature";
import { SuperToken } from "./token";
import { mergeUint8Arrays } from "@/utils/array";
import { useState } from "react";
import { AppPagination } from "../ui/pagination";
import { Accordion, AccordionTrigger, AccordionContent, AccordionItem } from "../ui/accordion";

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
          key={i}
          sample={sample}
          sampleName={`Sample ${(page - 1) * 10 + i + 1}`}
          maxFeatureAct={feature.maxFeatureAct}
        />
      ))}
      <AppPagination page={page} setPage={setPage} maxPage={maxPage} />
    </div>
  );
};

export type FeatureActivationSampleProps = {
  sample: Sample;
  sampleName: string;
  maxFeatureAct: number;
};

export const FeatureActivationSample = ({ sample, sampleName, maxFeatureAct }: FeatureActivationSampleProps) => {
  const sampleMaxFeatureAct = Math.max(...sample.featureActs);

  const decoder = new TextDecoder("utf-8", { fatal: true });

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

  const tokenGroupPositions = tokenGroups.reduce<number[]>(
    (acc, tokenGroup) => {
      const tokenCount = tokenGroup.length;
      return [...acc, acc[acc.length - 1] + tokenCount];
    },
    [0]
  );

  const tokensList = tokens.map((t) => t.featureAct);
  const startTrigger = Math.max(tokensList.indexOf(Math.max(...tokensList)) - 100, 0);
  const endTrigger = Math.min(tokensList.indexOf(Math.max(...tokensList)) + 10, sample.context.length);
  const tokensTrigger = sample.context.slice(startTrigger, endTrigger).map((token, i) => ({
    token,
    featureAct: sample.featureActs[startTrigger + i],
  }));

  const [tokenGroupsTrigger, __] = tokensTrigger.reduce<[Token[][], Token[]]>(
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

  const tokenGroupPositionsTrigger = tokenGroupsTrigger.reduce<number[]>(
    (acc, tokenGroup) => {
      const tokenCount = tokenGroup.length;
      return [...acc, acc[acc.length - 1] + tokenCount];
    },
    [0]
  );

  return (
    <div>
      <Accordion type="single" collapsible>
        <AccordionItem value={sampleMaxFeatureAct.toString()}>
          <AccordionTrigger>
            <div className="flex justify-start flex-wrap w-full">
              {sampleName && <span className="text-gray-700 font-bold whitespace-pre">{sampleName}: </span>}
              {startTrigger != 0 && <span>...</span>}
              {tokenGroupsTrigger.map((tokens, i) => (
                <div key={i} className="inline-block whitespace-pre">
                  <SuperToken
                    key={`group-${i}`}
                    tokens={tokens}
                    position={tokenGroupPositionsTrigger[i]}
                    maxFeatureAct={maxFeatureAct}
                    sampleMaxFeatureAct={sampleMaxFeatureAct}
                  />
                </div>
              ))}
              {endTrigger != 0 && <span> ...</span>}
            </div>
          </AccordionTrigger>
          <AccordionContent>
            {tokenGroups.map((tokens, i) => (
              <SuperToken
                key={`group-${i}`}
                tokens={tokens}
                position={tokenGroupPositions[i]}
                maxFeatureAct={maxFeatureAct}
                sampleMaxFeatureAct={sampleMaxFeatureAct}
              />
            ))}
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </div>
  );
};
