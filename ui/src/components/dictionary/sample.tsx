import { cn } from "@/lib/utils";
import { DictionarySample, DictionaryToken } from "@/types/dictionary";
import { mergeUint8Arrays, zip } from "@/utils/array";
import { useState } from "react";
import { ColumnDef } from "@tanstack/react-table";
import { DataTable } from "../ui/data-table";
import { getAccentClassname } from "@/utils/style";
import { SimpleSampleArea } from "../app/sample";
import { HoverCard, HoverCardContent } from "../ui/hover-card";
import { HoverCardTrigger } from "@radix-ui/react-hover-card";
import { FeatureLinkWithPreview } from "../app/feature-preview";

export type DictionarySampleAreaProps = {
  sample: DictionarySample;
  sampleName: string;
  dictionaryName: string;
};

export const DictionarySampleArea = ({ sample, sampleName, dictionaryName }: DictionarySampleAreaProps) => {
  // const [selectedTokenGroupIndex, setSelectedTokenGroupIndex] = useState<number | null>(null);
  const [selectedTokenGroupIndices, setSelectedTokenGroupIndices] = useState<number[]>([]);
  const toggleSelectedTokenGroupIndex = (index: number) => {
    setSelectedTokenGroupIndices((prev) =>
      prev.includes(index) ? prev.filter((i) => i !== index) : [...prev, index].sort()
    );
  };

  const decoder = new TextDecoder("utf-8", { fatal: true });

  const tokens = sample.context.map((token, i) => ({
    token,
    featureActs: zip(sample.featureActsIndices[i], sample.featureActs[i], sample.maxFeatureActs[i]).map(
      ([featureActIndex, featureAct, maxFeatureAct]) => ({
        featureActIndex,
        featureAct,
        maxFeatureAct,
      })
    ),
  }));

  const [tokenGroups, _] = tokens.reduce<[DictionaryToken[][], DictionaryToken[]]>(
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

  const selectedTokenGroups = selectedTokenGroupIndices.map((i) => tokenGroups[i]);
  const selectedTokens = selectedTokenGroups.flatMap((tokens) => tokens);
  const columns: ColumnDef<{ featureIndex: number; [key: `token${number}`]: string }, string>[] = [
    {
      accessorKey: "featureIndex",
      header: () => (
        <div>
          <span className="font-bold">Feature</span>
          <span> \ </span>
          <span className="font-bold">Token</span>
        </div>
      ),
      cell: ({ row }) => (
        <div>
          <FeatureLinkWithPreview dictionaryName={dictionaryName} featureIndex={row.getValue("featureIndex")} />
        </div>
      ),
    },
    ...(selectedTokenGroupIndices
      .map((i) => [tokenGroups[i], i] as const)
      .flatMap(([tokens, i]) => tokens.map((token) => [token, i] as const))
      .map(([token, tokenGroupIndex], i) => ({
        accessorKey: `token${i}`,
        header: () => (
          <HoverCard>
            <HoverCardTrigger className="select-none cursor-pointer">
              {token.token.reduce(
                (acc, b) =>
                  b < 32 || b > 126 ? `${acc}\\x${b.toString(16).padStart(2, "0")}` : `${acc}${String.fromCharCode(b)}`,
                ""
              )}
            </HoverCardTrigger>
            <HoverCardContent>
              <SimpleSampleArea
                sample={sample}
                sampleName={sampleName}
                tokenGroupClassName={(_, j) => (j === tokenGroupIndex ? "bg-orange-500" : "")}
              />
            </HoverCardContent>
          </HoverCard>
        ),
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        cell: ({ row }: { row: any }) => {
          const featureAct = row.getValue(`token${i}`);
          return (
            <div className={cn(getAccentClassname(featureAct.featureAct, featureAct.maxFeatureAct, "text"))}>
              {featureAct.featureAct.toFixed(3)}
            </div>
          );
        },
      })) || []),
  ];

  const data = Object.entries(
    selectedTokens
      ?.flatMap((token, i) =>
        token.featureActs.map((featureAct) => ({
          token: token.token,
          tokenIndex: i,
          ...featureAct,
        }))
      )
      .reduce((acc, featureAct) => {
        // Group by featureActIndex
        const key = featureAct.featureActIndex.toString();
        if (acc[key]) {
          acc[key].push(featureAct);
        } else {
          acc[key] = [featureAct];
        }
        return acc;
      }, {} as Record<string, { token: Uint8Array; tokenIndex: number; featureAct: number; maxFeatureAct: number }[]>) ||
      {}
  )
    .sort(
      // Sort by sum of featureAct
      ([_a, a], [_b, b]) =>
        b.reduce((acc, { featureAct }) => acc + featureAct, 0) - a.reduce((acc, { featureAct }) => acc + featureAct, 0)
    )
    .map(([featureIndex, featureActs]) => {
      return {
        featureIndex: parseInt(featureIndex),
        ...selectedTokens?.reduce(
          (acc, _, i) => ({
            ...acc,
            [`token${i}`]: featureActs.find((featureAct) => featureAct.tokenIndex === i) || {
              featureAct: 0,
              maxFeatureAct: 0,
            },
          }),
          {}
        ),
      };
    });

  return (
    <div className="flex flex-col gap-4">
      <SimpleSampleArea
        sample={sample}
        sampleName={sampleName}
        tokenGroupClassName={(_, i) =>
          cn(
            "hover:shadow-lg hover:text-gray-600 cursor-pointer",
            selectedTokenGroupIndices.includes(i) && "bg-orange-500"
          )
        }
        tokenGroupProps={(_, i) => ({
          onClick: () => toggleSelectedTokenGroupIndex(i),
        })}
      />

      {selectedTokens.length > 0 && <DataTable columns={columns} data={data} />}
    </div>
  );
};
