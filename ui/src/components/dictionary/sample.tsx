import { cn } from "@/lib/utils";
import { DictionarySampleCompact } from "@/types/dictionary";
import { zip } from "@/utils/array";
import { useState } from "react";
import { ColumnDef } from "@tanstack/react-table";
import { DataTable } from "../ui/data-table";
import { getAccentClassname } from "@/utils/style";
import { Sample } from "../app/sample";
import { HoverCard, HoverCardContent } from "../ui/hover-card";
import { HoverCardTrigger } from "@radix-ui/react-hover-card";
import { FeatureLinkWithPreview } from "../app/feature-preview";
import { Trash2 } from "lucide-react";
import { countTokenGroupPositions, groupToken, hex } from "@/utils/token";

export type DictionarySampleProps = {
  samples: DictionarySampleCompact[];
  onSamplesChange?: (samples: DictionarySampleCompact[]) => void;
  dictionaryName: string;
};

export const DictionarySample = ({ samples, onSamplesChange, dictionaryName }: DictionarySampleProps) => {
  const [selectedTokenGroupIndices, setSelectedTokenGroupIndices] = useState<[number, number][]>([]);
  const toggleSelectedTokenGroupIndex = (sampleIndex: number, tokenGroupIndex: number) => {
    setSelectedTokenGroupIndices((prev) =>
      prev.some(([s, t]) => s === sampleIndex && t === tokenGroupIndex)
        ? prev.filter(([s, t]) => s !== sampleIndex || t !== tokenGroupIndex)
        : ([...prev, [sampleIndex, tokenGroupIndex]] as [number, number][]).sort(
            ([s1, t1], [s2, t2]) => s1 - s2 || t1 - t2
          )
    );
  };

  const tokens = samples.map((sample) =>
    zip(sample.context, sample.featureActsIndices, sample.featureActs, sample.maxFeatureActs).map(
      ([token, featureActsIndices, featureActs, maxFeatureActs]) => ({
        token,
        featureActs: zip(featureActsIndices, featureActs, maxFeatureActs).map(
          ([featureActIndex, featureAct, maxFeatureAct]) => ({
            featureActIndex,
            featureAct,
            maxFeatureAct,
          })
        ),
      })
    )
  );

  const tokenGroups = tokens.map(groupToken);

  const tokenGroupPositions = tokenGroups.map(countTokenGroupPositions);

  const selectedTokenGroups = selectedTokenGroupIndices.map(([s, t]) => tokenGroups[s][t]);
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
      .map(([s, t]) => [tokenGroups[s][t], [s, t]] as const)
      .flatMap(([tokens, [s, t]]) => tokens.map((token, inGroupIndex) => [token, [s, t, inGroupIndex]] as const))
      .map(([token, [s, t, inGroupIndex]], i) => ({
        accessorKey: `token${i}`,
        header: () => (
          <HoverCard>
            <HoverCardTrigger className="select-none cursor-pointer">{hex(token)}</HoverCardTrigger>
            <HoverCardContent className="w-auto max-w-[800px] gap-4">
              <div>
                <b>Position:</b> {tokenGroupPositions[s][t] + inGroupIndex}
              </div>
              <Sample
                tokenGroups={tokenGroups[s]}
                sampleName={`Sample ${s + 1}`}
                tokenGroupClassName={(_, j) => (j === t ? "bg-orange-500" : "")}
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
      .reduce(
        (acc, featureAct) => {
          // Group by featureActIndex
          const key = featureAct.featureActIndex.toString();
          if (acc[key]) {
            acc[key].push(featureAct);
          } else {
            acc[key] = [featureAct];
          }
          return acc;
        },
        {} as Record<string, { token: Uint8Array; tokenIndex: number; featureAct: number; maxFeatureAct: number }[]>
      ) || {}
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
      {tokenGroups.map((tokenGroups, sampleIndex) => (
        <div className="w-full overflow-x-visible relative" key={sampleIndex}>
          <Sample
            tokenGroups={tokenGroups}
            sampleName={`Sample ${sampleIndex + 1}`}
            tokenGroupClassName={(_, tokenIndex) =>
              cn(
                "hover:shadow-lg hover:text-gray-600 cursor-pointer",
                selectedTokenGroupIndices.some(([s, t]) => s === sampleIndex && t === tokenIndex) && "bg-orange-500"
              )
            }
            tokenGroupProps={(_, i) => ({
              onClick: () => toggleSelectedTokenGroupIndex(sampleIndex, i),
            })}
          />
          <div className="absolute top-0 -left-8 h-full opacity-20 hover:opacity-100 transition-opacity">
            <Trash2
              className="cursor-pointer hover:text-red-500 shrink-0 m-0.5"
              size={20}
              onClick={() => {
                setSelectedTokenGroupIndices((prev) =>
                  prev.filter(([s, _]) => s !== sampleIndex).map(([s, t]) => (s > sampleIndex ? [s - 1, t] : [s, t]))
                );
                onSamplesChange?.(samples.filter((_, i) => i !== sampleIndex));
              }}
            />
          </div>
        </div>
      ))}

      {selectedTokens.length > 0 && <DataTable columns={columns} data={data} />}
    </div>
  );
};
