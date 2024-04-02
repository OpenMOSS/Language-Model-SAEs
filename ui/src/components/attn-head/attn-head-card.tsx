import { AttentionHead } from "@/types/attn-head";
import { Card, CardHeader, CardTitle, CardContent } from "../ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../ui/table";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "../ui/hover-card";
import { useAsync } from "react-use";
import { Feature, FeatureSchema } from "@/types/feature";
import { decode } from "@msgpack/msgpack";
import camelcaseKeys from "camelcase-keys";
import { create } from "zustand";

const useFeaturePreviewStore = create<{
  features: Record<string, Feature>;
  addFeature: (feature: Feature) => void;
}>((set) => ({
  features: {},
  addFeature: (feature: Feature) =>
    set((state) => ({
      features: {
        ...state.features,
        [`${feature.dictionaryName}---${feature.featureIndex}`]: feature,
      },
    })),
}));

const FeaturePreview = ({
  dictionaryName,
  featureIndex,
}: {
  dictionaryName: string;
  featureIndex: number;
}) => {
  const featureInStore: Feature | null = useFeaturePreviewStore(
    (state) => state.features[`${dictionaryName}---${featureIndex}`] || null
  );
  const addFeature = useFeaturePreviewStore((state) => state.addFeature);

  const state = useAsync(async () => {
    if (featureInStore) {
      return featureInStore;
    }
    const feature = await fetch(
      `${
        import.meta.env.VITE_BACKEND_URL
      }/dictionaries/${dictionaryName}/features/${featureIndex}`,
      {
        method: "GET",
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
          stopPaths: ["sample_groups.samples.context"],
        })
      )
      .then((res) => FeatureSchema.parse(res));
    addFeature(feature);
    return feature;
  });

  return (
    <div>
      {state.loading && <p>Loading...</p>}
      {state.error && <p>Error: {state.error.message}</p>}
      {state.value && (
        <div className="grid gap-4 whitespace-pre-wrap grid-cols-[auto,1fr]">
          <div className="text-sm font-bold">Feature:</div>
          <div className="text-sm">#{featureIndex}</div>
          <div className="text-sm font-bold">Interpretation:</div>
          <div className="text-sm">
            {state.value.interpretation?.text || "N/A"}
          </div>
        </div>
      )}
    </div>
  );
};

const FeatureLinkWithPreview = ({
  dictionaryName,
  featureIndex,
}: {
  dictionaryName: string;
  featureIndex: number;
}) => {
  return (
    <HoverCard>
      <HoverCardTrigger>
        <a
          href={`/features?dictionary=${encodeURIComponent(
            dictionaryName
          )}&featureIndex=${featureIndex}`}
        >
          #{featureIndex}
        </a>
      </HoverCardTrigger>
      <HoverCardContent className="w-[500px] max-h-[300px] overflow-y-auto">
        <FeaturePreview
          dictionaryName={dictionaryName}
          featureIndex={featureIndex}
        />
      </HoverCardContent>
    </HoverCard>
  );
};

export const AttentionHeadCard = ({
  attnHead,
}: {
  attnHead: AttentionHead;
}) => {
  return (
    <Card className="container">
      <CardHeader>
        <CardTitle className="text-xl">
          Attention Head {attnHead.layer}.{attnHead.head}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-12">
          {attnHead.attnScores.map((attnScoreGroup, idx) => (
            <div key={idx} className="flex flex-col gap-4">
              <p className="font-bold">
                {attnScoreGroup.dictionary1Name} {" -> "}
                {attnScoreGroup.dictionary2Name}
              </p>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Feature</TableHead>
                    <TableHead>Feature Attended</TableHead>
                    <TableHead>Attention Score</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {attnScoreGroup.topAttnScores.map((attnScore, idx) => (
                    <TableRow key={idx}>
                      <TableCell>
                        <FeatureLinkWithPreview
                          dictionaryName={attnScoreGroup.dictionary1Name}
                          featureIndex={attnScore.feature1Index}
                        />
                      </TableCell>
                      <TableCell>
                        <FeatureLinkWithPreview
                          dictionaryName={attnScoreGroup.dictionary2Name}
                          featureIndex={attnScore.feature2Index}
                        />
                      </TableCell>
                      <TableCell>{attnScore.attnScore.toFixed(3)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};
