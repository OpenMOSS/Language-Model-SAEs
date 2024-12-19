import { Feature, FeatureSampleCompact, ImageTokenOrigin, TextTokenOrigin } from "@/types/feature";
import { useState } from "react";
import { AppPagination } from "../ui/pagination";
import { getAccentClassname } from "@/utils/style";
import { cn } from "@/lib/utils";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "../ui/hover-card";

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
  featureAct: number;
  maxFeatureAct: number;
  origin: TextTokenOrigin | ImageTokenOrigin;
};

export const TokenInfo = ({ featureAct, maxFeatureAct, origin }: TokenInfoProps) => {
  return (
    <div className="grid grid-cols-2 gap-2">
      {origin.key === "text" ? (
        <>
          <div className="text-sm font-bold">Text Range:</div>
          <div className="text-sm">{origin.range.join(" - ")}</div>
        </>
      ) : (
        <>
          <div className="text-sm font-bold">Image Region:</div>
          <div className="text-sm">{origin.rect.map((n) => n.toFixed(3)).join(", ")}</div>
        </>
      )}
      <div className="text-sm font-bold">Activation:</div>
      <div className={cn("text-sm", getAccentClassname(featureAct, maxFeatureAct, "text"))}>
        {featureAct.toFixed(3)}
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
  // Process text highlights
  const textHighlights = sample.origins
    .map((origin, index) => ({
      origin,
      featureAct: sample.featureActs[index],
    }))
    .filter((item): item is { origin: TextTokenOrigin; featureAct: number } => item.origin?.key === "text");

  // Create segments for overlapping highlights
  const segments: { start: number; end: number; highlights: typeof textHighlights }[] = [];
  if (sample.text && textHighlights.length > 0) {
    // Get all unique positions
    const positions = new Set<number>();
    textHighlights.forEach((h) => {
      positions.add(h.origin.range[0]);
      positions.add(h.origin.range[1]);
    });
    const sortedPositions = Array.from(positions).sort((a, b) => a - b);

    // Create segments between each pair of positions
    for (let i = 0; i < sortedPositions.length - 1; i++) {
      const start = sortedPositions[i];
      const end = sortedPositions[i + 1];
      const activeHighlights = textHighlights.filter((h) => h.origin.range[0] <= start && h.origin.range[1] >= end);
      if (activeHighlights.length > 0) {
        segments.push({ start, end, highlights: activeHighlights });
      }
    }
  }
  
  // Process image highlights
  const imageHighlights = sample.origins
    .map((origin, index) => ({
      origin,
      featureAct: sample.featureActs[index],
    }))
    .filter((item): item is { origin: ImageTokenOrigin; featureAct: number } => item.origin?.key === "image");

  return (
    <div className="border rounded p-4 w-full flex flex-col gap-2">
      <h3 className="font-bold mb-2">{sampleName}</h3>

      <div className="flex gap-4 w-full justify-between">
        {/* Text display with highlights */}
        {sample.text && (
          <div className="relative whitespace-pre-wrap mb-4">
            {segments.map((segment, index) => {
              const beforeText = index === 0 ? sample.text!.slice(0, segment.start) : "";
              const segmentText = sample.text!.slice(segment.start, segment.end);
              // Get the highest activation for this segment
              const maxSegmentAct = Math.max(...segment.highlights.map((h) => h.featureAct));

              return (
                <span key={index}>
                  {beforeText}
                  <HoverCard>
                    <HoverCardTrigger>
                      <span
                        className={cn("relative cursor-help", getAccentClassname(maxSegmentAct, maxFeatureAct, "bg"))}
                      >
                        {segmentText}
                      </span>
                    </HoverCardTrigger>
                    <HoverCardContent>
                      <div className="flex flex-col gap-2">
                        {segment.highlights.map((highlight, i) => (
                          <TokenInfo
                            key={i}
                            featureAct={highlight.featureAct}
                            maxFeatureAct={maxFeatureAct}
                            origin={highlight.origin}
                          />
                        ))}
                      </div>
                    </HoverCardContent>
                  </HoverCard>
                  {index === segments.length - 1 && sample.text!.slice(segment.end)}
                </span>
              );
            })}

            <div className="flex flex-col gap-2">
              <div className="text-sm font-bold">Image Highlights:</div>
              {imageHighlights.filter((v) => v.featureAct > 0.01).map((highlight, index) => {
                return <div key={index}>{highlight.origin.rect.join(", ")}</div>;
              })}
            </div>
          </div>
        )}

        {/* Images with highlight overlays */}
        {sample.images && (
          <div className="flex flex-wrap gap-2">
            {sample.images.map((imageUrl, imgIndex) => {
              // Concat imageUrl with baseUrl
              const fullImageUrl = `${import.meta.env.VITE_BACKEND_URL}${imageUrl}`;
              return (
                <div key={imgIndex} className="relative">
                  <img src={fullImageUrl} alt="" className="max-w-[200px] h-auto" />
                  {imageHighlights
                    .filter((highlight) => highlight.origin.imageIndex === imgIndex)
                    .map((highlight, index) => {
                      const [x1, y1, x2, y2] = highlight.origin.rect;
                      const left = x1;
                      const top = y1;
                      const width = x2 - x1;
                      const height = y2 - y1;
                      return (
                        <HoverCard key={index}>
                          <HoverCardTrigger>
                            <div
                              className={cn(
                                "absolute cursor-help",
                                getAccentClassname(highlight.featureAct, maxFeatureAct, "bg")
                              )}
                              style={{
                                left: `${left * 100}%`,
                                top: `${top * 100}%`,
                                width: `${width * 100}%`,
                                height: `${height * 100}%`,
                                opacity: 0.3,
                              }}
                            />
                          </HoverCardTrigger>
                          <HoverCardContent>
                            <TokenInfo
                              featureAct={highlight.featureAct}
                              maxFeatureAct={maxFeatureAct}
                              origin={highlight.origin}
                            />
                          </HoverCardContent>
                        </HoverCard>
                      );
                    })}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
