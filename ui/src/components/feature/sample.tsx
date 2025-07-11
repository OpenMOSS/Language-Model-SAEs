import {
  Feature,
  FeatureSampleCompact,
  ImageTokenOrigin,
  TextTokenOrigin,
  MultipleTextTokenOrigin,
} from "@/types/feature";
import { useState, useCallback, useMemo, memo } from "react";
import { AppPagination } from "../ui/pagination";
import { getAccentClassname } from "@/utils/style";
import { cn } from "@/lib/utils";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "../ui/hover-card";
import { Switch } from "../ui/switch";
import { Input } from "../ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";

export const FeatureSampleGroup = ({
  feature,
  sampleGroup,
}: {
  feature: Feature;
  sampleGroup: Feature["sampleGroups"][0];
}) => {
  const [page, setPage] = useState<number>(1);
  const [visibleRange, setVisibleRange] = useState<number>(50);

  const maxPage = useMemo(() => Math.ceil(sampleGroup.samples.length / 10), [sampleGroup.samples.length]);

  const currentSamples = useMemo(
    () => sampleGroup.samples.slice((page - 1) * 10, page * 10),
    [sampleGroup.samples, page]
  );

  const maxActivation = useMemo(
    () => (sampleGroup.samples.length > 0 ? Math.max(...sampleGroup.samples[0].featureActs) : 0),
    [sampleGroup.samples]
  );

  // Handle input change
  const handleRangeChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value);
    if (!isNaN(value) && value > 0) {
      setVisibleRange(value);
    }
  }, []);

  return (
    <div className="flex flex-col gap-4 mt-4">
      <p className="font-bold">Max Activation: {maxActivation.toFixed(3)}</p>

      {/* Feature-level configuration controls */}
      <div className="flex flex-col gap-2 mb-4">
        <div className="flex items-center gap-4">
          <div className="text-sm font-bold min-w-[120px]">Visible Range:</div>
          <Input
            type="number"
            value={visibleRange.toString()}
            onChange={handleRangeChange}
            className="w-[100px]"
            min={1}
          />
          <div className="text-sm text-muted-foreground">segments</div>
        </div>
        <div className="text-xs text-muted-foreground max-w-[600px]">
          A segment is a contiguous range of tokens that share the same activation pattern.
        </div>
      </div>

      {currentSamples.map((sample, i) => (
        <FeatureActivationSample
          key={(page - 1) * 10 + i}
          sample={sample}
          sampleName={`Sample ${(page - 1) * 10 + i + 1}`}
          maxFeatureAct={feature.maxFeatureAct}
          visibleRange={visibleRange}
        />
      ))}
      <AppPagination page={page} setPage={setPage} maxPage={maxPage} />
    </div>
  );
};

export type TokenInfoProps = {
  featureAct: number;
  maxFeatureAct: number;
  origin: TextTokenOrigin | ImageTokenOrigin | MultipleTextTokenOrigin;
  currentField: string;
  allTextFields: { key: string; label: string; text: string | null | undefined }[];
};

export const TokenInfo = ({ featureAct, maxFeatureAct, origin, currentField, allTextFields }: TokenInfoProps) => {
  // Helper function to get text content for a field
  const getTextForField = (fieldKey: string): string => {
    const field = allTextFields.find((f) => f.key === fieldKey);
    return field?.text || "";
  };

  // For multiple_text origins, we assume they apply to the main "text" field
  // but have different segment ranges for different parts
  const baseText = getTextForField("text");

  return (
    <div className="grid grid-cols-2 gap-2">
      {origin.type === "text" ? (
        <>
          <div className="text-sm font-bold">Text Range:</div>
          <div className="text-sm">{origin.range.join(" - ")}</div>
          {baseText && (
            <>
              <div className="text-sm font-bold">Text:</div>
              <div className="text-sm text-muted-foreground max-w-[300px] overflow-hidden text-ellipsis">
                "
                {baseText
                  .slice(origin.range[0], origin.range[1])
                  .replaceAll("\n", "↵")
                  .replaceAll("\t", "→")
                  .replaceAll("<|mdm_mask|>", "[M]")}
                "
              </div>
            </>
          )}
        </>
      ) : origin.type === "multiple_text" ? (
        <>
          <div className="text-sm font-bold">Current Field:</div>
          <div className="text-sm">{currentField}</div>
          <div className="text-sm font-bold">Current Range:</div>
          <div className="text-sm">{origin.range[currentField] ? origin.range[currentField].join(" - ") : "N/A"}</div>
          <div className="text-sm font-bold">All Field Segments:</div>
          <div className="text-sm space-y-1">
            {Object.entries(origin.range).map(([field, range]) => {
              const fieldText = getTextForField(field); // multiple_text origins reference the main text
              return (
                <div key={field} className="mb-1">
                  <span className="font-medium text-blue-600">{field}:</span>
                  <br />
                  <span className="text-muted-foreground text-xs">
                    [{range.join(" - ")}] "
                    {fieldText
                      ? fieldText
                          .slice(range[0], range[1])
                          .replaceAll("\n", "↵")
                          .replaceAll("\t", "→")
                          .replaceAll("<|mdm_mask|>", "[M]")
                      : ""}
                    "
                  </span>
                </div>
              );
            })}
          </div>
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
  visibleRange?: number;
};

export const FeatureActivationSample = memo(
  ({ sample, sampleName, maxFeatureAct, visibleRange }: FeatureActivationSampleProps) => {
    // Add state for selected text field
    const [selectedField, setSelectedField] = useState<string>("text");

    // Get available text fields
    const availableFields = useMemo(() => {
      const fields: { key: string; label: string; text: string | null | undefined }[] = [];
      if (sample.text) fields.push({ key: "text", label: "Text", text: sample.text });
      if (sample.predictedText)
        fields.push({ key: "predictedText", label: "Predicted Text", text: sample.predictedText });
      if (sample.originalText) fields.push({ key: "originalText", label: "Original Text", text: sample.originalText });
      return fields;
    }, [sample.text, sample.predictedText, sample.originalText]);

    // Get current text content
    const currentText = useMemo(() => {
      const field = availableFields.find((f) => f.key === selectedField);
      return field?.text || null;
    }, [availableFields, selectedField]);

    // Memoize text highlights processing - filter by selected text field
    const currentFieldHighlights = useMemo(
      () =>
        sample.origins
          .map((origin, index) => ({
            origin,
            featureAct: sample.featureActs[index],
          }))
          .filter((item): item is { origin: TextTokenOrigin | MultipleTextTokenOrigin; featureAct: number } => {
            if (item.origin?.type === "text") {
              // For simple text origins, only show if we're viewing the main text field
              return selectedField === "text";
            } else if (item.origin?.type === "multiple_text") {
              // For multiple text origins, check if the selected text field has a corresponding range
              return selectedField in item.origin.range;
            }
            return false;
          }),
      [sample.origins, sample.featureActs, selectedField]
    );

    // Memoize max activation highlight for current field
    const maxActivationHighlight = useMemo(
      () =>
        currentFieldHighlights.length > 0
          ? currentFieldHighlights.reduce(
              (max, current) => (current.featureAct > max.featureAct ? current : max),
              currentFieldHighlights[0]
            )
          : null,
      [currentFieldHighlights]
    );

    // Memoize segments calculation
    const segments = useMemo(() => {
      const currentSegments: {
        start: number;
        end: number;
        highlights: typeof currentFieldHighlights;
        maxSegmentAct: number;
        index: number;
      }[] = [];

      if (currentText && currentFieldHighlights.length > 0 && maxActivationHighlight) {
        // Get all unique positions for available segment fields
        const positions = new Set<number>();
        currentFieldHighlights.forEach((h) => {
          if (h.origin.type === "text") {
            positions.add(h.origin.range[0]);
            positions.add(h.origin.range[1]);
          } else {
            positions.add(h.origin.range[selectedField][0]);
            positions.add(h.origin.range[selectedField][1]);
          }
        });

        // remove duplicates for each field
        const sortedPositions = Array.from(positions).sort((a, b) => a - b);

        // Create segments between each pair of positions
        for (let i = 0; i < sortedPositions.length - 1; i++) {
          const start = sortedPositions[i];
          const end = sortedPositions[i + 1];

          // Only include highlights that are relevant to this specific field
          const activeHighlights = currentFieldHighlights.filter((h) =>
            h.origin.type === "text"
              ? h.origin.range[0] <= start && h.origin.range[1] >= end
              : h.origin.range[selectedField][0] <= start && h.origin.range[selectedField][1] >= end
          );

          if (activeHighlights.length > 0) {
            const maxSegmentAct = Math.max(...activeHighlights.map((h) => h.featureAct));
            currentSegments.push({ start, end, highlights: activeHighlights, maxSegmentAct, index: i });
          }
        }
      }
      return currentSegments;
    }, [currentText, currentFieldHighlights, maxActivationHighlight, selectedField]);

    const maxActivationSegmentIndex = useMemo(
      () =>
        segments.findIndex((segment) =>
          segment.highlights.some((highlight) => highlight.featureAct === maxActivationHighlight?.featureAct)
        ),
      [segments, maxActivationHighlight]
    );

    // Memoize visible segments for selected field
    const visibleSegments = useMemo(
      () =>
        segments.filter((segment) => {
          if (!visibleRange) return true;
          if (maxActivationSegmentIndex === -1) return true;
          return Math.abs(segment.index - maxActivationSegmentIndex) <= visibleRange;
        }),
      [segments, visibleRange, maxActivationSegmentIndex]
    );

    // Memoize image highlights processing
    const imageHighlights = useMemo(
      () =>
        sample.origins
          .map((origin, index) => ({
            origin,
            featureAct: sample.featureActs[index],
          }))
          .filter((item): item is { origin: ImageTokenOrigin; featureAct: number } => item.origin?.type === "image"),
      [sample.origins, sample.featureActs]
    );

    const [showImageHighlights, setShowImageHighlights] = useState(true);
    const [showImageGrid, setShowImageGrid] = useState(false);

    // Determine if we have any images to display
    const hasImages = sample.images && sample.images.length > 0;

    return (
      <div className="border rounded p-4 w-full flex flex-col gap-2">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-bold">{sampleName}</h3>
          {sample.maskRatio !== null && sample.maskRatio !== undefined && (
            <div className="text-sm text-muted-foreground">
              Mask Ratio: <span className="font-mono">{sample.maskRatio.toFixed(2)}</span>
            </div>
          )}
        </div>

        <div className="flex flex-col gap-4 w-full">
          {/* Add text field selector */}
          {availableFields.length > 1 && (
            <div className="flex items-center gap-4">
              <div className="text-sm font-bold min-w-[120px]">Text Source:</div>
              <Select
                value={selectedField}
                onValueChange={(value) => {
                  setSelectedField(value);
                  console.log("selectedField", value);
                }}
              >
                <SelectTrigger className="w-[200px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {availableFields.map((field) => (
                    <SelectItem key={field.key} value={field.key}>
                      {field.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          <div className="flex gap-4 w-full justify-between">
            {/* Text display with highlights */}
            {currentText && (
              <div className="flex flex-col gap-2">
                <div className="relative whitespace-pre-wrap mb-4">
                  {visibleSegments.map((segment, index) => {
                    const segmentText = currentText!.slice(segment.start, segment.end);

                    return (
                      <span key={index}>
                        {segment.maxSegmentAct > 0 ? (
                          <HoverCard>
                            <HoverCardTrigger>
                              <span
                                className={cn(
                                  "relative cursor-help",
                                  getAccentClassname(segment.maxSegmentAct, maxFeatureAct, "bg")
                                )}
                              >
                                {segmentText
                                  .replaceAll("\n", "↵")
                                  .replaceAll("\t", "→")
                                  .replaceAll("<|mdm_mask|>", "[M]")}
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
                                    currentField={selectedField}
                                    allTextFields={availableFields}
                                  />
                                ))}
                              </div>
                            </HoverCardContent>
                          </HoverCard>
                        ) : (
                          <span>
                            {segmentText.replaceAll("\n", "↵").replaceAll("\t", "→").replaceAll("<|mdm_mask|>", "[M]")}
                          </span>
                        )}
                      </span>
                    );
                  })}
                </div>
                {hasImages && (
                  <div className="flex flex-col gap-2">
                    <div className="text-sm font-bold">Top 10 Image Highlights:</div>
                    {imageHighlights
                      .filter((v) => v.featureAct > 0.01)
                      .sort((a, b) => b.featureAct - a.featureAct)
                      .slice(0, 10)
                      .map((highlight, index) => {
                        return (
                          <div key={index}>
                            [{highlight.origin.rect.map((n) => (n * 100).toFixed(1) + "%").join(", ")}]
                          </div>
                        );
                      })}
                  </div>
                )}
              </div>
            )}

            {/* Images with highlight overlays */}
            {hasImages && (
              <div className="flex flex-col gap-2">
                <div className="flex gap-4">
                  <div className="flex gap-2">
                    <div className="text-sm font-bold">Show Image Highlights</div>
                    <Switch checked={showImageHighlights} onCheckedChange={setShowImageHighlights} />
                  </div>
                  <div className="flex gap-2">
                    <div className="text-sm font-bold">Show Image Grid</div>
                    <Switch checked={showImageGrid} onCheckedChange={setShowImageGrid} />
                  </div>
                </div>
                {hasImages &&
                  sample.images!.map((imageUrl, imgIndex) => {
                    // Concat imageUrl with baseUrl
                    const fullImageUrl = `${import.meta.env.VITE_BACKEND_URL}${imageUrl}`;
                    return (
                      <div key={imgIndex} className="relative">
                        {(showImageHighlights || showImageGrid) &&
                          imageHighlights
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
                                        "absolute cursor-help bg-opacity-30",
                                        showImageHighlights &&
                                          getAccentClassname(highlight.featureAct, maxFeatureAct, "bg"),
                                        showImageGrid && "border-[1px] border-gray-500",
                                        showImageHighlights && highlight.featureAct > 0 && "border-2 border-orange-500"
                                      )}
                                      style={{
                                        left: `${left * 100}%`,
                                        top: `${top * 100}%`,
                                        width: `${width * 100}%`,
                                        height: `${height * 100}%`,
                                      }}
                                    />
                                  </HoverCardTrigger>
                                  <HoverCardContent>
                                    <TokenInfo
                                      featureAct={highlight.featureAct}
                                      maxFeatureAct={maxFeatureAct}
                                      origin={highlight.origin}
                                      currentField={selectedField}
                                      allTextFields={availableFields}
                                    />
                                  </HoverCardContent>
                                </HoverCard>
                              );
                            })}
                        <img src={fullImageUrl} alt="" className="max-w-[500px] h-auto" />
                      </div>
                    );
                  })}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }
);
