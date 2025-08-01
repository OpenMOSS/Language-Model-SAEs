import { Feature, FeatureSampleCompact, ImageTokenOrigin, TextTokenOrigin } from "@/types/feature";
import { useState, useCallback, useMemo, memo } from "react";
import { AppPagination } from "../ui/pagination";
import { getAccentClassname } from "@/utils/style";
import { cn } from "@/lib/utils";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "../ui/hover-card";
import { Switch } from "../ui/switch";
import { Input } from "../ui/input";
import { getZPatternForToken, findHighestActivatingToken } from "@/utils/token";

// Helper function to get activation value for a given index from COO format
const getActivationValue = (indices: number[], values: number[], targetIndex: number): number => {
  const indexPosition = indices.indexOf(targetIndex);
  return indexPosition !== -1 ? values[indexPosition] : 0;
};

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
  
  const currentSamples = useMemo(() => 
    sampleGroup.samples.slice((page - 1) * 10, page * 10),
    [sampleGroup.samples, page]
  );

  const maxActivation = useMemo(() => 
    sampleGroup.samples.length > 0 ? Math.max(...sampleGroup.samples[0].featureActsValues.flat()) : 0,
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

// New component for displaying z pattern information
export const ZPatternInfo = ({ 
  contributingTokens, 
  contributions, 
  sample 
}: { 
  contributingTokens: number[]; 
  contributions: number[]; 
  sample: FeatureSampleCompact;
}) => {
  if (contributingTokens.length === 0) {
    return <div className="text-sm text-muted-foreground">No z pattern data available</div>;
  }

  // Calculate total contribution
  const totalContribution = contributions.reduce((sum, contribution) => sum + contribution, 0);

  // Helper function to get text content for a token
  const getTokenText = (tokenIndex: number): string => {
    const origin = sample.origins[tokenIndex];
    if (origin?.key === "text" && sample.text) {
      const [start, end] = origin.range;
      return sample.text.slice(start, end);
    }
    return "";
  };

  return (
    <div className="flex flex-col gap-2 border-t pt-2">
      <div className="flex items-center gap-2">
        <div className="text-sm font-bold text-green-700">Z Pattern Contributions:</div>
        <div className="text-xs text-muted-foreground">
          Total: {totalContribution.toFixed(3)}
        </div>
      </div>
      <div className="flex flex-col gap-1 max-h-32 overflow-y-auto">
        {contributingTokens.map((tokenIndex, i) => {
          const origin = sample.origins[tokenIndex];
          const contribution = contributions[i];
          const percentage = (contribution / totalContribution * 100).toFixed(1);
          const tokenText = getTokenText(tokenIndex);
          
          return (
            <div key={i} className="flex flex-col gap-1 p-2 bg-gray-50 rounded">
              <div className="flex items-center justify-between gap-2 text-sm">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs">Token {tokenIndex}:</span>
                  <span className="text-green-600 font-bold">{contribution.toFixed(3)}</span>
                  <span className="text-xs text-muted-foreground">({percentage}%)</span>
                </div>
                {origin?.key === "text" && (
                  <span className="text-xs text-muted-foreground">
                    range: {origin.range.join("-")}
                  </span>
                )}
              </div>
              {tokenText && (
                <div className="text-xs text-muted-foreground font-mono bg-white px-1 py-0.5 rounded">
                  "{tokenText.replaceAll("\n", "↵").replaceAll("\t", "→")}"
                </div>
              )}
            </div>
          );
        })}
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

export const FeatureActivationSample = memo(({
  sample,
  sampleName,
  maxFeatureAct,
  visibleRange,
}: FeatureActivationSampleProps) => {
  // State to track which token is being hovered
  const [hoveredTokenIndex, setHoveredTokenIndex] = useState<number | null>(null);

  // Memoize text highlights processing
  const textHighlights = useMemo(() => 
    sample.origins
      .map((origin, index) => ({
        origin,
        featureAct: getActivationValue(sample.featureActsIndices, sample.featureActsValues, index),
      }))
      .filter((item): item is { origin: TextTokenOrigin; featureAct: number } => item.origin?.key === "text"),
    [sample.origins, sample.featureActsIndices, sample.featureActsValues]
  );

  // Memoize max activation highlight
  const maxActivationHighlight = useMemo(() =>
    textHighlights.length > 0
      ? textHighlights.reduce(
          (max, current) => (current.featureAct > max.featureAct ? current : max),
          textHighlights[0]
        )
      : null,
    [textHighlights]
  );

  // Find the highest activating token for z pattern highlighting
  const highestActivatingToken = useMemo(() => 
    findHighestActivatingToken(sample.featureActsIndices, sample.featureActsValues),
    [sample.featureActsIndices, sample.featureActsValues]
  );

  // Get z pattern for hovered token
  const hoveredZPattern = useMemo(() => {
    if (hoveredTokenIndex === null) return null;
    return getZPatternForToken(
      sample.zPatternIndices,
      sample.zPatternValues,
      hoveredTokenIndex
    );
  }, [hoveredTokenIndex, sample.zPatternIndices, sample.zPatternValues]);

  // Memoize segments calculation
  const segments = useMemo(() => {
    const segmentList: {
      start: number;
      end: number;
      highlights: typeof textHighlights;
      maxSegmentAct: number;
      index: number;
    }[] = [];
    
    if (sample.text && textHighlights.length > 0 && maxActivationHighlight) {
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
          const maxSegmentAct = Math.max(...activeHighlights.map((h) => h.featureAct));
          segmentList.push({ start, end, highlights: activeHighlights, maxSegmentAct, index: i });
        }
      }
    }
    return segmentList;
  }, [sample.text, textHighlights, maxActivationHighlight]);

  // Memoize max activation segment index
  const maxActivationSegmentIndex = useMemo(() => 
    segments.findIndex((segment) =>
      segment.highlights.some((highlight) => highlight.featureAct === maxActivationHighlight?.featureAct)
    ),
    [segments, maxActivationHighlight]
  );

  // Memoize visible segments
  const visibleSegments = useMemo(() => 
    segments.filter((segment) => {
      if (!visibleRange) return true;
      if (maxActivationSegmentIndex === -1) return true;
      return Math.abs(segment.index - maxActivationSegmentIndex) <= visibleRange;
    }),
    [segments, visibleRange, maxActivationSegmentIndex]
  );

  // Memoize image highlights processing
  const imageHighlights = useMemo(() => 
    sample.origins
      .map((origin, index) => ({
        origin,
        featureAct: getActivationValue(sample.featureActsIndices, sample.featureActsValues, index),
      }))
      .filter((item): item is { origin: ImageTokenOrigin; featureAct: number } => item.origin?.key === "image"),
    [sample.origins, sample.featureActsIndices, sample.featureActsValues]
  );

  const [showImageHighlights, setShowImageHighlights] = useState(true);
  const [showImageGrid, setShowImageGrid] = useState(false);

  // Determine if we have any images to display
  const hasImages = sample.images && sample.images.length > 0;

  // Helper function to determine if a segment should be highlighted
  const getSegmentHighlightClass = (segment: typeof visibleSegments[0]) => {
    if (hoveredTokenIndex !== null) {
      console.log('getSegmentHighlightClass - hoveredTokenIndex:', hoveredTokenIndex);
      console.log('getSegmentHighlightClass - hoveredZPattern:', hoveredZPattern);
      
      // Check if this segment contains the hovered token
      const containsHoveredToken = segment.highlights.some(highlight => {
        // Check if this highlight's origin corresponds to the hovered token index
        const hoveredOrigin = sample.origins[hoveredTokenIndex];
        console.log('Checking hovered token - hoveredTokenIndex:', hoveredTokenIndex, 'hoveredOrigin:', hoveredOrigin, 'highlight.origin:', highlight.origin);
        return hoveredOrigin === highlight.origin;
      });
      
      console.log('Contains hovered token:', containsHoveredToken);
      
      if (containsHoveredToken) {
        console.log('Returning orange highlight for hovered token');
        return "bg-orange-500 text-white"; // Keep orange for the hovered token
      }
        
        // Check if this segment contains contributing tokens
        if (hoveredZPattern) {
          const containsContributingToken = segment.highlights.some(highlight => {
            // Find the token index for this highlight's origin
            const tokenIndex = sample.origins.findIndex(origin => origin === highlight.origin);
            console.log('Checking contributing token - tokenIndex:', tokenIndex, 'highlight.origin:', highlight.origin);
            const isContributing = hoveredZPattern.contributingTokens.includes(tokenIndex);
            console.log('Is contributing:', isContributing);
            return isContributing;
          });
        
        console.log('Contains contributing token:', containsContributingToken);
        
        if (containsContributingToken) {
          console.log('Returning green highlight for contributing token');
          return "bg-green-500 text-white"; // Green for contributing tokens
        }
      }
      
      console.log('Returning no highlight');
      return ""; // No highlight for other tokens when hovering
    }
    
    // Default state: only highlight activating tokens in orange
    return segment.maxSegmentAct > 0 ? getAccentClassname(segment.maxSegmentAct, maxFeatureAct, "bg") : "";
  };

  return (
    <div className="border rounded p-4 w-full flex flex-col gap-2">
      <div className="flex items-center gap-2 mb-2">
        <h3 className="font-bold">{sampleName}</h3>
        {highestActivatingToken && (
          <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full">
            Max: {highestActivatingToken.activationValue.toFixed(3)}
          </span>
        )}
      </div>

      <div className="flex flex-col gap-4 w-full">
        <div className="flex gap-4 w-full justify-between">
          {/* Text display with highlights */}
          {sample.text && (
            <div className="flex flex-col gap-2">
              <div className="relative whitespace-pre-wrap mb-4">
                {visibleSegments.map((segment, index) => {
                  const segmentText = sample.text!.slice(segment.start, segment.end);

                  return (
                    <span key={index}>
                      {segment.maxSegmentAct > 0 ? (
                        <HoverCard>
                          <HoverCardTrigger>
                            <span
                              className={cn(
                                "relative cursor-help",
                                getSegmentHighlightClass(segment)
                              )}
                              onMouseEnter={() => {
                                // Find the token index for this segment by looking at the feature acts indices
                                // We want to find the token that has the highest activation in this segment
                                let maxActivation = -1;
                                let maxTokenIndex = -1;
                                
                                for (let i = 0; i < sample.featureActsIndices.length; i++) {
                                  const tokenIndex = sample.featureActsIndices[i];
                                  const activation = sample.featureActsValues[i];
                                  
                                  // Check if this token's origin is in our segment
                                  const origin = sample.origins[tokenIndex];
                                  if (origin?.key === "text" && 
                                      segment.highlights.some(highlight => highlight.origin === origin)) {
                                    if (activation > maxActivation) {
                                      maxActivation = activation;
                                      maxTokenIndex = tokenIndex;
                                    }
                                  }
                                }
                                
                                if (maxTokenIndex !== -1) {
                                  setHoveredTokenIndex(maxTokenIndex);
                                }
                              }}
                              onMouseLeave={() => {
                                setHoveredTokenIndex(null);
                              }}
                            >
                              {segmentText.replaceAll("\n", "↵").replaceAll("\t", "→")}
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
                      ) : (
                        <span>{segmentText.replaceAll("\n", "↵").replaceAll("\t", "→")}</span>
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
});
