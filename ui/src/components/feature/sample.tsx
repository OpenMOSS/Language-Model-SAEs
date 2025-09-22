import { Feature, FeatureSampleCompact, ImageTokenOrigin, TextTokenOrigin } from "@/types/feature";
import { useState, useCallback, useMemo, memo } from "react";
import { AppPagination } from "../ui/pagination";
import { getAccentClassname } from "@/utils/style";
import { cn } from "@/lib/utils";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "../ui/hover-card";
import { Switch } from "../ui/switch";
import { Input } from "../ui/input";
import { Button } from "../ui/button";
import { Checkbox } from "../ui/checkbox";
import { Download } from "lucide-react";

export const FeatureSampleGroup = ({
  feature,
  sampleGroup,
}: {
  feature: Feature;
  sampleGroup: Feature["sampleGroups"][0];
}) => {
  const [page, setPage] = useState<number>(1);
  const [visibleRange, setVisibleRange] = useState<number>(50);
  
  // SVG Export state
  const [selectedSamples, setSelectedSamples] = useState<Set<number>>(new Set());
  const [svgWidth, setSvgWidth] = useState<number>(800);
  const [svgOffset, setSvgOffset] = useState<number>(0);
  
  const maxPage = useMemo(() => Math.ceil(sampleGroup.samples.length / 10), [sampleGroup.samples.length]);
  
  const currentSamples = useMemo(() => 
    sampleGroup.samples.slice((page - 1) * 10, page * 10),
    [sampleGroup.samples, page]
  );

  const maxActivation = useMemo(() => 
    sampleGroup.samples.length > 0 ? Math.max(...sampleGroup.samples[0].featureActs) : 0,
    [sampleGroup.samples]
  );

  // Handle input change
  const handleRangeChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value);
    if (!isNaN(value) && value > 0) {
      setVisibleRange(value);
    }
  }, []);

  // SVG Export handlers
  const handleSampleSelection = useCallback((sampleIndex: number, checked: boolean) => {
    const newSelected = new Set(selectedSamples);
    if (checked) {
      newSelected.add(sampleIndex);
    } else {
      newSelected.delete(sampleIndex);
    }
    setSelectedSamples(newSelected);
  }, [selectedSamples]);

  const handleSelectAll = useCallback((checked: boolean) => {
    if (checked) {
      const allSamples = new Set(currentSamples.map((_, i) => (page - 1) * 10 + i));
      setSelectedSamples(allSamples);
    } else {
      setSelectedSamples(new Set());
    }
  }, [currentSamples, page]);

  // SVG Export utility functions
  const escapeXML = useCallback((text: string): string => {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }, []);

  const downloadSVG = useCallback((content: string, filename: string) => {
    const blob = new Blob([content], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, []);

  const getColorFromActivation = useCallback((featureAct: number, maxFeatureAct: number): string => {
    const intensity = Math.ceil(Math.min(featureAct / maxFeatureAct, 1) * 5);
    
    const colorMap: Record<number, string> = {
      1: '#fff5e6',
      2: '#ffe0b3',
      3: '#ffcc80', 
      4: '#ffb74d',
      5: '#ffa726',
    };
    
    return colorMap[intensity] || 'transparent';
  }, []);

  const exportSelectedSamplesToSVG = useCallback(() => {
    if (selectedSamples.size === 0) {
      alert('No samples selected for export');
      return;
    }

    // SVG configuration
    const fontSize = 16;
    const fontFamily = '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif';
    const lineHeight = fontSize * 1.2;
    const lineSpacing = 2;

    // Create a temporary canvas to measure text
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.font = `${fontSize}px ${fontFamily}`;

    const svgElements: Array<{
      type: 'rect' | 'text';
      x: number;
      y: number;
      width?: number;
      height?: number;
      fill: string;
      text?: string;
    }> = [];
    
    let currentY = fontSize; // Start with baseline position for first line
    let maxWidth = 0;

    // Process each selected sample
    const selectedSampleIndices = Array.from(selectedSamples).sort((a, b) => a - b);
    
    selectedSampleIndices.forEach((sampleIndex, _lineIndex) => {
      const sample = sampleGroup.samples[sampleIndex];
      if (!sample.text) return;

      // Get text highlights for this sample
      const textHighlights = sample.origins
        .map((origin, index) => ({
          origin,
          featureAct: sample.featureActs[index],
        }))
        .filter((item): item is { origin: TextTokenOrigin; featureAct: number } => 
          item.origin?.key === "text"
        );

      if (textHighlights.length === 0) return;

      // Find max activation highlight for centering
      const maxActivationHighlight = textHighlights.reduce(
        (max, current) => (current.featureAct > max.featureAct ? current : max),
        textHighlights[0]
      );

      // Create segments
      const segments: Array<{
        start: number;
        end: number;
        maxSegmentAct: number;
      }> = [];
      
      const positions = new Set<number>();
      textHighlights.forEach((h) => {
        positions.add(h.origin.range[0]);
        positions.add(h.origin.range[1]);
      });
      const sortedPositions = Array.from(positions).sort((a, b) => a - b);

      for (let i = 0; i < sortedPositions.length - 1; i++) {
        const start = sortedPositions[i];
        const end = sortedPositions[i + 1];
        const activeHighlights = textHighlights.filter(
          (h) => h.origin.range[0] <= start && h.origin.range[1] >= end
        );
        if (activeHighlights.length > 0) {
          const maxSegmentAct = Math.max(...activeHighlights.map((h) => h.featureAct));
          segments.push({ start, end, maxSegmentAct });
        }
      }

      // Filter visible segments based on visibleRange
      const maxActivationSegmentIndex = segments.findIndex((segment) => {
        const activeHighlights = textHighlights.filter(
          (h) => h.origin.range[0] <= segment.start && h.origin.range[1] >= segment.end
        );
        return activeHighlights.some((h) => h.featureAct === maxActivationHighlight.featureAct);
      });

      const visibleSegments = segments.filter((_segment, index) => {
        if (maxActivationSegmentIndex === -1) return true;
        return Math.abs(index - maxActivationSegmentIndex) <= visibleRange;
      });

      // Calculate centering offset based on max activation token
      let centeringOffset = 0;
      if (maxActivationSegmentIndex >= 0 && visibleSegments.length > 0) {
        const maxActivationSegment = segments[maxActivationSegmentIndex];
        if (maxActivationSegment) {
          // Calculate text width up to the max activation segment
          let widthToMaxActivation = 0;
          for (const segment of visibleSegments) {
            if (segment.start >= maxActivationSegment.start) break;
            const segmentText = sample.text.slice(segment.start, segment.end)
              .replaceAll('\n', '↵')
              .replaceAll('\t', '→');
            widthToMaxActivation += ctx.measureText(segmentText).width;
          }
          centeringOffset = svgWidth / 2 - widthToMaxActivation;
        }
      }

      // Apply offset parameter
      let currentX = centeringOffset + svgOffset;

      // Process each visible segment
      visibleSegments.forEach((segment) => {
        const segmentText = sample.text!
          .slice(segment.start, segment.end)
          .replaceAll('\n', '↵')
          .replaceAll('\t', '→');
        
        const textWidth = ctx.measureText(segmentText).width;

        if (segment.maxSegmentAct > 0) {
          // Create background rectangle for highlight
          svgElements.push({
            type: 'rect',
            x: currentX,
            y: currentY - fontSize * 0.9,
            width: textWidth,
            height: fontSize * 1.2,
            fill: getColorFromActivation(segment.maxSegmentAct, feature.maxFeatureAct),
          });
        }

        // Create text element
        svgElements.push({
          type: 'text',
          x: currentX,
          y: currentY,
          text: segmentText,
          fill: '#333',
        });

        currentX += textWidth;
        maxWidth = Math.max(maxWidth, currentX);
      });

      currentY += lineHeight + lineSpacing;
    });

    // Adjust total height
    const totalHeight = currentY - lineHeight - lineSpacing + fontSize * 0.3;

    // Generate SVG content
    let svgContent = `<?xml version="1.0" encoding="UTF-8"?>
<svg width="${svgWidth}" height="${totalHeight}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .text-element {
        font-family: ${fontFamily};
        font-size: ${fontSize}px;
        fill: #333;
        dominant-baseline: alphabetic;
        white-space: pre;
      } 
    </style>
  </defs>
`;

    // Add all SVG elements
    svgElements.forEach((element) => {
      if (element.type === 'rect') {
        svgContent += `  <rect x="${element.x}" y="${element.y}" width="${element.width}" height="${element.height}" fill="${element.fill}" stroke="none"/>
`;
      } else if (element.type === 'text') {
        const escapedText = escapeXML(element.text!);
        svgContent += `  <text x="${element.x}" y="${element.y}" class="text-element">${escapedText}</text>
`;
      }
    });

    svgContent += '</svg>';

    // Download the SVG file
    const filename = `feature-${feature.featureIndex}-activations.svg`;
    downloadSVG(svgContent, filename);
  }, [selectedSamples, sampleGroup.samples, visibleRange, svgWidth, svgOffset, feature.maxFeatureAct, feature.featureIndex, getColorFromActivation, escapeXML, downloadSVG]);

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

      {/* SVG Export Controls */}
      <div className="border rounded p-4 bg-gray-50">
        <h4 className="font-bold mb-3">SVG Export Configuration</h4>
        <div className="flex flex-col gap-3">
          <div className="flex items-center gap-4">
            <Checkbox 
              checked={selectedSamples.size === currentSamples.length && currentSamples.length > 0}
              onCheckedChange={handleSelectAll}
            />
            <span className="text-sm font-medium">Select All ({selectedSamples.size} selected)</span>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-sm font-bold min-w-[80px]">Width:</div>
            <Input
              type="number"
              value={svgWidth.toString()}
              onChange={(e) => setSvgWidth(parseInt(e.target.value) || 800)}
              className="w-[100px]"
              min={100}
            />
            <div className="text-sm text-muted-foreground">pixels</div>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-sm font-bold min-w-[80px]">Offset:</div>
            <Input
              type="number"
              value={svgOffset.toString()}
              onChange={(e) => setSvgOffset(parseInt(e.target.value) || 0)}
              className="w-[100px]"
            />
            <div className="text-sm text-muted-foreground">pixels</div>
          </div>
          <Button 
            onClick={exportSelectedSamplesToSVG}
            disabled={selectedSamples.size === 0}
            className="w-fit"
            size="sm"
          >
            <Download className="h-4 w-4 mr-2" />
            Export SVG ({selectedSamples.size} samples)
          </Button>
        </div>
      </div>

      {currentSamples.map((sample, i) => {
        const sampleIndex = (page - 1) * 10 + i;
        return (
          <FeatureActivationSample
            key={sampleIndex}
            sample={sample}
            sampleName={`Sample ${sampleIndex + 1}`}
            maxFeatureAct={feature.maxFeatureAct}
            visibleRange={visibleRange}
            isSelected={selectedSamples.has(sampleIndex)}
            onSelectionChange={(checked) => handleSampleSelection(sampleIndex, checked)}
          />
        );
      })}
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
  visibleRange?: number;
  isSelected?: boolean;
  onSelectionChange?: (checked: boolean) => void;
};

export const FeatureActivationSample = memo(({
  sample,
  sampleName,
  maxFeatureAct,
  visibleRange,
  isSelected = false,
  onSelectionChange,
}: FeatureActivationSampleProps) => {
  // Memoize text highlights processing
  const textHighlights = useMemo(() => 
    sample.origins
      .map((origin, index) => ({
        origin,
        featureAct: sample.featureActs[index],
      }))
      .filter((item): item is { origin: TextTokenOrigin; featureAct: number } => item.origin?.key === "text"),
    [sample.origins, sample.featureActs]
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
        featureAct: sample.featureActs[index],
      }))
      .filter((item): item is { origin: ImageTokenOrigin; featureAct: number } => item.origin?.key === "image"),
    [sample.origins, sample.featureActs]
  );

  const [showImageHighlights, setShowImageHighlights] = useState(true);
  const [showImageGrid, setShowImageGrid] = useState(false);

  // Determine if we have any images to display
  const hasImages = sample.images && sample.images.length > 0;

  return (
    <div className="border rounded p-4 w-full flex flex-col gap-2">
      <div className="flex items-center gap-3 mb-2">
        {onSelectionChange && (
          <Checkbox
            checked={isSelected}
            onCheckedChange={onSelectionChange}
          />
        )}
        <h3 className="font-bold">{sampleName}</h3>
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
                                getAccentClassname(segment.maxSegmentAct, maxFeatureAct, "bg")
                              )}
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
