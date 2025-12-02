import { useState } from "react";
import { TokenGroup } from "./token";
import { cn } from "@/lib/utils";

export type SampleProps<T extends { token: Uint8Array }> = {
  sampleName?: string;
  tokenGroups: T[][];
  tokenGroupClassName?: (tokenGroup: T[], i: number) => string;
  tokenGroupProps?: (tokenGroup: T[], i: number) => React.HTMLProps<HTMLSpanElement>;
  tokenInfoContent?: (tokenGroup: T[], i: number) => (token: T, i: number) => React.ReactNode;
  tokenGroupInfoContent?: (tokenGroup: T[], i: number) => React.ReactNode;
  customTokenGroup?: (tokens: T[], i: number) => React.ReactNode;
  foldedStart?: number;
};

export const Sample = <T extends { token: Uint8Array }>({
  sampleName,
  tokenGroups,
  tokenGroupClassName,
  tokenGroupProps,
  tokenInfoContent,
  tokenGroupInfoContent,
  customTokenGroup,
  foldedStart,
}: SampleProps<T>) => {
  const [folded, setFolded] = useState(true);

  return (
    <div className={cn(folded && foldedStart !== undefined && "cursor-pointer -m-1 p-1 rounded-lg hover:bg-gray-100")}>
      <div
        className={cn(folded && foldedStart !== undefined && "line-clamp-3 pb-[1px]")}
        onClick={foldedStart !== undefined && folded ? () => setFolded(!folded) : undefined}
      >
        {sampleName && <span className="text-gray-700 font-bold">{sampleName}: </span>}
        {folded && !!foldedStart && <span className="text-gray-700">...</span>}
        {tokenGroups
          .slice((folded && foldedStart) || 0)
          .map((tokens, i) =>
            customTokenGroup ? (
              customTokenGroup(tokens, i)
            ) : (
              <TokenGroup
                key={`group-${i}`}
                tokenGroup={tokens}
                tokenGroupClassName={tokenGroupClassName?.(tokens, i)}
                tokenGroupProps={tokenGroupProps?.(tokens, i)}
                tokenInfoContent={tokenInfoContent?.(tokens, i)}
                tokenGroupInfoContent={tokenGroupInfoContent?.(tokens, i)}
              />
            )
          )}
      </div>
    </div>
  );
};
