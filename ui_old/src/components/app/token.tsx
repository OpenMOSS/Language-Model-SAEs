import { cn } from "@/lib/utils";
import { mergeUint8Arrays } from "@/utils/array";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "../ui/hover-card";
import { Fragment } from "react/jsx-runtime";
import { Separator } from "../ui/separator";

export type PlainTokenGroupProps<T extends { token: Uint8Array }> = {
  tokenGroup: T[];
  tokenGroupClassName?: string;
  tokenGroupProps?: React.HTMLProps<HTMLSpanElement>;
};

export const PlainTokenGroup = <T extends { token: Uint8Array }>({
  tokenGroup,
  tokenGroupClassName,
  tokenGroupProps,
}: PlainTokenGroupProps<T>) => {
  const decoder = new TextDecoder("utf-8", { fatal: true });

  return (
    <span
      className={cn(
        "underline decoration-slate-400 decoration-1 decoration-dotted underline-offset-[6px]",
        tokenGroupClassName
      )}
      {...tokenGroupProps}
    >
      {decoder
        .decode(mergeUint8Arrays(tokenGroup.map((t) => t.token)))
        .replace("\n", "⏎")
        .replace("\t", "⇥")
        .replace("\r", "↵")}
    </span>
  );
};

export type TokenGroupProps<T extends { token: Uint8Array }> = PlainTokenGroupProps<T> & {
  tokenInfoContent?: (token: T, i: number) => React.ReactNode;
  tokenGroupInfoContent?: React.ReactNode;
};

export const TokenGroup = <T extends { token: Uint8Array }>({
  tokenGroup,
  tokenGroupClassName,
  tokenGroupProps,
  tokenInfoContent,
  tokenGroupInfoContent,
}: TokenGroupProps<T>) => {
  if (!tokenInfoContent && !tokenGroupInfoContent) {
    return (
      <PlainTokenGroup
        tokenGroup={tokenGroup}
        tokenGroupClassName={tokenGroupClassName}
        tokenGroupProps={tokenGroupProps}
      />
    );
  }

  return (
    <HoverCard>
      <HoverCardTrigger>
        <PlainTokenGroup
          tokenGroup={tokenGroup}
          tokenGroupClassName={tokenGroupClassName}
          tokenGroupProps={tokenGroupProps}
        />
      </HoverCardTrigger>
      <HoverCardContent className="w-[300px] text-wrap flex flex-col gap-4">
        {tokenGroupInfoContent ? (
          <Fragment>{tokenGroupInfoContent}</Fragment>
        ) : (
          tokenGroup.map((token, i) => (
            <Fragment key={i}>
              {tokenInfoContent?.(token, i)}
              {i < tokenGroup.length - 1 && <Separator />}
            </Fragment>
          ))
        )}
      </HoverCardContent>
    </HoverCard>
  );
};
