import { mergeUint8Arrays } from "./array";

type Token = {
  token: Uint8Array;
};

export const groupToken = <T extends Token>(tokens: T[]): T[][] => {
  const decoder = new TextDecoder("utf-8", { fatal: true });

  return tokens.reduce<[T[][], T[]]>(
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
  )[0];
};

/**
 * Count the positions of each token group in a list of token groups.
 *
 * @param tokenGroups A list of token groups.
 * @returns A list of starting positions of each token group. Will always start with 0, and the last element will be the total number of tokens.
 */
export const countTokenGroupPositions = <T extends Token>(tokenGroups: T[][]): number[] => {
  return tokenGroups.reduce<number[]>(
    (acc, tokenGroup) => {
      const tokenCount = tokenGroup.length;
      return [...acc, acc[acc.length - 1] + tokenCount];
    },
    [0]
  );
};

export const hex = (token: Token | Uint8Array): string => {
  const tokenArray = "token" in token ? token.token : token;
  return tokenArray.reduce(
    (acc, b) => (b < 32 || b > 126 ? `${acc}\\x${b.toString(16).padStart(2, "0")}` : `${acc}${String.fromCharCode(b)}`),
    ""
  );
};
