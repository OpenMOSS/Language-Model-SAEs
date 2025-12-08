import { mergeUint8Arrays } from './array'

type Token = {
  token: Uint8Array
}

/**
 * Groups tokens into decodable UTF-8 sequences.
 *
 * @param tokens - The tokens to group.
 * @returns An array of token groups.
 */
export const groupToken = <T extends Token>(tokens: T[]): T[][] => {
  const decoder = new TextDecoder('utf-8', { fatal: true })

  return tokens.reduce<[T[][], T[]]>(
    ([groups, currentGroup], token) => {
      const newGroup = [...currentGroup, token]
      try {
        decoder.decode(mergeUint8Arrays(newGroup.map((t) => t.token)))
        return [[...groups, newGroup], []]
      } catch {
        return [groups, newGroup]
      }
    },
    [[], []],
  )[0]
}

/**
 * Count the positions of each token group in a list of token groups.
 *
 * @param tokenGroups - A list of token groups.
 * @returns A list of starting positions of each token group. Will always start with 0.
 */
export const countTokenGroupPositions = <T extends Token>(
  tokenGroups: T[][],
): number[] => {
  return tokenGroups.reduce<number[]>(
    (acc, tokenGroup) => {
      const tokenCount = tokenGroup.length
      return [...acc, acc[acc.length - 1] + tokenCount]
    },
    [0],
  )
}

/**
 * Converts a token to a hex string representation.
 *
 * @param token - The token to convert.
 * @returns A hex string representation.
 */
export const hex = (token: Token | Uint8Array): string => {
  const tokenArray = 'token' in token ? token.token : token
  return tokenArray.reduce(
    (acc, b) =>
      b < 32 || b > 126
        ? `${acc}\\x${b.toString(16).padStart(2, '0')}`
        : `${acc}${String.fromCharCode(b)}`,
    '',
  )
}

/**
 * Helper function to get z pattern for a specific token index.
 *
 * @param zPatternIndices - The z pattern indices array.
 * @param zPatternValues - The z pattern values array.
 * @param tokenIndex - The token index to look up.
 * @returns The contributing tokens and their contributions, or null.
 */
export const getZPatternForToken = (
  zPatternIndices: number[][] | null | undefined,
  zPatternValues: number[] | null | undefined,
  tokenIndex: number,
): {
  contributingTokens: number[]
  contributions: number[]
} | null => {
  if (!zPatternIndices || !zPatternValues) {
    return null
  }

  const contributingTokens: number[] = []
  const contributions: number[] = []

  // Find all z pattern entries that contribute to this token
  for (let i = 0; i < zPatternIndices[0].length; i++) {
    const indices = zPatternIndices[0][i]

    if (indices === tokenIndex) {
      contributingTokens.push(zPatternIndices[1][i])
      contributions.push(zPatternValues[i])
    }
  }

  return { contributingTokens, contributions }
}

/**
 * Helper function to find the highest activating token.
 *
 * @param featureActsIndices - The feature activation indices.
 * @param featureActsValues - The feature activation values.
 * @returns The token index and activation value, or null.
 */
export const findHighestActivatingToken = (
  featureActsIndices: number[],
  featureActsValues: number[],
): { tokenIndex: number; activationValue: number } | null => {
  if (featureActsIndices.length === 0 || featureActsValues.length === 0) {
    return null
  }

  let maxIndex = 0
  let maxValue = featureActsValues[0]

  for (let i = 1; i < featureActsValues.length; i++) {
    if (featureActsValues[i] > maxValue) {
      maxValue = featureActsValues[i]
      maxIndex = i
    }
  }

  return {
    tokenIndex: featureActsIndices[maxIndex],
    activationValue: maxValue,
  }
}
