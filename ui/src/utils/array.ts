/**
 * Zips multiple arrays together, returning an array of tuples.
 *
 * @param arrays - The arrays to zip.
 * @returns An array of tuples.
 */
export const zip = <T extends unknown[][]>(
  ...arrays: T
): { [K in keyof T]: T[K] extends (infer U)[] ? U : never }[] => {
  const minLength = arrays.reduce(
    (min, arr) => Math.min(min, arr.length),
    Infinity,
  )
  const result = []

  for (let i = 0; i < minLength; i++) {
    result.push(
      arrays.map((arr) => arr[i]) as {
        [K in keyof T]: T[K] extends (infer U)[] ? U : never
      },
    )
  }

  return result
}

/**
 * Merges multiple Uint8Arrays into a single Uint8Array.
 *
 * @param arrays - The arrays to merge.
 * @returns A single Uint8Array.
 */
export const mergeUint8Arrays = (arrays: Uint8Array[]): Uint8Array => {
  const totalLength = arrays.reduce((acc, arr) => acc + arr.length, 0)
  const result = new Uint8Array(totalLength)
  let offset = 0
  for (const arr of arrays) {
    result.set(arr, offset)
    offset += arr.length
  }
  return result
}
