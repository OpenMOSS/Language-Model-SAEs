declare module 'rbush-knn' {
  import type RBush from 'rbush'

  interface BBox {
    minX: number
    minY: number
    maxX: number
    maxY: number
  }

  function knn<T extends BBox>(
    tree: RBush<T>,
    x: number,
    y: number,
    n?: number,
    predicate?: (item: T) => boolean,
    maxDistance?: number,
  ): T[]

  export default knn
}
