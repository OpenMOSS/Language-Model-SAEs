import { memo, useCallback, useLayoutEffect, useRef, useState } from 'react'
import { IntersectionObserver } from './intersection-observer'
import { Spinner } from './spinner'
import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

type InfiniteListProps = {
  children: ReactNode
  onLoadMore: () => void
  hasNextPage: boolean
  onLoadPrevious?: () => void
  hasPreviousPage?: boolean
  isLoading?: boolean
  className?: string
  intersectionRootMargin?: string
  onFirstItemChange?: (firstItemKey: string | number | null) => void
}

export const InfiniteList = memo(
  ({
    children,
    onLoadMore,
    hasNextPage,
    onLoadPrevious,
    hasPreviousPage,
    isLoading = false,
    className,
    intersectionRootMargin = '200px',
    onFirstItemChange,
  }: InfiniteListProps) => {
    const containerRef = useRef<HTMLDivElement>(null)
    const [containerElement, setContainerElement] =
      useState<HTMLDivElement | null>(null)
    const previousScrollHeightRef = useRef(0)
    const previousFirstItemKeyRef = useRef<string | number | null>(null)

    const setRef = useCallback((node: HTMLDivElement | null) => {
      containerRef.current = node
      setContainerElement(node)
    }, [])

    const handleLoadPrevious = useCallback(() => {
      if (containerRef.current) {
        previousScrollHeightRef.current = containerRef.current.scrollHeight
      }
      onLoadPrevious?.()
    }, [onLoadPrevious])

    useLayoutEffect(() => {
      if (!containerRef.current) return

      const firstElement = containerRef.current.firstElementChild
      const currentFirstKey = firstElement?.getAttribute('data-key') ?? null

      if (previousScrollHeightRef.current > 0) {
        const newScrollHeight = containerRef.current.scrollHeight
        const diff = newScrollHeight - previousScrollHeightRef.current

        if (diff > 0 && currentFirstKey !== previousFirstItemKeyRef.current) {
          containerRef.current.scrollTop += diff
        }
        previousScrollHeightRef.current = 0
      }

      if (currentFirstKey !== previousFirstItemKeyRef.current) {
        previousFirstItemKeyRef.current = currentFirstKey
        onFirstItemChange?.(currentFirstKey)
      }
    }, [children, onFirstItemChange])

    return (
      <div ref={setRef} className={cn('flex flex-col', className)}>
        {hasPreviousPage && onLoadPrevious && (
          <IntersectionObserver
            onIntersect={handleLoadPrevious}
            enabled={hasPreviousPage && !isLoading && !!containerElement}
            root={containerElement}
            rootMargin={`${intersectionRootMargin} 0px`}
            className="h-8 w-full flex items-center justify-center p-1"
          >
            <Spinner isAnimating={true} className="text-slate-400" />
          </IntersectionObserver>
        )}
        {children}
        {hasNextPage && (
          <IntersectionObserver
            onIntersect={onLoadMore}
            enabled={hasNextPage && !isLoading && !!containerElement}
            root={containerElement}
            rootMargin={`0px 0px ${intersectionRootMargin} 0px`}
            className="h-8 w-full flex items-center justify-center p-1"
          >
            <Spinner isAnimating={true} className="text-slate-400" />
          </IntersectionObserver>
        )}
      </div>
    )
  },
)

InfiniteList.displayName = 'InfiniteList'
