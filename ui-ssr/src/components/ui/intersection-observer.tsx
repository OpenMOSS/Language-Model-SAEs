import { useEffect, useRef } from 'react'

interface IntersectionObserverProps extends React.HTMLAttributes<HTMLDivElement> {
  onIntersect: () => void
  root?: Element | Document | null
  rootMargin?: string
  threshold?: number | number[]
  enabled?: boolean
}

export function IntersectionObserver({
  onIntersect,
  root,
  rootMargin,
  threshold = 0.1,
  enabled = true,
  children,
  ...props
}: IntersectionObserverProps) {
  const targetRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!enabled) return

    const observer = new window.IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            onIntersect()
          }
        })
      },
      {
        root,
        rootMargin,
        threshold,
      },
    )

    const element = targetRef.current
    if (element) {
      observer.observe(element)
    }

    return () => {
      if (element) {
        observer.unobserve(element)
      }
    }
  }, [enabled, onIntersect, root, rootMargin, threshold])

  return (
    <div ref={targetRef} {...props}>
      {children}
    </div>
  )
}
