import { useRouterState } from '@tanstack/react-router'
import NProgress from 'nprogress'
import { useEffect } from 'react'
import 'nprogress/nprogress.css'

NProgress.configure({ showSpinner: false })

export function NavigationProgressBar() {
  const isLoading = useRouterState({ select: (s) => s.isLoading })

  useEffect(() => {
    if (isLoading) {
      NProgress.start()
    } else {
      NProgress.done()
    }
  }, [isLoading])

  return null
}
