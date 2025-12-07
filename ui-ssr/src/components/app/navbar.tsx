import { Link, useRouterState } from '@tanstack/react-router'
import { cn } from '@/lib/utils'

/**
 * Application navigation bar with links to main sections.
 */
export const AppNavbar = () => {
  const routerState = useRouterState()
  const pathname = routerState.location.pathname

  return (
    <nav className="p-4 border-b border-border bg-card">
      <div className="container mx-auto flex items-center gap-8">
        <Link to="/" className="flex items-center gap-2">
          <span className="text-xl font-bold">SAE Visualizer</span>
        </Link>

        <div className="flex gap-4 items-center">
          <Link
            className={cn(
              'transition-colors hover:text-foreground/80 text-foreground/60',
              pathname === '/dictionaries' && 'text-foreground font-medium',
            )}
            to="/dictionaries"
          >
            Dictionaries
          </Link>
        </div>
      </div>
    </nav>
  )
}
