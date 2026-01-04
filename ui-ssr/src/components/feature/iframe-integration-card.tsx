import { Check, Copy } from 'lucide-react'
import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Info } from '@/components/ui/info'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

type IframeIntegrationCardProps = {
  dictionaryName: string
  featureIndex: number
  className?: string
}

export function IframeIntegrationCard({
  dictionaryName,
  featureIndex,
  className,
}: IframeIntegrationCardProps) {
  const [copied, setCopied] = useState(false)

  const baseUrl =
    typeof window !== 'undefined'
      ? `${window.location.origin}`
      : 'http://localhost:3000'

  const iframeUrl = `${baseUrl}/embed/dictionaries/${dictionaryName}/features/${featureIndex}`

  const iframeSnippet = `<iframe src="${iframeUrl}" width="100%" height="800" frameborder="0"></iframe>`

  const handleCopy = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  return (
    <Card className={cn('w-full', className)}>
      <CardHeader>
        <CardTitle className="font-semibold tracking-tight flex justify-center items-center text-sm text-slate-700 gap-1 cursor-default">
          IFRAME INTEGRATION{' '}
          <Info iconSize={14}>
            Embed this feature view in external websites or documentation using
            an iframe.
          </Info>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-slate-600">
                Direct URL
              </label>
              <Button
                variant="ghost"
                size="sm"
                className="h-6 px-2 text-xs"
                onClick={() => handleCopy(iframeUrl)}
              >
                {copied ? (
                  <>
                    <Check className="h-3 w-3 mr-1" />
                    Copied
                  </>
                ) : (
                  <>
                    <Copy className="h-3 w-3 mr-1" />
                    Copy
                  </>
                )}
              </Button>
            </div>
            <div className="relative">
              <input
                type="text"
                readOnly
                value={iframeUrl}
                className="w-full px-3 py-2 text-xs font-mono bg-slate-50 border border-slate-200 rounded-md focus:outline-none focus:ring-2 focus:ring-slate-400 select-all"
              />
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-slate-600">
                Iframe Code
              </label>
              <Button
                variant="ghost"
                size="sm"
                className="h-6 px-2 text-xs"
                onClick={() => handleCopy(iframeSnippet)}
              >
                {copied ? (
                  <>
                    <Check className="h-3 w-3 mr-1" />
                    Copied
                  </>
                ) : (
                  <>
                    <Copy className="h-3 w-3 mr-1" />
                    Copy
                  </>
                )}
              </Button>
            </div>
            <div className="relative">
              <textarea
                readOnly
                value={iframeSnippet}
                rows={3}
                className="w-full px-3 py-2 text-xs font-mono bg-slate-50 border border-slate-200 rounded-md focus:outline-none focus:ring-2 focus:ring-slate-400 select-all resize-none"
              />
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
