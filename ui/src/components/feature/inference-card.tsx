import { useMutation } from '@tanstack/react-query'
import { Send } from 'lucide-react'
import { useState } from 'react'
import { Textarea } from '../ui/textarea'
import { FeatureActivationSample } from './sample'
import { submitCustomInput } from '@/api/features'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Info } from '@/components/ui/info'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'
import { Spinner } from '@/components/ui/spinner'

type InferenceCardProps = {
  dictionaryName: string
  featureIndex: number
  maxFeatureAct: number
}

export const InferenceCard = ({
  dictionaryName,
  featureIndex,
  maxFeatureAct,
}: InferenceCardProps) => {
  const [customInput, setCustomInput] = useState<string>('')

  const submitMutation = useMutation({
    mutationFn: submitCustomInput,
    onSuccess: () => {
      setCustomInput('')
    },
  })

  return (
    <Card
      className={cn(
        'relative w-full overflow-hidden transition-all duration-200',
        submitMutation.isError && 'border-red-500 hover:border-red-600',
      )}
    >
      {/* <ProgressBar isAnimating={submitMutation.isPending} /> */}
      <CardHeader>
        <CardTitle className="font-semibold tracking-tight flex justify-center items-center text-sm text-slate-700 gap-1 cursor-default">
          INFERENCE{' '}
          <Info iconSize={14}>
            Send custom input to the model, and see what features are activated.
          </Info>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col gap-4">
          <div className="flex items-center space-x-2">
            <Textarea
              value={customInput}
              onChange={(e) => setCustomInput(e.target.value)}
              placeholder="Enter text to infer feature activations"
              className="min-h-0 resize-none text-sm overflow-hidden field-sizing-content"
            />
            <Button
              size="icon"
              variant="default"
              className="rounded-full flex items-center justify-center"
              disabled={submitMutation.isPending}
              onClick={() =>
                submitMutation.mutate({
                  data: {
                    dictionaryName,
                    featureIndex,
                    inputText: customInput,
                  },
                })
              }
            >
              {submitMutation.isPending ? (
                <Spinner isAnimating={true} className="text-white" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </div>
          {submitMutation.data && (
            <FeatureActivationSample
              className="rounded-lg bg-slate-100 py-3"
              sample={submitMutation.data}
              maxFeatureAct={maxFeatureAct}
            />
          )}
        </div>
      </CardContent>
    </Card>
  )
}
