import { useMutation } from '@tanstack/react-query'
import { Send } from 'lucide-react'
import { useState } from 'react'
import { FeatureCardHorizontal } from '../feature/feature-card-horizontal'
import { FeatureActivationSample } from '../feature/sample'
import { FeatureInterpretation } from '../feature/feature-interpretation'
import { dictionaryInference } from '@/api/features'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Info } from '@/components/ui/info'
import { Textarea } from '@/components/ui/textarea'
import { Spinner } from '@/components/ui/spinner'

type DictionaryInferenceProps = {
  dictionaryName: string
}

export const DictionaryInference = ({
  dictionaryName,
}: DictionaryInferenceProps) => {
  const [inputText, setInputText] = useState('')

  const inferenceMutation = useMutation({
    mutationFn: dictionaryInference,
  })

  const handleInference = () => {
    if (!inputText.trim()) return
    inferenceMutation.mutate({
      data: {
        dictionaryName,
        text: inputText.trim(),
      },
    })
  }

  return (
    <div className="w-full flex flex-col gap-6">
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="font-semibold tracking-tight flex justify-center items-center text-sm text-slate-700 gap-1 cursor-default">
            DICTIONARY INFERENCE{' '}
            <Info iconSize={14}>
              Run inference on the entire dictionary with custom text to find
              the most activated features.
            </Info>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4">
            <Textarea
              placeholder="Enter text to find activated features..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              className="min-h-[100px] flex-1"
            />
            <Button
              className="h-auto px-6"
              onClick={handleInference}
              disabled={inferenceMutation.isPending || !inputText.trim()}
            >
              {inferenceMutation.isPending ? (
                <Spinner isAnimating={true} className="text-white" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {inferenceMutation.data && (
        <div className="flex flex-col gap-6 w-full">
          <h3 className="text-lg font-semibold px-4">Activated Features</h3>
          {inferenceMutation.data.map((result, index) => (
            <div key={index} className="flex flex-col gap-2 w-full">
              <div className="flex gap-4 items-stretch">
                <div className="flex-1">
                  <FeatureInterpretation
                    feature={result.feature}
                    dictionaryName={dictionaryName}
                    featureIndex={result.feature.featureIndex}
                    className="h-full"
                  />
                </div>
                <div className="flex-1">
                  <Card className="h-full">
                    <CardHeader className="py-3">
                      <CardTitle className="text-sm font-semibold text-slate-700 flex justify-center items-center gap-1">
                        INFERENCE RESULT
                        <Info iconSize={14}>
                          How this feature activates on your input text.
                        </Info>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <FeatureActivationSample
                        sample={result.inference}
                        maxFeatureAct={result.feature.maxFeatureAct}
                      />
                    </CardContent>
                  </Card>
                </div>
              </div>
              <FeatureCardHorizontal
                feature={result.feature as any}
                hidePlots
              />
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
