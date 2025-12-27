import { useMutation } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import { Send } from 'lucide-react'
import { useState } from 'react'
import { FeatureActivationSample } from '../feature/sample'
import { FeatureCardWithSamples } from '../feature/feature-card'
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
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="font-semibold tracking-tight flex justify-center items-center text-sm text-slate-700 gap-1 cursor-default">
          DICTIONARY INFERENCE{' '}
          <Info iconSize={14}>
            Run inference on the entire dictionary with custom text to find the
            most activated features.
          </Info>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-6">
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

        {inferenceMutation.data && inferenceMutation.data.length > 0 && (
          <>
            <div className="border-t border-slate-200" />
            <div className="flex flex-col gap-5">
              {inferenceMutation.data.map((result, index) => (
                <div
                  key={`${result.feature.featureIndex}-${index}`}
                  className="flex gap-3"
                >
                  <div className="flex items-start gap-4">
                    <div className="flex-1 space-y-3">
                      {result.feature.interpretation?.text && (
                        <div className="text-sm text-slate-700 leading-relaxed font-token">
                          {result.feature.interpretation.text}
                        </div>
                      )}
                      {!result.feature.interpretation?.text && (
                        <div className="text-sm text-slate-400 italic">
                          No interpretation available
                        </div>
                      )}
                      <div className="bg-slate-50 rounded-lg p-3">
                        <FeatureActivationSample
                          sample={result.inference}
                          maxFeatureAct={result.feature.maxFeatureAct}
                        />
                      </div>
                    </div>
                    <Link
                      to="/dictionaries/$dictionaryName/features/$featureIndex"
                      params={{
                        dictionaryName,
                        featureIndex: String(result.feature.featureIndex),
                      }}
                      className="text-xs text-slate-500 hover:text-slate-700 shrink-0 pt-1"
                    >
                      #{result.feature.featureIndex}
                    </Link>
                  </div>
                  <FeatureCardWithSamples
                    feature={result.feature}
                    className="border-0 shadow-none bg-slate-50"
                  />
                </div>
              ))}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  )
}
