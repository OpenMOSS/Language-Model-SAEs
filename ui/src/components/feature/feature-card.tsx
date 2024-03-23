import { Feature } from "@/types/feature";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { FeatureActivationSample } from "./sample";
import { Button } from "../ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "../ui/tabs";

export const FeatureCard = ({ feature }: { feature: Feature }) => {
  const analysisNameMap = (analysisName: string) => {
    if (analysisName === "top_activations") {
      return "Top Activations";
    } else if (/^subsample-/.test(analysisName)) {
      const [, proportion] = analysisName.split("-");
      const percentage = parseFloat(proportion) * 100;
      return `Subsample ${percentage}%`;
    }
  };

  return (
    <Card className="container">
      <CardHeader>
        <CardTitle className="flex justify-between items-center text-xl">
          <span>
            #{feature.featureIndex}{" "}
            <span className="font-medium">
              (Activation Times ={" "}
              <span className="font-bold">{feature.actTimes}</span>)
            </span>
          </span>
          <Button>Try Custom Input</Button>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col gap-4">
          <div className="flex flex-col w-full gap-4">
            <Tabs defaultValue="top_activations">
              <TabsList className="font-bold">
                {feature.sampleGroups.map((sampleGroup) => (
                  <TabsTrigger
                    key={`tab-trigger-${sampleGroup.analysisName}`}
                    value={sampleGroup.analysisName}
                  >
                    {analysisNameMap(sampleGroup.analysisName)}
                  </TabsTrigger>
                ))}
              </TabsList>
              {feature.sampleGroups.map((sampleGroup) => (
                <TabsContent value={sampleGroup.analysisName} className="mt-0">
                  <div className="flex flex-col gap-4 mt-4">
                    <p className="font-bold">
                      Max Activation: {feature.maxFeatureAct.toFixed(4)}
                    </p>
                    {sampleGroup.samples.slice(0, 5).map((sample, i) => (
                      <FeatureActivationSample
                        key={i}
                        sample={sample}
                        sampleIndex={i}
                        maxFeatureAct={feature.maxFeatureAct}
                      />
                    ))}
                  </div>
                </TabsContent>
              ))}
            </Tabs>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
