import { Feature } from "@/types/feature";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { FeatureActivationSample } from "./sample";
import { Button } from "../ui/button";

export const FeatureCard = ({ feature }: { feature: Feature }) => {
  return (
    <Card className="container">
      <CardHeader>
        <CardTitle className="flex justify-between items-center">
          #{feature.featureIndex}
          <Button>Try Custom Input</Button>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col gap-4">
          <div className="flex flex-col w-full gap-4">
            <h2 className="text-xl font-bold py-1">
              Top Activations (Max = {feature.maxFeatureAct.toFixed(3)})
            </h2>
            {feature.samples.slice(0, 20).map((sample, i) => (
              <FeatureActivationSample
                key={i}
                sample={sample}
                sampleIndex={i}
                maxFeatureAct={feature.maxFeatureAct}
              />
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
