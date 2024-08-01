import { useState } from "react";
import { Button } from "../ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Textarea } from "../ui/textarea";

const ModelCustomInputArea = () => {
  const [customInput, setCustomInput] = useState<string>("");
  const submit = async () => {};
  const disabled = false;
  return (
    <div className="flex flex-col gap-4">
      <p className="font-bold">Custom Input</p>
      <Textarea
        placeholder="Type your custom input here."
        value={customInput}
        onChange={(e) => setCustomInput(e.target.value)}
      />
      <Button onClick={submit} disabled={disabled}>
        Submit
      </Button>
    </div>
  );
};

export const ModelCard = () => {
  return (
    <Card className="container">
      <CardHeader>
        <CardTitle className="flex justify-between items-center text-xl">
          <span>Model</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col gap-4">
          <ModelCustomInputArea />
        </div>
      </CardContent>
    </Card>
  );
};
