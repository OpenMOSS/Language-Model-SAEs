import { AppNavbar } from "@/components/app/navbar";
import { LogitLensVisualization } from "@/components/logit-lens/logit-lens-visualization";

export const LogitLensPage = () => {
  return (
    <div className="min-h-screen bg-background">
      <AppNavbar />
      <LogitLensVisualization />
    </div>
  );
};
