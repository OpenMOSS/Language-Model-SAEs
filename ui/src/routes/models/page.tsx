import { AppNavbar } from "@/components/app/navbar";
import { ModelCard } from "@/components/model/model-card";

export const ModelsPage = () => {
  return (
    <div id="Top">
      <AppNavbar />
      <div className="pt-4 pb-20 px-20 flex flex-col items-center gap-12">
        <div className="container flex gap-12">
          <ModelCard />
        </div>
      </div>
    </div>
  );
};
