import { AppNavbar } from "@/components/app/navbar";
import { CircuitTaxonomyAnnotation } from "@/components/circuits/circuit-taxonomy-annotation";

export const CircuitTaxonomyPage = () => {
  return (
    <div className="min-h-screen bg-background">
      <AppNavbar />
      <div className="container mx-auto p-4">
        <div className="mb-6">
          <h1 className="text-3xl font-bold">Circuit Taxonomy Annotation</h1>
          <p className="mt-2 text-gray-600">
            Load saved circuit files, review features in layer order, and write taxonomy labels back to MongoDB interpretations.
          </p>
        </div>

        <CircuitTaxonomyAnnotation />
      </div>
    </div>
  );
};
