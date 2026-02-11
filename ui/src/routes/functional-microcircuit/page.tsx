import React from 'react';
import { AppNavbar } from '@/components/app/navbar';
import { FunctionalMicrocircuitVisualization } from '@/components/functional-microcircuit/functional-microcircuit-visualization';

export const FunctionalMicrocircuitPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-background">
      <AppNavbar />
      <div className="container mx-auto p-6">
        <FunctionalMicrocircuitVisualization />
      </div>
    </div>
  );
};
