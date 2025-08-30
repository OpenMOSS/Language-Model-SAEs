import React, { createContext, useContext, useState, ReactNode } from 'react';
import type { Tracing } from '@/types/model';

interface CircuitState {
  selectedFile: string;
  availableFiles: CircuitFile[];
  tracings: Tracing[] | null;
  usedUrl: string;
  customFileName: string;
  showCustomInput: boolean;
}

interface CircuitFile {
  name: string;
  path: string;
  size?: number;
  lastModified?: string;
}

interface CircuitContextType {
  state: CircuitState;
  updateState: (updates: Partial<CircuitState>) => void;
  resetState: () => void;
}

const initialState: CircuitState = {
  selectedFile: "",
  availableFiles: [],
  tracings: null,
  usedUrl: "",
  customFileName: "",
  showCustomInput: false,
};

const CircuitContext = createContext<CircuitContextType | undefined>(undefined);

export const CircuitProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [state, setState] = useState<CircuitState>(initialState);

  const updateState = (updates: Partial<CircuitState>) => {
    setState(prev => ({ ...prev, ...updates }));
  };

  const resetState = () => {
    setState(initialState);
  };

  return (
    <CircuitContext.Provider value={{ state, updateState, resetState }}>
      {children}
    </CircuitContext.Provider>
  );
};

export const useCircuitContext = () => {
  const context = useContext(CircuitContext);
  if (context === undefined) {
    throw new Error('useCircuitContext must be used within a CircuitProvider');
  }
  return context;
}; 