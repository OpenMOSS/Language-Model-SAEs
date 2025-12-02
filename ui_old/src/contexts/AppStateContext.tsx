import React, { createContext, useContext, useReducer, ReactNode, useEffect } from 'react';
import { LinkGraphData } from '@/components/circuits/link-graph/types';
import { FeatureSchema } from '@/types/feature';
import { z } from 'zod';
import { saveStateToStorage, loadStateFromStorage, clearSessionState } from '@/utils/statePersistence';

// Define the state structure
interface AppState {
  circuits: {
    linkGraphData: LinkGraphData | null;
    isLoading: boolean;
    error: string | null;
    clickedId: string | null;
    hoveredId: string | null;
    pinnedIds: string[];
    hiddenIds: string[];
  };
  features: {
    selectedDictionary: string | null;
    selectedAnalysis: string | null;
    featureIndex: number;
    currentFeature: z.infer<typeof FeatureSchema> | null;
    isLoading: boolean;
    error: string | null;
  };
}

// Define action types
type AppAction =
  | { type: 'SET_CIRCUIT_DATA'; payload: LinkGraphData | null }
  | { type: 'SET_CIRCUIT_LOADING'; payload: boolean }
  | { type: 'SET_CIRCUIT_ERROR'; payload: string | null }
  | { type: 'SET_CIRCUIT_CLICKED_ID'; payload: string | null }
  | { type: 'SET_CIRCUIT_HOVERED_ID'; payload: string | null }
  | { type: 'SET_CIRCUIT_PINNED_IDS'; payload: string[] }
  | { type: 'SET_CIRCUIT_HIDDEN_IDS'; payload: string[] }
  | { type: 'SET_FEATURE_DICTIONARY'; payload: string | null }
  | { type: 'SET_FEATURE_ANALYSIS'; payload: string | null }
  | { type: 'SET_FEATURE_INDEX'; payload: number }
  | { type: 'SET_CURRENT_FEATURE'; payload: z.infer<typeof FeatureSchema> | null }
  | { type: 'SET_FEATURE_LOADING'; payload: boolean }
  | { type: 'SET_FEATURE_ERROR'; payload: string | null }
  | { type: 'RESTORE_STATE'; payload: Partial<AppState> }
  | { type: 'RESET_CIRCUITS' }
  | { type: 'RESET_FEATURES' }
  | { type: 'RESET_ALL' };

// Initial state
const initialState: AppState = {
  circuits: {
    linkGraphData: null,
    isLoading: false,
    error: null,
    clickedId: null,
    hoveredId: null,
    pinnedIds: [],
    hiddenIds: [],
  },
  features: {
    selectedDictionary: null,
    selectedAnalysis: null,
    featureIndex: 0,
    currentFeature: null,
    isLoading: false,
    error: null,
  },
};

// Reducer function
function appStateReducer(state: AppState, action: AppAction): AppState {
  let newState: AppState;
  
  switch (action.type) {
    case 'SET_CIRCUIT_DATA':
      newState = {
        ...state,
        circuits: {
          ...state.circuits,
          linkGraphData: action.payload,
        },
      };
      break;
    case 'SET_CIRCUIT_LOADING':
      newState = {
        ...state,
        circuits: {
          ...state.circuits,
          isLoading: action.payload,
        },
      };
      break;
    case 'SET_CIRCUIT_ERROR':
      newState = {
        ...state,
        circuits: {
          ...state.circuits,
          error: action.payload,
        },
      };
      break;
    case 'SET_CIRCUIT_CLICKED_ID':
      newState = {
        ...state,
        circuits: {
          ...state.circuits,
          clickedId: action.payload,
        },
      };
      break;
    case 'SET_CIRCUIT_HOVERED_ID':
      newState = {
        ...state,
        circuits: {
          ...state.circuits,
          hoveredId: action.payload,
        },
      };
      break;
    case 'SET_CIRCUIT_PINNED_IDS':
      newState = {
        ...state,
        circuits: {
          ...state.circuits,
          pinnedIds: action.payload,
        },
      };
      break;
    case 'SET_CIRCUIT_HIDDEN_IDS':
      newState = {
        ...state,
        circuits: {
          ...state.circuits,
          hiddenIds: action.payload,
        },
      };
      break;
    case 'SET_FEATURE_DICTIONARY':
      newState = {
        ...state,
        features: {
          ...state.features,
          selectedDictionary: action.payload,
        },
      };
      break;
    case 'SET_FEATURE_ANALYSIS':
      newState = {
        ...state,
        features: {
          ...state.features,
          selectedAnalysis: action.payload,
        },
      };
      break;
    case 'SET_FEATURE_INDEX':
      newState = {
        ...state,
        features: {
          ...state.features,
          featureIndex: action.payload,
        },
      };
      break;
    case 'SET_CURRENT_FEATURE':
      newState = {
        ...state,
        features: {
          ...state.features,
          currentFeature: action.payload,
        },
      };
      break;
    case 'SET_FEATURE_LOADING':
      newState = {
        ...state,
        features: {
          ...state.features,
          isLoading: action.payload,
        },
      };
      break;
    case 'SET_FEATURE_ERROR':
      newState = {
        ...state,
        features: {
          ...state.features,
          error: action.payload,
        },
      };
      break;
    case 'RESTORE_STATE':
      newState = {
        ...state,
        ...action.payload,
      };
      break;
    case 'RESET_CIRCUITS':
      newState = {
        ...state,
        circuits: initialState.circuits,
      };
      break;
    case 'RESET_FEATURES':
      newState = {
        ...state,
        features: initialState.features,
      };
      break;
    case 'RESET_ALL':
      newState = initialState;
      break;
    default:
      return state;
  }

  // Save to localStorage after state changes (except for loading states and errors)
  if (action.type !== 'SET_CIRCUIT_LOADING' && 
      action.type !== 'SET_FEATURE_LOADING' && 
      action.type !== 'SET_CIRCUIT_ERROR' && 
      action.type !== 'SET_FEATURE_ERROR' &&
      action.type !== 'RESTORE_STATE') {
    saveStateToStorage({
      circuits: {
        linkGraphData: newState.circuits.linkGraphData,
        clickedId: newState.circuits.clickedId,
        hoveredId: newState.circuits.hoveredId,
        pinnedIds: newState.circuits.pinnedIds,
        hiddenIds: newState.circuits.hiddenIds,
      },
      features: {
        selectedDictionary: newState.features.selectedDictionary,
        selectedAnalysis: newState.features.selectedAnalysis,
        featureIndex: newState.features.featureIndex,
      },
    });
  }

  return newState;
}

// Create context
interface AppStateContextType {
  state: AppState;
  dispatch: React.Dispatch<AppAction>;
  clearSessionState: () => void;
}

const AppStateContext = createContext<AppStateContextType | undefined>(undefined);

// Provider component
interface AppStateProviderProps {
  children: ReactNode;
}

export const AppStateProvider: React.FC<AppStateProviderProps> = ({ children }) => {
  const [state, dispatch] = useReducer(appStateReducer, initialState);

  // Load state from sessionStorage on mount
  // Note: Using sessionStorage means state persists when switching tabs
  // but is cleared when the page is refreshed or the browser is closed
  useEffect(() => {
    const savedState = loadStateFromStorage();
    if (savedState) {
      dispatch({ type: 'RESTORE_STATE', payload: savedState });
    }
  }, []);

  const handleClearSessionState = () => {
    clearSessionState();
    dispatch({ type: 'RESET_ALL' });
  };

  return (
    <AppStateContext.Provider value={{ state, dispatch, clearSessionState: handleClearSessionState }}>
      {children}
    </AppStateContext.Provider>
  );
};

// Custom hook to use the context
export const useAppState = () => {
  const context = useContext(AppStateContext);
  if (context === undefined) {
    throw new Error('useAppState must be used within an AppStateProvider');
  }
  return context;
};

// Convenience hooks for specific state sections
export const useCircuitState = () => {
  const { state, dispatch } = useAppState();
  return {
    circuitData: state.circuits.linkGraphData,
    isLoading: state.circuits.isLoading,
    error: state.circuits.error,
    clickedId: state.circuits.clickedId,
    hoveredId: state.circuits.hoveredId,
    pinnedIds: state.circuits.pinnedIds,
    hiddenIds: state.circuits.hiddenIds,
    setCircuitData: (data: LinkGraphData | null) => 
      dispatch({ type: 'SET_CIRCUIT_DATA', payload: data }),
    setLoading: (loading: boolean) => 
      dispatch({ type: 'SET_CIRCUIT_LOADING', payload: loading }),
    setError: (error: string | null) => 
      dispatch({ type: 'SET_CIRCUIT_ERROR', payload: error }),
    setClickedId: (id: string | null) => 
      dispatch({ type: 'SET_CIRCUIT_CLICKED_ID', payload: id }),
    setHoveredId: (id: string | null) => 
      dispatch({ type: 'SET_CIRCUIT_HOVERED_ID', payload: id }),
    setPinnedIds: (ids: string[]) => 
      dispatch({ type: 'SET_CIRCUIT_PINNED_IDS', payload: ids }),
    setHiddenIds: (ids: string[]) => 
      dispatch({ type: 'SET_CIRCUIT_HIDDEN_IDS', payload: ids }),
    reset: () => dispatch({ type: 'RESET_CIRCUITS' }),
  };
};

export const useFeatureState = () => {
  const { state, dispatch } = useAppState();
  return {
    selectedDictionary: state.features.selectedDictionary,
    selectedAnalysis: state.features.selectedAnalysis,
    featureIndex: state.features.featureIndex,
    currentFeature: state.features.currentFeature,
    isLoading: state.features.isLoading,
    error: state.features.error,
    setDictionary: (dictionary: string | null) => 
      dispatch({ type: 'SET_FEATURE_DICTIONARY', payload: dictionary }),
    setAnalysis: (analysis: string | null) => 
      dispatch({ type: 'SET_FEATURE_ANALYSIS', payload: analysis }),
    setFeatureIndex: (index: number) => 
      dispatch({ type: 'SET_FEATURE_INDEX', payload: index }),
    setCurrentFeature: (feature: z.infer<typeof FeatureSchema> | null) => 
      dispatch({ type: 'SET_CURRENT_FEATURE', payload: feature }),
    setLoading: (loading: boolean) => 
      dispatch({ type: 'SET_FEATURE_LOADING', payload: loading }),
    setError: (error: string | null) => 
      dispatch({ type: 'SET_FEATURE_ERROR', payload: error }),
    reset: () => dispatch({ type: 'RESET_FEATURES' }),
  };
}; 