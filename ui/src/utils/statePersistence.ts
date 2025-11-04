import { LinkGraphData } from '@/components/circuits/link-graph/types';
import { FeatureSchema } from '@/types/feature';
import { z } from 'zod';

const STORAGE_KEY = 'lm_saes_app_state';

// Serializable versions of the types without circular references
interface SerializableNode {
  id: string;
  nodeId: string;
  featureId: string;
  feature_type: string;
  ctx_idx: number;
  layerIdx: number;
  pos: [number, number];
  xOffset: number;
  yOffset: number;
  nodeColor: string;
  logitPct?: number;
  logitToken?: string;
  featureIndex?: number;
  localClerp?: string;
  remoteClerp?: string;
}

interface SerializableLink {
  source: string;
  target: string;
  pathStr: string;
  color: string;
  strokeWidth: number;
  weight?: number;
  pctInput?: number;
}

interface SerializableLinkGraphData {
  nodes: SerializableNode[];
  links: SerializableLink[];
  metadata: {
    prompt_tokens: string[];
    lorsa_analysis_name?: string;
    clt_analysis_name?: string;
  };
}

interface PersistedState {
  circuits: {
    linkGraphData: SerializableLinkGraphData | null;
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

// Check if sessionStorage is available
const isSessionStorageAvailable = (): boolean => {
  try {
    const test = '__test__';
    sessionStorage.setItem(test, test);
    sessionStorage.removeItem(test);
    return true;
  } catch {
    return false;
  }
};

/**
 * Creates a serializable version of LinkGraphData by removing circular references
 * This function strips out the sourceNode and targetNode properties from links
 * and the sourceLinks and targetLinks properties from nodes
 */
const createSerializableLinkGraphData = (data: LinkGraphData | null): SerializableLinkGraphData | null => {
  if (!data) return null;
  
  return {
    nodes: data.nodes.map(node => {
      // Create a clean node without circular references
      const { sourceLinks, targetLinks, ...cleanNode } = node;
      return cleanNode;
    }),
    links: data.links.map(link => {
      // Create a clean link without circular references
      const { sourceNode, targetNode, ...cleanLink } = link;
      return cleanLink;
    }),
    metadata: data.metadata
  };
};

export const saveStateToStorage = (state: {
  circuits: { linkGraphData: LinkGraphData | null };
  features: {
    selectedDictionary: string | null;
    selectedAnalysis: string | null;
    featureIndex: number;
  };
}): void => {
  if (!isSessionStorageAvailable()) {
    console.warn('sessionStorage is not available, state will not be persisted');
    return;
  }
  
  try {
    // Create a serializable version of the state by removing circular references
    const serializableState = {
      ...state,
      circuits: {
        ...state.circuits,
        linkGraphData: createSerializableLinkGraphData(state.circuits.linkGraphData)
      }
    };
    
    // Use sessionStorage instead of localStorage to persist only during the session
    // This means state will be preserved when switching tabs but not on page refresh
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(serializableState));
  } catch (error) {
    console.warn('Failed to save state to sessionStorage:', error);
  }
};

export const loadStateFromStorage = (): Partial<PersistedState> | null => {
  if (!isSessionStorageAvailable()) {
    return null;
  }
  
  try {
    // Load from sessionStorage instead of localStorage
    const stored = sessionStorage.getItem(STORAGE_KEY);
    if (!stored) return null;
    
    const parsed = JSON.parse(stored);
    
    // Validate the structure
    if (parsed && typeof parsed === 'object') {
      return {
        circuits: {
          linkGraphData: parsed.circuits?.linkGraphData || null,
          isLoading: false,
          error: null,
          clickedId: parsed.circuits?.clickedId || null,
          hoveredId: parsed.circuits?.hoveredId || null,
          pinnedIds: parsed.circuits?.pinnedIds || [],
          hiddenIds: parsed.circuits?.hiddenIds || [],
        },
        features: {
          selectedDictionary: parsed.features?.selectedDictionary || null,
          selectedAnalysis: parsed.features?.selectedAnalysis || null,
          featureIndex: typeof parsed.features?.featureIndex === 'number' ? parsed.features.featureIndex : 0,
          currentFeature: null,
          isLoading: false,
          error: null,
        },
      };
    }
    return null;
  } catch (error) {
    console.warn('Failed to load state from sessionStorage:', error);
    return null;
  }
};

export const clearStateFromStorage = (): void => {
  if (!isSessionStorageAvailable()) {
    return;
  }
  
  try {
    // Clear from sessionStorage instead of localStorage
    sessionStorage.removeItem(STORAGE_KEY);
  } catch (error) {
    console.warn('Failed to clear state from sessionStorage:', error);
  }
};

// Function to manually clear session state (useful for testing or manual cleanup)
export const clearSessionState = (): void => {
  clearStateFromStorage();
};

// Function to check if there's any persisted state
export const hasPersistedState = (): boolean => {
  if (!isSessionStorageAvailable()) {
    return false;
  }
  
  try {
    const stored = sessionStorage.getItem(STORAGE_KEY);
    return stored !== null;
  } catch {
    return false;
  }
}; 