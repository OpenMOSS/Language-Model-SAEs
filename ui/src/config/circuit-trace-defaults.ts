import defaultsJson from './circuit-trace-defaults.json';

type CircuitTraceDefaults = {
  max_feature_nodes: number;
  node_threshold: number;
  edge_threshold: number;
  save_activation_info: boolean;
};

export const CIRCUIT_TRACE_DEFAULTS: CircuitTraceDefaults = defaultsJson;
