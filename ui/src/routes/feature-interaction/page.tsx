import { useState, useEffect, useCallback } from "react";
import { AppNavbar } from "@/components/app/navbar";
import { SaeComboLoader } from "@/components/common/SaeComboLoader";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Loader2, Trash2 } from "lucide-react";
import { ChessBoard } from "@/components/chess/chess-board";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { GetFeatureFromFen } from "@/components/feature/get-feature-from-fen";

const LOCAL_STORAGE_KEY = "bt4_sae_combo_id";

interface FeatureNode {
  layer: number;
  pos: number;
  feature: number;
  feature_type: "lorsa" | "transcoder";
  activation_value?: number;
}

interface InteractResult {
  target_node: string;
  original_activation: number;
  modified_activation: number;
  activation_change: number;
  activation_ratio: number;
  steering_scale: number;
  steering_details: Array<{
    node: string;
    found: boolean;
    activation_value?: number;
  }>;
}

interface LoadingStatus {
  loaded: boolean;
  loading: boolean;
  model_name?: string;
  combo_id?: string;
  error?: string;
}

export const FeatureInteractionPage = () => {
  const [fen, setFen] = useState<string>("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  const [layer, setLayer] = useState<number>(0);
  const [position, setPosition] = useState<number>(0);
  const [componentType, setComponentType] = useState<"attn" | "mlp">("attn");
  const [saeComboId, setSaeComboId] = useState<string | undefined>(undefined);
  const [sourceFeatures, setSourceFeatures] = useState<FeatureNode[]>([]);
  const [targetFeature, setTargetFeature] = useState<FeatureNode | null>(null);
  const [steeringScale, setSteeringScale] = useState<number>(-2.0);
  const [interactResult, setInteractResult] = useState<InteractResult | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState<LoadingStatus>({ loaded: false, loading: false });

  useEffect(() => {
    const stored = window.localStorage.getItem(LOCAL_STORAGE_KEY);
    if (stored) {
      setSaeComboId(stored);
    }
  }, []);

  useEffect(() => {
    const handleStorageChange = () => {
      const stored = window.localStorage.getItem(LOCAL_STORAGE_KEY);
      if (stored !== saeComboId) {
        setSaeComboId(stored || undefined);
      }
    };

    window.addEventListener("storage", handleStorageChange);
    const interval = setInterval(() => {
      const stored = window.localStorage.getItem(LOCAL_STORAGE_KEY);
      if (stored !== saeComboId) {
        setSaeComboId(stored || undefined);
      }
    }, 500);

    return () => {
      window.removeEventListener("storage", handleStorageChange);
      clearInterval(interval);
    };
  }, [saeComboId]);

  const checkLoadingStatus = useCallback(async () => {
    if (!saeComboId) return;

    try {
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/circuit/loading_logs?model_name=lc0/BT4-1024x15x32h&sae_combo_id=${saeComboId}`);
      if (response.ok) {
        const data = await response.json();
        setLoadingStatus({
          loaded: !data.is_loading,
          loading: data.is_loading,
          model_name: 'lc0/BT4-1024x15x32h',
          combo_id: saeComboId,
        });
      }
    } catch (error) {
      setLoadingStatus({
        loaded: false,
        loading: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }, [saeComboId]);

  useEffect(() => {
    checkLoadingStatus();
    const interval = setInterval(checkLoadingStatus, 2000);
    return () => clearInterval(interval);
  }, [checkLoadingStatus]);

  // Add source feature
  const addSourceFeature = useCallback((featureIndex: number, activationValue: number) => {
    const newFeature: FeatureNode = {
      layer,
      pos: position,
      feature: featureIndex,
      feature_type: componentType === "attn" ? "lorsa" : "transcoder",
      activation_value: activationValue,
    };
    setSourceFeatures(prev => {
      // Check if already exists
      if (prev.some(f => f.layer === newFeature.layer && f.pos === newFeature.pos && f.feature === newFeature.feature && f.feature_type === newFeature.feature_type)) {
        return prev;
      }
      return [...prev, newFeature];
    });
  }, [layer, position, componentType]);

  // Set target feature
  const setTargetFeatureByIndex = useCallback((featureIndex: number, activationValue: number) => {
    const newFeature: FeatureNode = {
      layer,
      pos: position,
      feature: featureIndex,
      feature_type: componentType === "attn" ? "lorsa" : "transcoder",
      activation_value: activationValue,
    };
    setTargetFeature(newFeature);
  }, [layer, position, componentType]);

  // Remove source feature
  const removeSourceFeature = useCallback((index: number) => {
    setSourceFeatures(prev => prev.filter((_, i) => i !== index));
  }, []);

  // Run feature interaction
  const runFeatureInteraction = useCallback(async () => {
    if (!fen.trim() || sourceFeatures.length === 0 || !targetFeature || !saeComboId) {
      alert('Please ensure FEN is set, at least one source feature and one target feature are selected, and an SAE combo is selected');
      return;
    }

    setAnalyzing(true);
    setInteractResult(null);

    try {
      // Use the first source feature as the steering node
      const sourceFeature = sourceFeatures[0];

      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/interaction/analyze_node_interaction`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_name: 'lc0/BT4-1024x15x32h',
          sae_combo_id: saeComboId,
          fen: fen.trim(),
          steering_node: sourceFeature,
          target_node: targetFeature,
          steering_scale: steeringScale,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const result = await response.json();
      setInteractResult(result);
    } catch (error) {
      console.error('Failed to analyze feature interaction:', error);
      alert(`Analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setAnalyzing(false);
    }
  }, [fen, sourceFeatures, targetFeature, saeComboId, steeringScale]);

  return (
    <div className="min-h-screen bg-background">
      <AppNavbar />
      <div className="container mx-auto p-4">
        <div className="mb-6">
          <h1 className="text-3xl font-bold">Feature Interaction Analysis</h1>
          <p className="text-gray-600 mt-2">
            Analyze the interaction between two features, explore the causal relationships in the neural network
          </p>
        </div>

        {/* SAE Combo Loader */}
        <div className="mb-6">
          <SaeComboLoader />
        </div>

        {/* Loading Status */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Global Model Loading Status</CardTitle>
          </CardHeader>
          <CardContent>
            {loadingStatus.error ? (
              <div className="text-red-600">
                Loading status check failed: {loadingStatus.error}
              </div>
            ) : loadingStatus.loading ? (
              <div className="flex items-center space-x-2">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Loading model: {loadingStatus.model_name} ({loadingStatus.combo_id})</span>
              </div>
            ) : loadingStatus.loaded ? (
              <div className="text-green-600">
                ✅ Model loaded: {loadingStatus.model_name} ({loadingStatus.combo_id})
              </div>
            ) : (
              <div className="text-gray-500">
                Waiting for loading status...
              </div>
            )}
          </CardContent>
        </Card>

        {/* Configuration */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Analysis Configuration</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
              <div className="space-y-2">
                <Label htmlFor="fen-input">FEN string</Label>
                <Input
                  id="fen-input"
                  value={fen}
                  onChange={(e) => setFen(e.target.value)}
                  placeholder="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                  className="bg-white"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="layer-input">Layer (Layer)</Label>
                <Input
                  id="layer-input"
                  type="number"
                  min="0"
                  max="14"
                  value={layer}
                  onChange={(e) => setLayer(parseInt(e.target.value) || 0)}
                  className="bg-white"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="position-input">Position (Position)</Label>
                <Input
                  id="position-input"
                  type="number"
                  min="0"
                  max="63"
                  value={position}
                  onChange={(e) => setPosition(parseInt(e.target.value) || 0)}
                  className="bg-white"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="component-type">Component Type</Label>
                <select
                  id="component-type"
                  value={componentType}
                  onChange={(e) => setComponentType(e.target.value as "attn" | "mlp")}
                  className="w-full p-2 border rounded bg-white"
                >
                  <option value="attn">Attention (Lorsa)</option>
                  <option value="mlp">MLP (Transcoder)</option>
                </select>
              </div>
            </div>

            <div className="flex justify-center">
              <ChessBoard
                fen={fen}
                size="small"
                showCoordinates={true}
                flip_activation={fen.includes(" b ")}
                autoFlipWhenBlack={true}
              />
            </div>
          </CardContent>
        </Card>

        {/* Feature Selection using shared component */}
        <GetFeatureFromFen
          fen={fen}
          layer={layer}
          position={position}
          componentType={componentType}
          modelName="lc0/BT4-1024x15x32h"
          saeComboId={saeComboId}
          actionTypes={["add_to_source", "set_as_target"]}
          showTopActivations={true}
          showFenActivations={true}
          onFeatureAction={(action) => {
            if (action.type === "add_to_source") {
              addSourceFeature(action.featureIndex, action.activationValue);
            } else if (action.type === "set_as_target") {
              setTargetFeatureByIndex(action.featureIndex, action.activationValue);
            }
          }}
          className="mb-6"
        />

        {/* Selected Features */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Source Features */}
          <Card>
            <CardHeader>
              <CardTitle>Source Features (Steering)</CardTitle>
              <p className="text-sm text-gray-600">
                Selected {sourceFeatures.length} steering features
                {sourceFeatures.length > 0 && (
                  <span className="text-xs text-gray-500 ml-2">
                    (The first feature will be used for steering)
                  </span>
                )}
              </p>
            </CardHeader>
            <CardContent>
              {sourceFeatures.length > 0 ? (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Feature</TableHead>
                      <TableHead>Activation</TableHead>
                      <TableHead>Layer/Pos</TableHead>
                      <TableHead></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {sourceFeatures.map((feature, index) => (
                      <TableRow key={index}>
                        <TableCell className="font-mono">#{feature.feature}</TableCell>
                        <TableCell>
                          <span className={feature.activation_value && feature.activation_value > 0 ? 'text-green-600' : 'text-red-600'}>
                            {feature.activation_value?.toFixed(6) || 'N/A'}
                          </span>
                        </TableCell>
                        <TableCell>L{layer} P{feature.pos}</TableCell>
                        <TableCell>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => removeSourceFeature(index)}
                          >
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              ) : (
                <p className="text-gray-500 text-sm">No source features selected</p>
              )}
            </CardContent>
          </Card>

          {/* Target Feature */}
          <Card>
            <CardHeader>
              <CardTitle>Target Feature</CardTitle>
              <p className="text-sm text-gray-600">Select a target feature to observe activation changes</p>
            </CardHeader>
            <CardContent>
              {targetFeature ? (
                <div className="p-4 bg-green-50 border border-green-200 rounded">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-semibold">
                        {targetFeature.feature_type === 'lorsa' ? 'Lorsa' : 'TC'} Feature #{targetFeature.feature}
                      </div>
                      <div className="text-sm text-gray-600">
                        Layer {targetFeature.layer}, Position {targetFeature.pos}
                      </div>
                      <div className="text-sm">
                        Activation: <span className={targetFeature.activation_value && targetFeature.activation_value > 0 ? 'text-green-600' : 'text-red-600'}>
                          {targetFeature.activation_value?.toFixed(6) || 'N/A'}
                        </span>
                      </div>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setTargetFeature(null)}
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              ) : (
                <p className="text-gray-500 text-sm">No target feature selected</p>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Run Analysis */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Run Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center space-x-4">
              <div className="space-y-2">
                <Label htmlFor="steering-scale">Steering Scale</Label>
                <Input
                  id="steering-scale"
                  type="number"
                  step="0.1"
                  value={steeringScale}
                  onChange={(e) => {
                    const value = e.target.value;
                    if (value === '' || value === '-') {
                      setSteeringScale(value === '-' ? -1.0 : 0.0);
                    } else {
                      const parsed = parseFloat(value);
                      setSteeringScale(isNaN(parsed) ? -2.0 : parsed);
                    }
                  }}
                  placeholder="-2.0, 0, 1.5"
                  className="w-24 bg-white"
                />
              </div>
              <Button
                onClick={runFeatureInteraction}
                disabled={analyzing || sourceFeatures.length === 0 || !targetFeature || !loadingStatus.loaded}
                className="mt-6"
              >
                {analyzing ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin mr-2" />
                    Analyzing...
                  </>
                ) : (
                  'Run Feature Interaction'
                )}
              </Button>
              <span className="text-sm text-gray-600">
                Source: {sourceFeatures.length} features, Target: {targetFeature ? '1 feature' : 'none'}
              </span>
            </div>
          </CardContent>
        </Card>

        {/* Results */}
        {interactResult && (
          <Card>
            <CardHeader>
              <CardTitle>Analysis Results</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-4 bg-blue-50 rounded border">
                    <h4 className="font-semibold mb-2">Activation Value Change</h4>
                    <div className="space-y-1 text-sm">
                      <div>Original Activation: {interactResult.original_activation?.toFixed(6)}</div>
                      <div>Modified Activation: {interactResult.modified_activation?.toFixed(6)}</div>
                      <div className={interactResult.activation_change > 0 ? 'text-green-600' : 'text-red-600'}>
                        Change: {interactResult.activation_change?.toFixed(6)}
                      </div>
                      <div className={interactResult.activation_ratio > 1 ? 'text-green-600' : interactResult.activation_ratio < 1 ? 'text-red-600' : 'text-gray-600'}>
                        Ratio: {interactResult.activation_ratio === Number.POSITIVE_INFINITY ? '∞' : interactResult.activation_ratio?.toFixed(4) + 'x'}
                      </div>
                    </div>
                  </div>
                  <div className="p-4 bg-green-50 rounded border">
                    <h4 className="font-semibold mb-2">Steering Information</h4>
                    <div className="space-y-1 text-sm">
                      <div>Scale: {interactResult.steering_scale}</div>
                      <div>Target: {interactResult.target_node}</div>
                      <div>Steering Node Count: {interactResult.steering_details?.length || 0}</div>
                    </div>
                  </div>
                </div>

                {/* Steering Details */}
                {interactResult.steering_details && interactResult.steering_details.length > 0 && (
                  <div className="mt-4">
                    <h4 className="font-semibold mb-2">Steering Details</h4>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Node</TableHead>
                          <TableHead>Found</TableHead>
                          <TableHead>Activation Value</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {interactResult.steering_details.map((detail, index) => (
                          <TableRow key={index}>
                            <TableCell className="font-mono text-sm">{detail.node}</TableCell>
                            <TableCell>
                              <span className={detail.found ? 'text-green-600' : 'text-red-600'}>
                                {detail.found ? '✓' : '✗'}
                              </span>
                            </TableCell>
                            <TableCell>
                              {detail.activation_value !== undefined ? detail.activation_value.toFixed(6) : '-'}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};