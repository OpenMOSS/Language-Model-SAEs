import React, { useState, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Loader2, Upload, FileText, ExternalLink } from 'lucide-react';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { SaeComboLoader } from '@/components/common/SaeComboLoader';

interface FeatureResult {
  layer: number;
  feature: number;
  diff: number;
  p_random: number;
  p_tactic: number;
  kind: string;
}

interface AnalysisResult {
  valid_tactic_fens: number;
  invalid_tactic_fens: number;
  random_fens: number;
  tactic_fens: number;
  top_lorsa_features: FeatureResult[];
  top_tc_features: FeatureResult[];
  invalid_fens_sample: string[];
  specific_layer_lorsa?: FeatureResult[];
  specific_layer_tc?: FeatureResult[];
  specific_layer?: number;
}

export const TacticFeaturesVisualization: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [topKLorsa, setTopKLorsa] = useState<number>(10);
  const [topKTC, setTopKTC] = useState<number>(10);
  const [nFens, setNFens] = useState<number>(200);
  const [specificLayer, setSpecificLayer] = useState<string>('');
  const [specificLayerTopK, setSpecificLayerTopK] = useState<number>(20);

  const buildDictionaryName = useCallback((layer: number, kind: string): string => {
    if (kind === 'Lorsa') {
      return `BT4_lorsa_L${layer}A`;
    } else { // TC
      return `BT4_tc_L${layer}M`;
    }
  }, []);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.type === 'text/plain' || file.name.endsWith('.txt')) {
        setSelectedFile(file);
        setError(null);
      } else {
        setError('Please upload a .txt file');
        setSelectedFile(null);
      }
    }
  }, []);

  const runAnalysis = useCallback(async () => {
    if (!selectedFile) {
      setError('Please select a file first');
      return;
    }

    setIsLoading(true);
    setError(null);
    setAnalysisResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('n_random', nFens.toString());
      formData.append('n_fens', nFens.toString());
      formData.append('top_k_lorsa', topKLorsa.toString());
      formData.append('top_k_tc', topKTC.toString());
      
      if (specificLayer && !isNaN(parseInt(specificLayer))) {
        formData.append('specific_layer', specificLayer);
        formData.append('specific_layer_top_k', specificLayerTopK.toString());
      }
      
      console.log('🔍 Sending analysis request (fixed using BT4 model):', {
        n_fens: nFens,
        top_k_lorsa: topKLorsa,
        top_k_tc: topKTC,
        specific_layer: specificLayer,
        specific_layer_top_k: specificLayerTopK
      });

      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/tactic_features/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        console.log('✅ Received analysis result:', data);
        console.log('🔍 Specific layer data check:', {
          specific_layer: data.specific_layer,
          has_specific_layer_lorsa: !!data.specific_layer_lorsa,
          specific_layer_lorsa_length: data.specific_layer_lorsa?.length || 0,
          has_specific_layer_tc: !!data.specific_layer_tc,
          specific_layer_tc_length: data.specific_layer_tc?.length || 0,
        });
        setAnalysisResult(data);
      } else {
        const errorText = await response.text();
        setError(`Analysis failed: ${errorText}`);
      }
    } catch (error) {
      console.error('Analysis failed:', error);
      setError('Analysis failed, please check the backend server');
    } finally {
      setIsLoading(false);
    }
  }, [selectedFile, nFens, topKLorsa, topKTC, specificLayer, specificLayerTopK]);

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Global BT4 SAE combo selection (Lorsa / Transcoder), shares backend cache and loading logs */}
      <SaeComboLoader />

      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <FileText className="w-8 h-8" />
          Tactic Features Analysis
        </h1>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Configuration */}
        <div className="space-y-4">

          {/* File upload */}
          <Card>
            <CardHeader>
              <CardTitle>Upload FEN File</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium">Select file (.txt)</label>
                <div className="mt-2 flex items-center gap-2">
                  <Input
                    type="file"
                    accept=".txt"
                    onChange={handleFileChange}
                    className="cursor-pointer"
                  />
                </div>
                {selectedFile && (
                  <div className="mt-2 text-sm text-gray-600">
                    Selected: {selectedFile.name}
                  </div>
                )}
                {error && (
                  <div className="mt-2 text-sm text-red-600">{error}</div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Parameter configuration */}
          <Card>
            <CardHeader>
              <CardTitle>Analysis Parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium">Number of FENs</label>
                <Input
                  type="number"
                  min="1"
                  max="1000"
                  value={nFens}
                  onChange={(e) => setNFens(parseInt(e.target.value) || 200)}
                  className="mt-1"
                />
                <div className="text-xs text-gray-500 mt-1">
                  Take this many from txt file and random FENs (if the file has less FENs than this number, use all of them)
                </div>
              </div>
              <div>
                <label className="text-sm font-medium">Display Top K Lorsa Features</label>
                <Input
                  type="number"
                  min="1"
                  max="100"
                  value={topKLorsa}
                  onChange={(e) => setTopKLorsa(parseInt(e.target.value) || 10)}
                  className="mt-1"
                />
              </div>
              <div>
                <label className="text-sm font-medium">Display Top K TC Features</label>
                <Input
                  type="number"
                  min="1"
                  max="100"
                  value={topKTC}
                  onChange={(e) => setTopKTC(parseInt(e.target.value) || 10)}
                  className="mt-1"
                />
              </div>
              <div className="border-t pt-4">
                <label className="text-sm font-medium">Specific Layer Analysis (optional)</label>
                <Input
                  type="number"
                  min="0"
                  max="14"
                  value={specificLayer}
                  onChange={(e) => setSpecificLayer(e.target.value)}
                  placeholder="Leave empty to not analyze specific layers"
                  className="mt-1"
                />
                <div className="text-xs text-gray-500 mt-1">
                  Input layer number (0-14) to get detailed features of that layer
                </div>
              </div>
              <div>
                <label className="text-sm font-medium">Number of Top K Features for Specific Layer</label>
                <Input
                  type="number"
                  min="1"
                  max="100"
                  value={specificLayerTopK}
                  onChange={(e) => setSpecificLayerTopK(parseInt(e.target.value) || 20)}
                  className="mt-1"
                />
              </div>
            </CardContent>
          </Card>

          {/* Run button */}
          <Button
            onClick={runAnalysis}
            disabled={isLoading || !selectedFile}
            className="w-full"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Upload className="w-4 h-4 mr-2" />
                Start Analysis
              </>
            )}
          </Button>
        </div>

        {/* Right: Result display */}
        <div className="lg:col-span-2 space-y-4">
          {analysisResult ? (
            <>
              {/* Statistics */}
              <Card>
                <CardHeader>
                  <CardTitle>Analysis Statistics</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-sm text-gray-600">Valid Tactic FENs</div>
                      <div className="text-2xl font-bold">{analysisResult.valid_tactic_fens}</div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-600">Invalid Tactic FENs</div>
                      <div className="text-2xl font-bold text-red-600">{analysisResult.invalid_tactic_fens}</div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-600">Random FENs</div>
                      <div className="text-2xl font-bold">{analysisResult.random_fens}</div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-600">Processed Tactic FENs</div>
                      <div className="text-2xl font-bold">{analysisResult.tactic_fens}</div>
                    </div>
                  </div>
                  {analysisResult.invalid_fens_sample.length > 0 && (
                    <div className="mt-4">
                      <div className="text-sm text-gray-600">Invalid FEN Examples:</div>
                      <div className="text-xs font-mono bg-gray-100 p-2 rounded mt-1">
                        {analysisResult.invalid_fens_sample.slice(0, 5).join(', ')}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Lorsa Features */}
              <Card>
                <CardHeader>
                  <CardTitle>Top {topKLorsa} Lorsa Features (largest difference)</CardTitle>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Rank</TableHead>
                        <TableHead>Layer</TableHead>
                        <TableHead>Feature Index</TableHead>
                        <TableHead>Difference (p_tactic - p_random)</TableHead>
                        <TableHead>p_random</TableHead>
                        <TableHead>p_tactic</TableHead>
                        <TableHead>Operation</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {analysisResult.top_lorsa_features.map((feat, idx) => {
                        const dictionary = buildDictionaryName(feat.layer, 'Lorsa');
                        const featureUrl = `/features?dictionary=${encodeURIComponent(dictionary)}&featureIndex=${feat.feature}`;
                        return (
                          <TableRow key={idx}>
                            <TableCell className="font-medium">#{idx + 1}</TableCell>
                            <TableCell>Layer {feat.layer}</TableCell>
                            <TableCell>
                              <Badge variant="outline">Feature {feat.feature}</Badge>
                            </TableCell>
                            <TableCell className="font-bold text-green-600">
                              {feat.diff.toFixed(6)}
                            </TableCell>
                            <TableCell>{feat.p_random.toFixed(6)}</TableCell>
                            <TableCell>{feat.p_tactic.toFixed(6)}</TableCell>
                            <TableCell>
                              <Link
                                to={featureUrl}
                                className="inline-flex items-center px-2 py-1 bg-blue-500 text-white text-xs font-medium rounded hover:bg-blue-600 transition-colors"
                                title={`View Layer ${feat.layer} Lorsa Feature #${feat.feature}`}
                              >
                                <ExternalLink className="w-3 h-3 mr-1" />
                                View
                              </Link>
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>

              {/* TC Features */}
              <Card>
                <CardHeader>
                  <CardTitle>Top {topKTC} TC Features (largest difference)</CardTitle>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Rank</TableHead>
                        <TableHead>Layer</TableHead>
                        <TableHead>Feature Index</TableHead>
                        <TableHead>Difference (p_tactic - p_random)</TableHead>
                        <TableHead>p_random</TableHead>
                        <TableHead>p_tactic</TableHead>
                        <TableHead>Operation</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {analysisResult.top_tc_features.map((feat, idx) => {
                        const dictionary = buildDictionaryName(feat.layer, 'TC');
                        const featureUrl = `/features?dictionary=${encodeURIComponent(dictionary)}&featureIndex=${feat.feature}`;
                        return (
                          <TableRow key={idx}>
                            <TableCell className="font-medium">#{idx + 1}</TableCell>
                            <TableCell>Layer {feat.layer}</TableCell>
                            <TableCell>
                              <Badge variant="outline">Feature {feat.feature}</Badge>
                            </TableCell>
                            <TableCell className="font-bold text-green-600">
                              {feat.diff.toFixed(6)}
                            </TableCell>
                            <TableCell>{feat.p_random.toFixed(6)}</TableCell>
                            <TableCell>{feat.p_tactic.toFixed(6)}</TableCell>
                            <TableCell>
                              <Link
                                to={featureUrl}
                                className="inline-flex items-center px-2 py-1 bg-blue-500 text-white text-xs font-medium rounded hover:bg-blue-600 transition-colors"
                                title={`View Layer ${feat.layer} TC Feature #${feat.feature}`}
                              >
                                <ExternalLink className="w-3 h-3 mr-1" />
                                View
                              </Link>
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>

              {/* Specific Layer Lorsa Features */}
              {analysisResult.specific_layer !== undefined && analysisResult.specific_layer !== null && (
                <Card className="border-2 border-purple-200">
                  <CardHeader className="bg-purple-50">
                    <CardTitle>Layer {analysisResult.specific_layer} - Top {specificLayerTopK} Lorsa Features</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {analysisResult.specific_layer_lorsa && analysisResult.specific_layer_lorsa.length > 0 ? (
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Rank</TableHead>
                            <TableHead>Feature Index</TableHead>
                            <TableHead>Difference (p_tactic - p_random)</TableHead>
                            <TableHead>p_random</TableHead>
                            <TableHead>p_tactic</TableHead>
                            <TableHead>Operation</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {analysisResult.specific_layer_lorsa.map((feat, idx) => {
                          const dictionary = buildDictionaryName(feat.layer, 'Lorsa');
                          const featureUrl = `/features?dictionary=${encodeURIComponent(dictionary)}&featureIndex=${feat.feature}`;
                          return (
                            <TableRow key={idx}>
                              <TableCell className="font-medium">#{idx + 1}</TableCell>
                              <TableCell>
                                <Badge variant="outline">Feature {feat.feature}</Badge>
                              </TableCell>
                              <TableCell className="font-bold text-purple-600">
                                {feat.diff.toFixed(6)}
                              </TableCell>
                              <TableCell>{feat.p_random.toFixed(6)}</TableCell>
                              <TableCell>{feat.p_tactic.toFixed(6)}</TableCell>
                              <TableCell>
                                <Link
                                  to={featureUrl}
                                  className="inline-flex items-center px-2 py-1 bg-blue-500 text-white text-xs font-medium rounded hover:bg-blue-600 transition-colors"
                                  title={`View Layer ${feat.layer} Lorsa Feature #${feat.feature}`}
                                >
                                  <ExternalLink className="w-3 h-3 mr-1" />
                                  View
                                </Link>
                              </TableCell>
                            </TableRow>
                          );
                        })}
                        </TableBody>
                      </Table>
                    ) : (
                      <div className="text-center py-8 text-gray-500">
                        <p>Layer {analysisResult.specific_layer} did not find any Lorsa features</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}

              {/* Specific Layer TC Features */}
              {analysisResult.specific_layer !== undefined && analysisResult.specific_layer !== null && (
                <Card className="border-2 border-purple-200">
                  <CardHeader className="bg-purple-50">
                    <CardTitle>Layer {analysisResult.specific_layer} - Top {specificLayerTopK} TC Features</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {analysisResult.specific_layer_tc && analysisResult.specific_layer_tc.length > 0 ? (
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Rank</TableHead>
                            <TableHead>Feature Index</TableHead>
                            <TableHead>Difference (p_tactic - p_random)</TableHead>
                            <TableHead>p_random</TableHead>
                            <TableHead>p_tactic</TableHead>
                            <TableHead>Operation</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {analysisResult.specific_layer_tc.map((feat, idx) => {
                          const dictionary = buildDictionaryName(feat.layer, 'TC');
                          const featureUrl = `/features?dictionary=${encodeURIComponent(dictionary)}&featureIndex=${feat.feature}`;
                          return (
                            <TableRow key={idx}>
                              <TableCell className="font-medium">#{idx + 1}</TableCell>
                              <TableCell>
                                <Badge variant="outline">Feature {feat.feature}</Badge>
                              </TableCell>
                              <TableCell className="font-bold text-purple-600">
                                {feat.diff.toFixed(6)}
                              </TableCell>
                              <TableCell>{feat.p_random.toFixed(6)}</TableCell>
                              <TableCell>{feat.p_tactic.toFixed(6)}</TableCell>
                              <TableCell>
                                <Link
                                  to={featureUrl}
                                  className="inline-flex items-center px-2 py-1 bg-blue-500 text-white text-xs font-medium rounded hover:bg-blue-600 transition-colors"
                                  title={`View Layer ${feat.layer} TC Feature #${feat.feature}`}
                                >
                                  <ExternalLink className="w-3 h-3 mr-1" />
                                  View
                                </Link>
                              </TableCell>
                            </TableRow>
                          );
                        })}
                        </TableBody>
                      </Table>
                    ) : (
                      <div className="text-center py-8 text-gray-500">
                        <p>Layer {analysisResult.specific_layer} did not find any TC features</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}
            </>
          ) : (
            <Card>
              <CardContent className="py-12">
                <div className="text-center text-gray-500">
                  <FileText className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>Upload FEN file and click "Start Analysis" to start tactic features analysis</p>
                  <p className="text-xs mt-2">The file should be in .txt format, one FEN string per line</p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};
