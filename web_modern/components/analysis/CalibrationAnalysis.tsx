'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import {
  Target,
  TrendingUp,
  TrendingDown,
  BarChart3,
  Activity,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Info
} from 'lucide-react';

/**
 * 🎯 Enhanced Calibration Analysis Component
 * 
 * Comprehensive calibration metrics and analysis based on the Streamlit version:
 * - Expected Calibration Error (ECE) calculation
 * - Confidence vs Accuracy curves
 * - Bin-wise calibration analysis
 * - Confidence distribution tracking
 * - Performance heatmaps
 */

interface CalibrationData {
  confidence: number;
  model_correct: boolean;
  timestamp: string;
  prompt_id: string;
}

interface CalibrationMetrics {
  ece: number;
  avgConfidence: number;
  accuracy: number;
  gap: number;
  binStats: CalibrationBin[];
}

interface CalibrationBin {
  bin: string;
  samples: number;
  avgConfidence: number;
  accuracy: number;
  absError: number;
  weight: number;
  contribECE: number;
}

export default function CalibrationAnalysis() {
  const [data, setData] = useState<CalibrationData[]>([]);
  const [metrics, setMetrics] = useState<CalibrationMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState('7d');

  // Mock data generation (replace with real API calls)
  useEffect(() => {
    const generateMockData = () => {
      const mockData: CalibrationData[] = [];
      
      for (let i = 0; i < 500; i++) {
        const confidence = Math.random();
        // Make model_correct probability correlated with confidence but not perfectly
        const correctProb = 0.3 + (confidence * 0.6) + (Math.random() * 0.2 - 0.1);
        const model_correct = Math.random() < correctProb;
        
        mockData.push({
          confidence,
          model_correct,
          timestamp: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
          prompt_id: `prompt_${i}`
        });
      }
      
      return mockData;
    };

    const calculateCalibrationMetrics = (data: CalibrationData[]): CalibrationMetrics => {
      const avgConfidence = data.reduce((sum, d) => sum + d.confidence, 0) / data.length;
      const accuracy = data.filter(d => d.model_correct).length / data.length;
      const gap = avgConfidence - accuracy;

      // Calculate ECE using confidence bins
      const bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
      const binLabels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', 
                        '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'];
      
      const binStats: CalibrationBin[] = [];
      let ece = 0;

      for (let i = 0; i < bins.length - 1; i++) {
        const binData = data.filter(d => d.confidence >= bins[i] && d.confidence < bins[i + 1]);
        
        if (binData.length > 0) {
          const binAvgConf = binData.reduce((sum, d) => sum + d.confidence, 0) / binData.length;
          const binAccuracy = binData.filter(d => d.model_correct).length / binData.length;
          const absError = Math.abs(binAvgConf - binAccuracy);
          const weight = binData.length / data.length;
          const contribECE = absError * weight;
          
          binStats.push({
            bin: binLabels[i],
            samples: binData.length,
            avgConfidence: binAvgConf,
            accuracy: binAccuracy,
            absError,
            weight,
            contribECE
          });
          
          ece += contribECE;
        }
      }

      return {
        ece,
        avgConfidence,
        accuracy,
        gap,
        binStats
      };
    };

    setTimeout(() => {
      const mockData = generateMockData();
      setData(mockData);
      setMetrics(calculateCalibrationMetrics(mockData));
      setLoading(false);
    }, 1000);
  }, [timeRange]);

  const getCalibrationStatus = (ece: number) => {
    if (ece < 0.01) return { status: 'excellent', color: 'green', icon: CheckCircle, text: 'Exceptionally well-calibrated' };
    if (ece < 0.05) return { status: 'good', color: 'blue', icon: CheckCircle, text: 'Good calibration' };
    if (ece < 0.10) return { status: 'needs-improvement', color: 'yellow', icon: AlertTriangle, text: 'Needs calibration improvement' };
    return { status: 'poor', color: 'red', icon: XCircle, text: 'Significant calibration improvement needed' };
  };

  const getGapStatus = (gap: number) => {
    if (Math.abs(gap) < 0.05) return { status: 'well-calibrated', icon: CheckCircle, color: 'green' };
    if (gap > 0) return { status: 'overconfident', icon: TrendingUp, color: 'orange' };
    return { status: 'underconfident', icon: TrendingDown, color: 'blue' };
  };

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>🎯 Calibration Analysis</CardTitle>
          <CardDescription>Loading calibration metrics...</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="animate-pulse space-y-2">
              <div className="h-4 bg-muted rounded w-3/4"></div>
              <div className="h-4 bg-muted rounded w-1/2"></div>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!metrics) return null;

  const calibrationStatus = getCalibrationStatus(metrics.ece);
  const gapStatus = getGapStatus(metrics.gap);
  const StatusIcon = calibrationStatus.icon;
  const GapIcon = gapStatus.icon;

  return (
    <div className="space-y-6">
      {/* Overview Metrics */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Expected Calibration Error</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{(metrics.ece * 100).toFixed(2)}%</div>
            <div className="flex items-center space-x-2 mt-2">
              <StatusIcon className={`h-4 w-4 text-${calibrationStatus.color}-500`} />
              <p className="text-xs text-muted-foreground">
                {calibrationStatus.text}
              </p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Mean Confidence</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{(metrics.avgConfidence * 100).toFixed(1)}%</div>
            <Progress value={metrics.avgConfidence * 100} className="mt-2" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Model Accuracy</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{(metrics.accuracy * 100).toFixed(1)}%</div>
            <Progress value={metrics.accuracy * 100} className="mt-2" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Confidence-Accuracy Gap</CardTitle>
            <GapIcon className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {metrics.gap > 0 ? '+' : ''}{(metrics.gap * 100).toFixed(1)}%
            </div>
            <div className="flex items-center space-x-2 mt-2">
              <Badge variant={gapStatus.status === 'well-calibrated' ? 'default' : 'secondary'}>
                {gapStatus.status === 'well-calibrated' && 'Well-calibrated'}
                {gapStatus.status === 'overconfident' && 'Overconfident'}
                {gapStatus.status === 'underconfident' && 'Underconfident'}
              </Badge>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Analysis Tabs */}
      <Tabs defaultValue="calibration-curve" className="space-y-4">
        <TabsList>
          <TabsTrigger value="calibration-curve">Calibration Curve</TabsTrigger>
          <TabsTrigger value="confidence-distribution">Confidence Distribution</TabsTrigger>
          <TabsTrigger value="bin-analysis">Bin Analysis</TabsTrigger>
          <TabsTrigger value="performance-heatmap">Performance Heatmap</TabsTrigger>
        </TabsList>

        <TabsContent value="calibration-curve" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Calibration Curve Analysis</CardTitle>
              <CardDescription>
                Model confidence vs actual accuracy across different confidence levels
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg flex items-center justify-center">
                <div className="text-center">
                  <Target className="h-12 w-12 text-blue-500 mx-auto mb-4" />
                  <p className="text-lg font-semibold">Calibration Curve</p>
                  <p className="text-sm text-muted-foreground">
                    Perfect calibration line vs model performance
                  </p>
                  <div className="mt-4 space-y-2">
                    <div className="flex items-center justify-center space-x-4 text-sm">
                      <div className="flex items-center space-x-2">
                        <div className="w-3 h-3 bg-blue-500 rounded"></div>
                        <span>Model Performance</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <div className="w-3 h-3 bg-gray-400 rounded"></div>
                        <span>Perfect Calibration</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="confidence-distribution" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Confidence Distribution</CardTitle>
              <CardDescription>
                Distribution of model confidence scores across predictions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="h-[300px] bg-gradient-to-br from-green-50 to-blue-50 rounded-lg flex items-center justify-center">
                  <div className="text-center">
                    <BarChart3 className="h-8 w-8 text-green-500 mx-auto mb-2" />
                    <p className="font-semibold">Confidence Histogram</p>
                    <p className="text-sm text-muted-foreground">All Predictions</p>
                  </div>
                </div>
                <div className="h-[300px] bg-gradient-to-br from-red-50 to-orange-50 rounded-lg flex items-center justify-center">
                  <div className="text-center">
                    <BarChart3 className="h-8 w-8 text-red-500 mx-auto mb-2" />
                    <p className="font-semibold">By Correctness</p>
                    <p className="text-sm text-muted-foreground">Correct vs Incorrect</p>
                  </div>
                </div>
              </div>

              <Separator className="my-4" />

              <div className="grid gap-4 md:grid-cols-2">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Correct Predictions</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-lg font-bold text-green-600">
                      {(data.filter(d => d.model_correct).reduce((sum, d) => sum + d.confidence, 0) / 
                        data.filter(d => d.model_correct).length * 100).toFixed(1)}%
                    </div>
                    <p className="text-xs text-muted-foreground">Average confidence</p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Incorrect Predictions</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-lg font-bold text-red-600">
                      {(data.filter(d => !d.model_correct).reduce((sum, d) => sum + d.confidence, 0) / 
                        data.filter(d => !d.model_correct).length * 100).toFixed(1)}%
                    </div>
                    <p className="text-xs text-muted-foreground">Average confidence</p>
                  </CardContent>
                </Card>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="bin-analysis" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Bin-wise Calibration Analysis</CardTitle>
              <CardDescription>
                Detailed breakdown of calibration performance across confidence bins
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {metrics.binStats.map((bin, index) => (
                  <div key={index} className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <div className="font-medium">{bin.bin}</div>
                      <Badge variant={bin.absError < 0.1 ? 'default' : 'secondary'}>
                        {bin.samples} samples
                      </Badge>
                    </div>
                    
                    <div className="grid gap-2 md:grid-cols-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">Avg Confidence:</span>
                        <div className="font-medium">{(bin.avgConfidence * 100).toFixed(1)}%</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Accuracy:</span>
                        <div className="font-medium">{(bin.accuracy * 100).toFixed(1)}%</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Abs Error:</span>
                        <div className="font-medium">{(bin.absError * 100).toFixed(1)}%</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">ECE Contribution:</span>
                        <div className="font-medium">{(bin.contribECE * 100).toFixed(2)}%</div>
                      </div>
                    </div>
                    
                    <Progress 
                      value={(1 - bin.absError) * 100} 
                      className="mt-2" 
                    />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="performance-heatmap" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Confidence vs Correctness Heatmap</CardTitle>
              <CardDescription>
                2D visualization of confidence and prediction accuracy relationship
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg flex items-center justify-center">
                <div className="text-center">
                  <Activity className="h-12 w-12 text-purple-500 mx-auto mb-4" />
                  <p className="text-lg font-semibold">Performance Heatmap</p>
                  <p className="text-sm text-muted-foreground">
                    Confidence bins vs correctness distribution
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Action Buttons */}
      <Card>
        <CardHeader>
          <CardTitle>Calibration Actions</CardTitle>
          <CardDescription>
            Tools to improve model calibration based on current analysis
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex space-x-2">
            <Button>
              <Target className="h-4 w-4 mr-2" />
              Run Calibration Training
            </Button>
            <Button variant="outline">
              Export Calibration Report
            </Button>
            <Button variant="outline">
              View Historical Trends
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
} 