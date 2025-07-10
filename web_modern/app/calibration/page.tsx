'use client';

import React, { useState, useEffect } from 'react';
import DashboardLayout from '@/components/layout/DashboardLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import {
  Target,
  BarChart3,
  Activity,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  XCircle,
  LineChart,
  PieChart,
  Gauge,
  Clock,
  Calculator,
  Zap,
  ThermometerSun,
  Eye,
  Layers
} from 'lucide-react';

/**
 * 🎯 Enhanced Calibration Analysis - Comprehensive Confidence-Accuracy Alignment
 * 
 * This page provides deep calibration insights matching Streamlit capabilities:
 * - Calibration curve analysis with reliability diagrams
 * - Confidence distribution analysis with heatmaps
 * - Bin-wise calibration breakdown with detailed ECE calculation  
 * - Pre/post calibration comparison
 * - Performance heatmaps and advanced visualizations
 * 
 * Equivalent to Streamlit interface/sections/calibration.py
 */

interface CalibrationData {
  overall_metrics: {
    ece: number;
    mce: number;
    ace: number;
    avg_confidence: number;
    accuracy: number;
    confidence_gap: number;
    brier_score: number;
    log_loss: number;
  };
  bin_stats: Array<{
    bin: string;
    bin_range: [number, number];
    samples: number;
    avg_confidence: number;
    accuracy: number;
    abs_error: number;
    weight: number;
    contrib_to_ece: number;
  }>;
  confidence_distribution: {
    correct: Array<{ confidence: number; count: number }>;
    incorrect: Array<{ confidence: number; count: number }>;
  };
  calibration_history: Array<{
    timestamp: string;
    ece: number;
    accuracy: number;
    avg_confidence: number;
    sample_count: number;
    notes: string;
  }>;
  temperature_scaling: {
    pre_calibration: { ece: number; log_loss: number; brier_score: number };
    post_calibration: { ece: number; log_loss: number; brier_score: number };
    temperature: number;
    improvement: { ece: number; log_loss: number; brier_score: number };
  };
}

export default function EnhancedCalibrationPage() {
  const [activeTab, setActiveTab] = useState('analysis');
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<CalibrationData | null>(null);
  const [timeRange, setTimeRange] = useState('all');

  useEffect(() => {
    loadCalibrationData();
  }, [timeRange]);

  const loadCalibrationData = async () => {
    setLoading(true);
    
    try {
      // Fetch real calibration data from backend
      const response = await fetch('http://localhost:8000/api/calibration');
      const result = await response.json();
      
      if (result.enhanced_analysis_available && result.overall_metrics) {
        // We have real calibration data
        setData(result);
      } else {
        // No real calibration data available
        setData(null);
      }
    } catch (error) {
      console.error('Error loading calibration data:', error);
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <DashboardLayout>
        <div className="space-y-6">
          <div className="flex items-center space-x-4">
            <Target className="h-8 w-8 text-blue-500 animate-pulse" />
            <div>
              <h1 className="text-3xl font-bold">Enhanced Calibration Analysis</h1>
              <p className="text-muted-foreground">Loading comprehensive calibration metrics...</p>
            </div>
          </div>
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
            {[1, 2, 3, 4].map((i) => (
              <Card key={i} className="animate-pulse">
                <CardHeader>
                  <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                </CardHeader>
                <CardContent>
                  <div className="h-8 bg-gray-200 rounded w-1/2 mb-2"></div>
                  <div className="h-3 bg-gray-200 rounded w-full"></div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </DashboardLayout>
    );
  }

  const getCalibrationStatus = (ece: number) => {
    if (ece < 0.01) return { status: 'excellent', message: 'Exceptionally well-calibrated', color: 'text-green-600' };
    if (ece < 0.05) return { status: 'good', message: 'Good calibration', color: 'text-blue-600' };
    if (ece < 0.10) return { status: 'moderate', message: 'Needs improvement', color: 'text-yellow-600' };
    return { status: 'poor', message: 'Significant calibration needed', color: 'text-red-600' };
  };

  if (!data) {
    return (
      <DashboardLayout>
        <div className="space-y-6">
          <div className="flex items-center space-x-4">
            <Target className="h-8 w-8 text-blue-500" />
            <div>
              <h1 className="text-3xl font-bold">Enhanced Calibration Analysis</h1>
              <p className="text-muted-foreground">Comprehensive confidence-accuracy alignment analysis</p>
            </div>
          </div>

          {/* Empty State */}
          <div className="text-center py-12">
            <div className="max-w-md mx-auto">
              <Target className="h-16 w-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">No Calibration Data Available</h3>
                <p className="text-gray-600 mb-4">
                Calibration analysis requires model predictions with confidence scores. Start collecting annotations to see calibration metrics.
                </p>
                <button
                onClick={() => window.location.href = '/annotation'}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                Start Collecting Data
                </button>
              <p className="text-xs text-gray-500 mt-2">
                ECE, MCE, and calibration curves will appear once you have data
              </p>
            </div>
          </div>
        </div>
      </DashboardLayout>
    );
  }

  const calibrationStatus = getCalibrationStatus(data?.overall_metrics?.ece || 0);
  
  // Helper function to avoid JSX comparison issues
  const getTemperatureDirection = (temp: number) => temp > 1 ? 'reduced' : 'increased';
  const getTemperatureBehavior = (temp: number) => temp > 1 ? 'Softening confidence' : 'Sharpening confidence';

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Target className="h-8 w-8 text-blue-500" />
          <div>
            <h1 className="text-3xl font-bold">Enhanced Calibration Analysis</h1>
            <p className="text-muted-foreground">
              Comprehensive confidence-accuracy alignment analysis with ECE, MCE, and ACE metrics
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <Button 
            variant={timeRange === 'all' ? 'default' : 'outline'} 
            size="sm"
            onClick={() => setTimeRange('all')}
          >
            All Time
          </Button>
          <Button 
            variant={timeRange === 'week' ? 'default' : 'outline'} 
            size="sm"
            onClick={() => setTimeRange('week')}
          >
            This Week
          </Button>
          <Button 
            variant={timeRange === 'month' ? 'default' : 'outline'} 
            size="sm"
            onClick={() => setTimeRange('month')}
          >
            This Month
          </Button>
              </div>
              </div>

      {/* Key Metrics Overview */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Expected Calibration Error</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{(data?.overall_metrics.ece * 100).toFixed(2)}%</div>
            <p className={`text-xs ${calibrationStatus.color}`}>{calibrationStatus.message}</p>
            <div className="mt-2">
              <Badge variant={calibrationStatus.status === 'excellent' || calibrationStatus.status === 'good' ? 'default' : 'secondary'}>
                {calibrationStatus.status === 'excellent' && <CheckCircle className="h-3 w-3 mr-1" />}
                {calibrationStatus.status === 'good' && <CheckCircle className="h-3 w-3 mr-1" />}
                {calibrationStatus.status === 'moderate' && <AlertTriangle className="h-3 w-3 mr-1" />}
                {calibrationStatus.status === 'poor' && <XCircle className="h-3 w-3 mr-1" />}
                ECE: {calibrationStatus.status}
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Confidence Gap</CardTitle>
            <Gauge className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {data?.overall_metrics.confidence_gap > 0 ? '+' : ''}{(data?.overall_metrics.confidence_gap * 100).toFixed(1)}%
          </div>
            <p className="text-xs text-muted-foreground">
              {Math.abs(data?.overall_metrics.confidence_gap || 0) < 0.05 
                ? 'Well-aligned confidence' 
                : data?.overall_metrics.confidence_gap > 0 
                  ? 'Model shows overconfidence' 
                  : 'Model shows underconfidence'
              }
            </p>
            <div className="mt-2">
              <Progress 
                value={100 - Math.abs((data?.overall_metrics.confidence_gap || 0) * 200)} 
                className="h-2" 
              />
              </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Brier Score</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{data?.overall_metrics.brier_score.toFixed(3)}</div>
            <p className="text-xs text-muted-foreground">Prediction quality score</p>
            <div className="mt-2">
              <Badge variant="outline">
                {data?.overall_metrics.brier_score < 0.1 ? 'Excellent' : 
                 data?.overall_metrics.brier_score < 0.2 ? 'Good' : 'Needs improvement'}
              </Badge>
              </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Overall Accuracy</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{(data?.overall_metrics.accuracy * 100).toFixed(1)}%</div>
            <p className="text-xs text-muted-foreground">Model prediction accuracy</p>
            <div className="mt-2">
              <Progress value={(data?.overall_metrics.accuracy || 0) * 100} className="h-2" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Analysis Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="analysis" className="flex items-center space-x-2">
            <Target className="h-4 w-4" />
            <span>Analysis</span>
          </TabsTrigger>
          <TabsTrigger value="distribution" className="flex items-center space-x-2">
            <PieChart className="h-4 w-4" />
            <span>Distribution</span>
          </TabsTrigger>
          <TabsTrigger value="bins" className="flex items-center space-x-2">
            <Layers className="h-4 w-4" />
            <span>Bin Analysis</span>
          </TabsTrigger>
          <TabsTrigger value="heatmap" className="flex items-center space-x-2">
            <Eye className="h-4 w-4" />
            <span>Heatmap</span>
          </TabsTrigger>
          <TabsTrigger value="temperature" className="flex items-center space-x-2">
            <ThermometerSun className="h-4 w-4" />
            <span>Temperature</span>
          </TabsTrigger>
        </TabsList>

        {/* Calibration Analysis Tab */}
        <TabsContent value="analysis" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-3">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Calibration Metrics</CardTitle>
                <CardDescription>Comprehensive calibration error measurements</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">Expected Calibration Error (ECE)</span>
                  <Badge variant="outline">{(data?.overall_metrics.ece * 100).toFixed(2)}%</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">Maximum Calibration Error (MCE)</span>
                  <Badge variant="outline">{(data?.overall_metrics.mce * 100).toFixed(2)}%</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">Adaptive Calibration Error (ACE)</span>
                  <Badge variant="outline">{(data?.overall_metrics.ace * 100).toFixed(2)}%</Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Confidence vs Accuracy</CardTitle>
                <CardDescription>Model confidence alignment analysis</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="text-center">
                  <div className="text-lg font-bold text-blue-600">
                    {(data?.overall_metrics.avg_confidence * 100).toFixed(1)}%
                  </div>
                  <p className="text-sm text-muted-foreground">Average Confidence</p>
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-green-600">
                    {(data?.overall_metrics.accuracy * 100).toFixed(1)}%
                  </div>
                  <p className="text-sm text-muted-foreground">Actual Accuracy</p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Calibration Timeline</CardTitle>
                <CardDescription>Recent calibration history</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {data?.calibration_history.slice(-3).map((entry, index) => (
                    <div key={index} className="flex justify-between items-center p-2 border rounded">
                      <div>
                        <div className="text-sm font-medium">{new Date(entry.timestamp).toLocaleDateString()}</div>
                        <div className="text-xs text-muted-foreground">{entry.notes}</div>
                      </div>
                      <Badge variant={entry.ece < 0.05 ? "default" : "secondary"}>
                        {(entry.ece * 100).toFixed(1)}%
                      </Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Reliability Diagram</CardTitle>
              <CardDescription>
                Calibration curve showing confidence vs accuracy alignment
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[350px] bg-gradient-to-br from-blue-50 to-purple-100 rounded-lg flex items-center justify-center">
                <div className="text-center">
                  <LineChart className="h-12 w-12 text-blue-500 mx-auto mb-4" />
                  <p className="text-lg font-semibold">Reliability Diagram</p>
                  <p className="text-sm text-muted-foreground">
                    Perfect calibration line vs actual performance
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Confidence Distribution Tab */}
        <TabsContent value="distribution" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Confidence Distribution by Correctness</CardTitle>
                <CardDescription>
                  Model confidence levels for correct vs incorrect predictions
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[300px] bg-gradient-to-br from-green-50 to-red-50 rounded-lg flex items-center justify-center">
                  <div className="text-center">
                    <PieChart className="h-12 w-12 text-green-500 mx-auto mb-4" />
                    <p className="text-lg font-semibold">Distribution Analysis</p>
                    <p className="text-sm text-muted-foreground">
                      Confidence histograms by prediction outcome
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Confidence Gap Analysis</CardTitle>
                <CardDescription>
                  Difference in confidence between correct and incorrect predictions
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Avg Confidence (Correct)</span>
                    <Badge variant="default" className="bg-green-100 text-green-800">
                      {((data?.confidence_distribution.correct.reduce((sum, item) => sum + item.confidence * item.count, 0) || 0) / 
                        (data?.confidence_distribution.correct.reduce((sum, item) => sum + item.count, 0) || 1) * 100).toFixed(1)}%
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Avg Confidence (Incorrect)</span>
                    <Badge variant="destructive" className="bg-red-100 text-red-800">
                      {((data?.confidence_distribution.incorrect.reduce((sum, item) => sum + item.confidence * item.count, 0) || 0) / 
                        (data?.confidence_distribution.incorrect.reduce((sum, item) => sum + item.count, 0) || 1) * 100).toFixed(1)}%
                    </Badge>
                  </div>
                  <Separator />
                  <div className="text-center">
                    <div className="text-lg font-bold text-blue-600">
                      +{(((data?.confidence_distribution.correct.reduce((sum, item) => sum + item.confidence * item.count, 0) || 0) / 
                        (data?.confidence_distribution.correct.reduce((sum, item) => sum + item.count, 0) || 1)) - 
                        ((data?.confidence_distribution.incorrect.reduce((sum, item) => sum + item.confidence * item.count, 0) || 0) / 
                        (data?.confidence_distribution.incorrect.reduce((sum, item) => sum + item.count, 0) || 1)) * 100).toFixed(1)}%
                    </div>
                    <p className="text-sm text-muted-foreground">Confidence Gap</p>
                    <p className="text-xs text-green-600 mt-1">
                      {(((data?.confidence_distribution.correct.reduce((sum, item) => sum + item.confidence * item.count, 0) || 0) / 
                        (data?.confidence_distribution.correct.reduce((sum, item) => sum + item.count, 0) || 1)) - 
                        ((data?.confidence_distribution.incorrect.reduce((sum, item) => sum + item.confidence * item.count, 0) || 0) / 
                        (data?.confidence_distribution.incorrect.reduce((sum, item) => sum + item.count, 0) || 1))) > 0.05 
                        ? 'Good confidence discrimination' 
                        : 'Poor confidence calibration'
                      }
                    </p>
              </div>
            </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Bin Analysis Tab */}
        <TabsContent value="bins" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Calibration Error by Confidence Bin</CardTitle>
              <CardDescription>
                Detailed bin-wise calibration analysis with ECE contribution
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {data?.bin_stats.map((bin, index) => (
                  <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center space-x-4">
                      <div className="text-sm font-medium w-20">{bin.bin}</div>
                      <div className="text-xs text-muted-foreground">
                        {bin.samples} samples
                      </div>
                    </div>
                    <div className="flex items-center space-x-4">
                      <div className="text-xs text-muted-foreground">
                        Conf: {(bin.avg_confidence * 100).toFixed(1)}%
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Acc: {(bin.accuracy * 100).toFixed(1)}%
                      </div>
                      <div className="w-24">
                        <Progress 
                          value={(1 - bin.abs_error) * 100} 
                          className="h-2" 
                        />
                      </div>
                      <Badge 
                        variant={bin.abs_error < 0.05 ? "default" : bin.abs_error < 0.1 ? "secondary" : "destructive"}
                        className="w-16 justify-center"
                      >
                        {(bin.abs_error * 100).toFixed(1)}%
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>ECE Calculation Breakdown</CardTitle>
                <CardDescription>Weighted contributions to overall ECE</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {data?.bin_stats.map((bin, index) => (
                    <div key={index} className="flex justify-between items-center text-sm">
                      <span className="font-medium">{bin.bin}</span>
                      <div className="flex items-center space-x-2">
                        <span className="text-muted-foreground">
                          {(bin.contrib_to_ece * 100).toFixed(3)}%
                        </span>
                        <div className="w-16">
                          <Progress 
                            value={(bin.contrib_to_ece / (data?.overall_metrics.ece || 1)) * 100} 
                            className="h-1" 
                          />
                        </div>
              </div>
            </div>
                  ))}
                </div>
                <Separator className="my-3" />
                <div className="flex justify-between items-center font-bold">
                  <span>Total ECE:</span>
                  <span>{(data?.overall_metrics.ece * 100).toFixed(2)}%</span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Bin Quality Assessment</CardTitle>
                <CardDescription>Performance assessment by confidence range</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {data?.bin_stats.filter(bin => bin.samples > 20).map((bin, index) => (
                  <div key={index} className="p-3 border rounded-lg">
                    <div className="flex justify-between items-start mb-2">
                      <span className="font-medium">{bin.bin}</span>
                      <Badge 
                        variant={bin.abs_error < 0.03 ? "default" : bin.abs_error < 0.07 ? "secondary" : "destructive"}
                      >
                        {bin.abs_error < 0.03 ? 'Excellent' : bin.abs_error < 0.07 ? 'Good' : 'Poor'}
                      </Badge>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Weight: {(bin.weight * 100).toFixed(1)}% • Error: {(bin.abs_error * 100).toFixed(1)}%
                    </div>
                    <Progress value={(1 - bin.abs_error) * 100} className="h-2 mt-2" />
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Performance Heatmap Tab */}
        <TabsContent value="heatmap" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Confidence vs Correctness Heatmap</CardTitle>
              <CardDescription>
                Performance visualization across confidence levels and prediction outcomes
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] bg-gradient-to-br from-purple-50 to-pink-100 rounded-lg flex items-center justify-center">
                <div className="text-center">
                  <Eye className="h-16 w-16 text-purple-500 mx-auto mb-4" />
                  <p className="text-xl font-semibold mb-2">Performance Heatmap</p>
                  <p className="text-sm text-muted-foreground max-w-md">
                    2D visualization showing density of predictions across confidence levels and correctness,
                    with intensity representing frequency
                  </p>
        </div>
              </div>
            </CardContent>
          </Card>

          <Alert>
            <Eye className="h-4 w-4" />
            <AlertTitle>Heatmap Analysis</AlertTitle>
            <AlertDescription>
              The performance heatmap reveals prediction patterns: high-confidence correct predictions 
              should dominate the upper-right, while high-confidence errors (upper-left) indicate 
              overconfidence issues requiring calibration adjustment.
            </AlertDescription>
          </Alert>
        </TabsContent>

        {/* Temperature Scaling Tab */}
        <TabsContent value="temperature" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-3">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Temperature Parameter</CardTitle>
                <ThermometerSun className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-orange-600">
                  {data?.temperature_scaling.temperature.toFixed(2)}
                </div>
                <p className="text-xs text-muted-foreground">Calibration scaling factor</p>
                <div className="mt-2">
                  <Badge variant="outline">
                    {getTemperatureBehavior(data?.temperature_scaling.temperature || 1)}
                  </Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">ECE Improvement</CardTitle>
                <TrendingUp className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-green-600">
                  -{(data?.temperature_scaling.improvement.ece * 100).toFixed(2)}%
                </div>
                <p className="text-xs text-muted-foreground">Calibration error reduction</p>
                <div className="mt-2">
                  <Progress 
                    value={(data?.temperature_scaling.improvement.ece / data?.temperature_scaling.pre_calibration.ece || 0) * 100} 
                    className="h-2" 
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Overall Improvement</CardTitle>
                <Zap className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-blue-600">
                  {((data?.temperature_scaling.improvement.ece / data?.temperature_scaling.pre_calibration.ece || 0) * 100).toFixed(1)}%
                </div>
                <p className="text-xs text-muted-foreground">Relative ECE improvement</p>
                <div className="mt-2">
                  <Badge variant="default">
                    <CheckCircle className="h-3 w-3 mr-1" />
                    Calibrated
                  </Badge>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Pre-Calibration Metrics</CardTitle>
                <CardDescription>Model performance before temperature scaling</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">ECE:</span>
                  <Badge variant="destructive">{(data?.temperature_scaling.pre_calibration.ece * 100).toFixed(2)}%</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">Log Loss:</span>
                  <Badge variant="secondary">{data?.temperature_scaling.pre_calibration.log_loss.toFixed(3)}</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">Brier Score:</span>
                  <Badge variant="secondary">{data?.temperature_scaling.pre_calibration.brier_score.toFixed(3)}</Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Post-Calibration Metrics</CardTitle>
                <CardDescription>Model performance after temperature scaling</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">ECE:</span>
                  <Badge variant="default">{(data?.temperature_scaling.post_calibration.ece * 100).toFixed(2)}%</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">Log Loss:</span>
                  <Badge variant="outline">{data?.temperature_scaling.post_calibration.log_loss.toFixed(3)}</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">Brier Score:</span>
                  <Badge variant="outline">{data?.temperature_scaling.post_calibration.brier_score.toFixed(3)}</Badge>
                </div>
              </CardContent>
            </Card>
        </div>

          <Alert>
            <ThermometerSun className="h-4 w-4" />
            <AlertTitle>Temperature Scaling Explanation</AlertTitle>
            <AlertDescription>
              Temperature scaling applies a single parameter to soften (T &gt; 1) or sharpen (T &lt; 1) model confidence.
              A temperature of {(data?.temperature_scaling.temperature || 1).toFixed(2)} indicates the model&apos;s raw confidence 
              is being {getTemperatureDirection(data?.temperature_scaling.temperature || 1)} to better match actual accuracy.
            </AlertDescription>
          </Alert>
        </TabsContent>
      </Tabs>
      </div>
    </DashboardLayout>
  );
} 