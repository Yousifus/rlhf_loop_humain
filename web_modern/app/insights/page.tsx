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
  Brain,
  Target,
  BarChart3,
  TrendingUp,
  Activity,
  Clock,
  Settings,
  Lightbulb,
  Zap,
  GitBranch,
  Gauge,
  AlertTriangle,
  CheckCircle,
  XCircle,
  LineChart,
  PieChart,
  Layers,
  Cpu,
  Database,
  Calendar,
  ThermometerSun,
  RotateCcw,
  Eye,
  MessageSquare
} from 'lucide-react';

/**
 * 🧠 Model Insights Dashboard - Comprehensive Model Performance Analysis
 * 
 * This page provides deep insights into:
 * - Model training progress and configuration
 * - Calibration metrics and temperature scaling
 * - Performance analysis and error breakdown
 * - Drift detection and clustering analysis
 * - Model introspection and self-analysis
 * 
 * Equivalent to Streamlit interface/sections/model_insights.py
 */

interface TrainingData {
  model_name: string;
  dataset_size: number;
  training_params: {
    learning_rate: number;
    batch_size: number;
    epochs: number;
    max_length: number;
    weight_decay: number;
    validation_split: number;
    save_steps: number;
    early_stopping_patience: number;
    seed: number;
  };
  timestamp: string;
}

interface CalibrationData {
  metrics: {
    pre_calibration: {
      ece: number;
      log_loss: number;
      brier_score: number;
    };
    post_calibration: {
      ece: number;
      log_loss: number;
      brier_score: number;
    };
    improvement: {
      ece: number;
      log_loss: number;
      brier_score: number;
    };
  };
  parameters: {
    temperature: number;
  };
  method: string;
  history: Array<{
    timestamp: string;
    calibration_error: number;
    sample_count: number;
    accuracy: number;
    avg_confidence_before: number;
    avg_confidence_after: number;
    notes: string;
  }>;
}

interface PerformanceData {
  accuracy: number;
  total_votes: number;
  correct_predictions: number;
  error_counts: {
    high_confidence_error: number;
    low_confidence_error: number;
    misalignment_error: number;
  };
  confidence_distribution: {
    [key: string]: number;
  };
}

interface DriftData {
  overall_drift_score: number;
  time_windows_analyzed: number;
  clusters_identified: number;
  time_based_analysis: {
    drift_over_time: {
      [key: string]: number;
    };
  };
  clustering_analysis: {
    algorithm: string;
    silhouette_score: number;
    cluster_characteristics: {
      [key: string]: {
        description: string;
        size: number;
        accuracy: number;
        avg_confidence: number;
      };
    };
  };
}

interface IntrospectionData {
  reflections_count: number;
  daily_activity: {
    [key: string]: number;
  };
  accuracy_patterns: {
    [key: string]: {
      mean_accuracy: number;
      count: number;
    };
  };
}

export default function ModelInsightsPage() {
  const [activeTab, setActiveTab] = useState('training');
  const [loading, setLoading] = useState(true);
  const [trainingData, setTrainingData] = useState<TrainingData | null>(null);
  const [calibrationData, setCalibrationData] = useState<CalibrationData | null>(null);
  const [performanceData, setPerformanceData] = useState<PerformanceData | null>(null);
  const [driftData, setDriftData] = useState<DriftData | null>(null);
  const [introspectionData, setIntrospectionData] = useState<IntrospectionData | null>(null);

  useEffect(() => {
    loadModelInsights();
  }, []);

  const loadModelInsights = async () => {
    setLoading(true);
    
    // Enhanced mock data matching Streamlit capabilities
    const mockTrainingData: TrainingData = {
      model_name: 'deepseek-chat-v2',
      dataset_size: 1247,
      training_params: {
        learning_rate: 5e-5,
        batch_size: 16,
        epochs: 3,
        max_length: 512,
        weight_decay: 0.01,
        validation_split: 0.2,
        save_steps: 500,
        early_stopping_patience: 3,
        seed: 42
      },
      timestamp: '2024-01-05T14:30:00'
    };

    const mockCalibrationData: CalibrationData = {
      metrics: {
        pre_calibration: {
          ece: 0.087,
          log_loss: 0.342,
          brier_score: 0.156
        },
        post_calibration: {
          ece: 0.023,
          log_loss: 0.234,
          brier_score: 0.089
        },
        improvement: {
          ece: 0.064,
          log_loss: 0.108,
          brier_score: 0.067
        }
      },
      parameters: {
        temperature: 1.47
      },
      method: 'temperature_scaling',
      history: [
        {
          timestamp: '2024-01-01T10:00:00',
          calibration_error: 0.087,
          sample_count: 856,
          accuracy: 0.84,
          avg_confidence_before: 0.91,
          avg_confidence_after: 0.86,
          notes: 'Initial calibration'
        },
        {
          timestamp: '2024-01-03T15:30:00',
          calibration_error: 0.045,
          sample_count: 1124,
          accuracy: 0.88,
          avg_confidence_before: 0.89,
          avg_confidence_after: 0.87,
          notes: 'Post-training calibration'
        },
        {
          timestamp: '2024-01-05T14:30:00',
          calibration_error: 0.023,
          sample_count: 1247,
          accuracy: 0.92,
          avg_confidence_before: 0.89,
          avg_confidence_after: 0.91,
          notes: 'Production calibration'
        }
      ]
    };

    const mockPerformanceData: PerformanceData = {
      accuracy: 0.924,
      total_votes: 1247,
      correct_predictions: 1152,
      error_counts: {
        high_confidence_error: 23,
        low_confidence_error: 42,
        misalignment_error: 30
      },
      confidence_distribution: {
        'low (0-0.3)': 89,
        'medium (0.3-0.7)': 234,
        'high (0.7-1.0)': 924
      }
    };

    const mockDriftData: DriftData = {
      overall_drift_score: 0.127,
      time_windows_analyzed: 12,
      clusters_identified: 5,
      time_based_analysis: {
        drift_over_time: {
          '2024-01-01': 0.045,
          '2024-01-02': 0.078,
          '2024-01-03': 0.123,
          '2024-01-04': 0.089,
          '2024-01-05': 0.127
        }
      },
      clustering_analysis: {
        algorithm: 'k_means',
        silhouette_score: 0.67,
        cluster_characteristics: {
          'cluster_0': {
            description: 'High-confidence correct predictions',
            size: 645,
            accuracy: 0.98,
            avg_confidence: 0.91
          },
          'cluster_1': {
            description: 'Medium-confidence mixed results',
            size: 342,
            accuracy: 0.76,
            avg_confidence: 0.64
          },
          'cluster_2': {
            description: 'Low-confidence uncertain cases',
            size: 178,
            accuracy: 0.45,
            avg_confidence: 0.34
          },
          'cluster_3': {
            description: 'Overconfident errors',
            size: 82,
            accuracy: 0.12,
            avg_confidence: 0.87
          }
        }
      }
    };

    const mockIntrospectionData: IntrospectionData = {
      reflections_count: 247,
      daily_activity: {
        '2024-01-01': 23,
        '2024-01-02': 45,
        '2024-01-03': 67,
        '2024-01-04': 52,
        '2024-01-05': 60
      },
      accuracy_patterns: {
        'choice_a': { mean_accuracy: 0.89, count: 456 },
        'choice_b': { mean_accuracy: 0.94, count: 523 },
        'choice_c': { mean_accuracy: 0.87, count: 268 }
      }
    };

    setTimeout(() => {
      setTrainingData(mockTrainingData);
      setCalibrationData(mockCalibrationData);
      setPerformanceData(mockPerformanceData);
      setDriftData(mockDriftData);
      setIntrospectionData(mockIntrospectionData);
      setLoading(false);
    }, 1500);
  };

  if (loading) {
    return (
      <div className="container mx-auto p-6 space-y-6">
        <div className="flex items-center space-x-4 mb-6">
          <Brain className="h-8 w-8 text-blue-500 animate-pulse" />
          <div>
            <h1 className="text-3xl font-bold">Model Insights</h1>
            <p className="text-muted-foreground">Loading comprehensive model analysis...</p>
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
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center space-x-4 mb-6">
        <Brain className="h-8 w-8 text-blue-500" />
        <div>
          <h1 className="text-3xl font-bold">Model Insights</h1>
          <p className="text-muted-foreground">
            Comprehensive analysis of model training, calibration, performance, and behavior
          </p>
        </div>
      </div>

      {/* Main Content */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="training" className="flex items-center space-x-2">
            <Cpu className="h-4 w-4" />
            <span>Training</span>
          </TabsTrigger>
          <TabsTrigger value="calibration" className="flex items-center space-x-2">
            <Target className="h-4 w-4" />
            <span>Calibration</span>
          </TabsTrigger>
          <TabsTrigger value="performance" className="flex items-center space-x-2">
            <BarChart3 className="h-4 w-4" />
            <span>Performance</span>
          </TabsTrigger>
          <TabsTrigger value="drift" className="flex items-center space-x-2">
            <TrendingUp className="h-4 w-4" />
            <span>Drift</span>
          </TabsTrigger>
          <TabsTrigger value="introspection" className="flex items-center space-x-2">
            <Eye className="h-4 w-4" />
            <span>Introspection</span>
          </TabsTrigger>
        </TabsList>

        {/* Training Insights Tab */}
        <TabsContent value="training" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-3">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Model Architecture</CardTitle>
                <Layers className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{trainingData?.model_name.split('/').pop()}</div>
                <p className="text-xs text-muted-foreground">Neural architecture</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Training Examples</CardTitle>
                <Database className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{trainingData?.dataset_size.toLocaleString()}</div>
                <p className="text-xs text-muted-foreground">Dataset size</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Learning Rate</CardTitle>
                <Zap className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{trainingData?.training_params.learning_rate.toExponential(0)}</div>
                <p className="text-xs text-muted-foreground">Parameter update rate</p>
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Core Parameters</CardTitle>
                <CardDescription>Essential training configuration</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-sm font-medium">Batch Size:</span>
                  <Badge variant="outline">{trainingData?.training_params.batch_size}</Badge>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm font-medium">Epochs:</span>
                  <Badge variant="outline">{trainingData?.training_params.epochs}</Badge>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm font-medium">Max Length:</span>
                  <Badge variant="outline">{trainingData?.training_params.max_length} tokens</Badge>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm font-medium">Weight Decay:</span>
                  <Badge variant="outline">{trainingData?.training_params.weight_decay}</Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Training Strategy</CardTitle>
                <CardDescription>Advanced configuration settings</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-sm font-medium">Validation Split:</span>
                  <Badge variant="outline">{(trainingData?.training_params.validation_split * 100)}%</Badge>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm font-medium">Save Steps:</span>
                  <Badge variant="outline">{trainingData?.training_params.save_steps}</Badge>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm font-medium">Early Stopping:</span>
                  <Badge variant="outline">{trainingData?.training_params.early_stopping_patience} patience</Badge>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm font-medium">Random Seed:</span>
                  <Badge variant="outline">{trainingData?.training_params.seed}</Badge>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Calendar className="h-5 w-5" />
                <span>Training Timeline</span>
              </CardTitle>
              <CardDescription>Model development progress</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center space-x-2 text-sm">
                <Clock className="h-4 w-4 text-blue-500" />
                <span>Last trained: {new Date(trainingData?.timestamp || '').toLocaleString()}</span>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Calibration Insights Tab */}
        <TabsContent value="calibration" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-3">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">ECE Improvement</CardTitle>
                <Target className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-green-600">
                  -{(calibrationData?.metrics.improvement.ece * 100).toFixed(2)}%
                </div>
                <p className="text-xs text-muted-foreground">Calibration error reduction</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Log Loss</CardTitle>
                <BarChart3 className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {calibrationData?.metrics.post_calibration.log_loss.toFixed(3)}
                </div>
                <p className="text-xs text-green-600">
                  -{(calibrationData?.metrics.improvement.log_loss * 100).toFixed(1)}% improvement
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Brier Score</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {calibrationData?.metrics.post_calibration.brier_score.toFixed(3)}
                </div>
                <p className="text-xs text-green-600">
                  -{(calibrationData?.metrics.improvement.brier_score * 100).toFixed(1)}% improvement
                </p>
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <ThermometerSun className="h-5 w-5" />
                  <span>Temperature Scaling</span>
                </CardTitle>
                <CardDescription>Confidence calibration parameters</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="text-center">
                  <div className="text-3xl font-bold text-blue-600">
                    {calibrationData?.parameters.temperature.toFixed(2)}
                  </div>
                  <p className="text-sm text-muted-foreground">Calibration Temperature</p>
                </div>
                <Separator />
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm font-medium">Method:</span>
                    <Badge variant="outline">
                      {calibrationData?.method.replace('_', ' ').toLowerCase().replace(/\b\w/g, l => l.toUpperCase())}
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Calibration History</CardTitle>
                <CardDescription>Progress over time</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {calibrationData?.history.slice(-3).map((entry, index) => (
                    <div key={index} className="p-3 border rounded-lg">
                      <div className="flex justify-between items-start mb-2">
                        <div className="text-sm font-medium">
                          {new Date(entry.timestamp).toLocaleDateString()}
                        </div>
                        <Badge variant={entry.calibration_error < 0.05 ? "default" : "secondary"}>
                          {(entry.calibration_error * 100).toFixed(1)}% ECE
                        </Badge>
                      </div>
                      <p className="text-xs text-muted-foreground">{entry.notes}</p>
                      <div className="text-xs text-muted-foreground mt-1">
                        {entry.sample_count} samples • {(entry.accuracy * 100).toFixed(1)}% accuracy
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Performance Metrics Tab */}
        <TabsContent value="performance" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Overall Accuracy</CardTitle>
                <CheckCircle className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{(performanceData?.accuracy * 100).toFixed(1)}%</div>
                <p className="text-xs text-muted-foreground">Prediction accuracy</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Evaluations</CardTitle>
                <Database className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{performanceData?.total_votes.toLocaleString()}</div>
                <p className="text-xs text-muted-foreground">Human preferences analyzed</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Correct Predictions</CardTitle>
                <CheckCircle className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{performanceData?.correct_predictions.toLocaleString()}</div>
                <p className="text-xs text-muted-foreground">Accurate predictions</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">High Conf. Errors</CardTitle>
                <AlertTriangle className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-red-600">
                  {performanceData?.error_counts.high_confidence_error}
                </div>
                <p className="text-xs text-muted-foreground">Overconfident errors</p>
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Error Analysis</CardTitle>
                <CardDescription>Breakdown of prediction errors by type</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">High Confidence Errors</span>
                    <div className="flex items-center space-x-2">
                      <div className="w-12 text-right text-sm">
                        {performanceData?.error_counts.high_confidence_error}
                      </div>
                      <div className="w-24">
                        <Progress 
                          value={(performanceData?.error_counts.high_confidence_error / (performanceData?.total_votes || 1)) * 100} 
                          className="h-2" 
                        />
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Low Confidence Errors</span>
                    <div className="flex items-center space-x-2">
                      <div className="w-12 text-right text-sm">
                        {performanceData?.error_counts.low_confidence_error}
                      </div>
                      <div className="w-24">
                        <Progress 
                          value={(performanceData?.error_counts.low_confidence_error / (performanceData?.total_votes || 1)) * 100} 
                          className="h-2" 
                        />
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Misalignment Errors</span>
                    <div className="flex items-center space-x-2">
                      <div className="w-12 text-right text-sm">
                        {performanceData?.error_counts.misalignment_error}
                      </div>
                      <div className="w-24">
                        <Progress 
                          value={(performanceData?.error_counts.misalignment_error / (performanceData?.total_votes || 1)) * 100} 
                          className="h-2" 
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Confidence Distribution</CardTitle>
                <CardDescription>Model confidence levels across predictions</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {Object.entries(performanceData?.confidence_distribution || {}).map(([level, count]) => (
                    <div key={level} className="flex items-center justify-between">
                      <span className="text-sm font-medium capitalize">{level}</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-16 text-right text-sm">{count.toLocaleString()}</div>
                        <div className="w-24">
                          <Progress 
                            value={(count / (performanceData?.total_votes || 1)) * 100} 
                            className="h-2" 
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Drift Analysis Tab */}
        <TabsContent value="drift" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-3">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Drift Score</CardTitle>
                <TrendingUp className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {driftData?.overall_drift_score.toFixed(3)}
                </div>
                <p className="text-xs text-muted-foreground">Model behavior change</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Time Windows</CardTitle>
                <Clock className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{driftData?.time_windows_analyzed}</div>
                <p className="text-xs text-muted-foreground">Periods analyzed</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Clusters Found</CardTitle>
                <Layers className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{driftData?.clusters_identified}</div>
                <p className="text-xs text-muted-foreground">Behavior patterns</p>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Behavioral Patterns</CardTitle>
              <CardDescription>
                Clustering analysis using {driftData?.clustering_analysis.algorithm} 
                (Silhouette Score: {driftData?.clustering_analysis.silhouette_score.toFixed(2)})
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {Object.entries(driftData?.clustering_analysis.cluster_characteristics || {}).map(([clusterId, char]) => (
                  <div key={clusterId} className="p-4 border rounded-lg">
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <h4 className="font-medium">{char.description}</h4>
                        <p className="text-sm text-muted-foreground">
                          {char.size} examples • {(char.accuracy * 100).toFixed(1)}% accuracy
                        </p>
                      </div>
                      <Badge variant="outline">
                        {(char.avg_confidence * 100).toFixed(1)}% confidence
                      </Badge>
                    </div>
                    <Progress value={char.accuracy * 100} className="h-2" />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Introspection Tab */}
        <TabsContent value="introspection" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Reflection Entries</CardTitle>
                <MessageSquare className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{introspectionData?.reflections_count}</div>
                <p className="text-xs text-muted-foreground">Self-analysis sessions</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Daily Activity</CardTitle>
                <Calendar className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {Math.max(...Object.values(introspectionData?.daily_activity || {}))}
                </div>
                <p className="text-xs text-muted-foreground">Peak daily reflections</p>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Accuracy by Prediction Type</CardTitle>
              <CardDescription>Self-reflection analysis patterns</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {Object.entries(introspectionData?.accuracy_patterns || {}).map(([choice, stats]) => (
                  <div key={choice} className="flex items-center justify-between p-3 border rounded-lg">
                    <div>
                      <span className="font-medium">{choice.replace('_', ' ').toUpperCase()}</span>
                      <p className="text-sm text-muted-foreground">{stats.count} instances</p>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold">{(stats.mean_accuracy * 100).toFixed(1)}%</div>
                      <Progress value={stats.mean_accuracy * 100} className="h-2 w-20" />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
} 