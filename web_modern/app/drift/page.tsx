'use client'

import React, { useState, useEffect } from 'react'
import { ExclamationTriangleIcon, ChartBarIcon, ClockIcon } from '@heroicons/react/24/outline'
import DashboardLayout from '@/components/layout/DashboardLayout'
import Chart from '@/components/charts/Chart'

interface DriftData {
  drift_timeline: Array<{
    date: string
    drift_score: number
    accuracy_drop: number
    data_points: number
  }>
  cluster_analysis: Array<{
    cluster_id: string
    size: number
    drift_severity: string
    representative_examples: string[]
  }>
  has_data: boolean
  current_drift_score: number | null
  alert_threshold: number
  last_updated: string
}

export default function DriftAnalysisPage() {
  const [isLoading, setIsLoading] = useState(true)
  const [data, setData] = useState<DriftData | null>(null)
  const [selectedTimeframe, setSelectedTimeframe] = useState('7d')

  useEffect(() => {
    loadDriftData()
  }, [])

  const loadDriftData = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/drift')
      const result = await response.json()
      setData(result)
    } catch (error) {
      console.error('Error loading drift data:', error)
      setData({
        drift_timeline: [],
        cluster_analysis: [],
        has_data: false,
        current_drift_score: null,
        alert_threshold: 0.3,
        last_updated: new Date().toISOString()
      })
    } finally {
      setIsLoading(false)
    }
  }

  if (isLoading) {
    return (
      <DashboardLayout>
        <div className="space-y-6 animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/3"></div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-64 bg-gray-200 rounded-xl"></div>
            ))}
          </div>
        </div>
      </DashboardLayout>
    )
  }

  if (!data) {
    return (
      <DashboardLayout>
        <div className="text-center py-12">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Failed to Load Drift Data</h3>
          <button onClick={loadDriftData} className="humain-btn-primary">Retry</button>
        </div>
      </DashboardLayout>
    )
  }

  // Empty state when no data
  if (!data.has_data) {
    return (
      <DashboardLayout>
        <div className="space-y-6 humain-fade-in">
          {/* Header */}
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Drift Analysis</h1>
              <p className="text-gray-600">Monitor data distribution changes and model performance degradation</p>
            </div>
            <ExclamationTriangleIcon className="h-8 w-8 text-humain-400" />
          </div>

          {/* Empty State */}
          <div className="text-center py-12">
            <div className="humain-card max-w-md mx-auto">
              <div className="humain-card-content text-center">
                <ChartBarIcon className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Insufficient Data for Drift Analysis</h3>
                <p className="text-gray-600 mb-4">
                  Drift analysis requires at least <strong>6 annotations</strong> to detect meaningful patterns in confidence and choice distributions over time.
                </p>
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-4">
                  <p className="text-sm text-blue-800">
                    <strong>Current:</strong> You have 5 annotations<br/>
                    <strong>Needed:</strong> 1 more annotation for drift analysis
                  </p>
                </div>
                <button
                  onClick={() => window.location.href = '/annotation'}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Add More Annotations
                </button>
                <p className="text-xs text-gray-500 mt-2">
                  Drift timeline and clusters will appear automatically
                </p>
              </div>
            </div>
          </div>
        </div>
      </DashboardLayout>
    )
  }

  const getDriftSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'high': return 'text-red-600 bg-red-100'
      case 'medium': return 'text-yellow-600 bg-yellow-100'
      case 'low': return 'text-green-600 bg-green-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  return (
    <DashboardLayout>
      <div className="space-y-6 humain-fade-in">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Drift Analysis</h1>
            <p className="text-gray-600">Monitor data distribution changes and model performance degradation</p>
          </div>
          <ExclamationTriangleIcon className="h-8 w-8 text-humain-400" />
        </div>

        {/* Current Status */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="humain-card">
            <div className="humain-card-content text-center">
              <ExclamationTriangleIcon className="h-8 w-8 text-humain-400 mx-auto mb-2" />
              <div className="text-2xl font-bold text-humain-600 mb-2">
                {data.current_drift_score ? data.current_drift_score.toFixed(3) : 'N/A'}
              </div>
              <div className="text-sm text-gray-600">Current Drift Score</div>
              <div className={`text-xs mt-1 ${
                data.current_drift_score && data.current_drift_score > data.alert_threshold 
                  ? 'text-red-600' : 'text-green-600'
              }`}>
                {data.current_drift_score 
                  ? (data.current_drift_score > data.alert_threshold ? 'Alert Threshold Exceeded' : 'Within Normal Range')
                  : 'No data yet'
                }
              </div>
            </div>
          </div>

          <div className="humain-card">
            <div className="humain-card-content text-center">
              <div className="text-2xl font-bold text-humain-600 mb-2">
                {data.alert_threshold.toFixed(1)}
              </div>
              <div className="text-sm text-gray-600">Alert Threshold</div>
              <div className="text-xs text-gray-500 mt-1">Configure in settings</div>
            </div>
          </div>

          <div className="humain-card">
            <div className="humain-card-content text-center">
              <div className="text-2xl font-bold text-humain-600 mb-2">
                {data.cluster_analysis.length}
              </div>
              <div className="text-sm text-gray-600">Drift Clusters</div>
              <div className="text-xs text-gray-500 mt-1">
                {data.cluster_analysis.length > 0 ? 'Detected patterns' : 'No clusters yet'}
              </div>
            </div>
          </div>

          <div className="humain-card">
            <div className="humain-card-content text-center">
              <div className="text-2xl font-bold text-humain-600 mb-2">
                {data.drift_timeline.length}
              </div>
              <div className="text-sm text-gray-600">Data Points</div>
              <div className="text-xs text-gray-500 mt-1">
                {data.drift_timeline.length > 0 ? 'Timeline entries' : 'No timeline yet'}
              </div>
            </div>
          </div>
        </div>

        {/* Timeframe Selector */}
        <div className="flex flex-wrap gap-3">
          {['1d', '7d', '30d', '90d'].map((timeframe) => (
            <button
              key={timeframe}
              onClick={() => setSelectedTimeframe(timeframe)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                selectedTimeframe === timeframe
                  ? 'bg-humain-400 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {timeframe}
            </button>
          ))}
        </div>

        {/* Drift Timeline */}
        <div className="humain-card">
          <div className="humain-card-content">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Drift Score Timeline</h3>
            {data.drift_timeline.length > 0 ? (
              <Chart
                data={data.drift_timeline}
                lines={[
                  { key: 'drift_score', name: 'Drift Score', color: '#EF4444' },
                  { key: 'accuracy_drop', name: 'Accuracy Drop', color: '#F59E0B' }
                ]}
                height={300}
              />
            ) : (
              <div className="text-center py-12 text-gray-500">
                <ClockIcon className="h-12 w-12 mx-auto mb-2 text-gray-300" />
                <p>Drift timeline will appear after continuous data collection</p>
              </div>
            )}
          </div>
        </div>

        {/* Cluster Analysis */}
        <div className="humain-card">
          <div className="humain-card-content">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Drift Cluster Analysis</h3>
            {data.cluster_analysis.length > 0 ? (
              <div className="space-y-4">
                {data.cluster_analysis.map((cluster, index) => (
                  <div key={index} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="font-medium text-gray-900">Cluster {cluster.cluster_id}</h4>
                      <div className="flex items-center space-x-3">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getDriftSeverityColor(cluster.drift_severity)}`}>
                          {cluster.drift_severity.charAt(0).toUpperCase() + cluster.drift_severity.slice(1)} Drift
                        </span>
                        <span className="text-sm text-gray-600">{cluster.size} examples</span>
                      </div>
                    </div>
                    <div className="text-sm text-gray-700">
                      <div className="font-medium mb-2">Representative Examples:</div>
                      <ul className="space-y-1">
                        {cluster.representative_examples.map((example, i) => (
                          <li key={i} className="text-gray-600">• {example}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <p>Cluster analysis will appear after detecting drift patterns</p>
              </div>
            )}
          </div>
        </div>

        {/* System Info */}
        <div className="text-center text-sm text-gray-500">
          Last updated: {new Date(data.last_updated).toLocaleString()}
        </div>
      </div>
    </DashboardLayout>
  )
} 