'use client'

import React, { useState, useEffect } from 'react'
import { DocumentChartBarIcon, ScaleIcon, ChartBarIcon, ClockIcon } from '@heroicons/react/24/outline'
import DashboardLayout from '@/components/layout/DashboardLayout'
import Chart from '@/components/charts/Chart'

interface CalibrationData {
  reliability_data: Array<{
    confidence: number
    accuracy: number
    count: number
  }>
  ece_score: number | null
  brier_score: number | null
  log_loss: number | null
  has_data: boolean
  last_updated: string
}

export default function CalibrationPage() {
  const [isLoading, setIsLoading] = useState(true)
  const [data, setData] = useState<CalibrationData | null>(null)
  const [selectedView, setSelectedView] = useState('reliability')

  useEffect(() => {
    loadCalibrationData()
  }, [])

  const loadCalibrationData = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/calibration')
      const result = await response.json()
      setData(result)
    } catch (error) {
      console.error('Error loading calibration data:', error)
      setData({
        reliability_data: [],
        ece_score: null,
        brier_score: null,
        log_loss: null,
        has_data: false,
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
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Failed to Load Calibration</h3>
          <button onClick={loadCalibrationData} className="humain-btn-primary">Retry</button>
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
              <h1 className="text-2xl font-bold text-gray-900">Model Calibration</h1>
              <p className="text-gray-600">Analyze confidence alignment and prediction reliability</p>
            </div>
            <DocumentChartBarIcon className="h-8 w-8 text-humain-400" />
          </div>

          {/* Empty State */}
          <div className="text-center py-12">
            <div className="humain-card max-w-md mx-auto">
              <div className="humain-card-content text-center">
                <ScaleIcon className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-gray-900 mb-2">No Calibration Data</h3>
                <p className="text-gray-600 mb-4">
                  Calibration analysis requires model predictions with confidence scores.
                </p>
                <button
                  onClick={() => window.location.href = '/'}
                  className="humain-btn-primary"
                >
                  Generate Predictions
                </button>
              </div>
            </div>
          </div>
        </div>
      </DashboardLayout>
    )
  }

  return (
    <DashboardLayout>
      <div className="space-y-6 humain-fade-in">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Model Calibration</h1>
            <p className="text-gray-600">Analyze confidence alignment and prediction reliability</p>
          </div>
          <DocumentChartBarIcon className="h-8 w-8 text-humain-400" />
        </div>

        {/* Calibration Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="humain-card">
            <div className="humain-card-content text-center">
              <ScaleIcon className="h-8 w-8 text-humain-400 mx-auto mb-2" />
              <div className="text-2xl font-bold text-humain-600 mb-2">
                {data.ece_score ? `${(data.ece_score * 100).toFixed(1)}%` : 'N/A'}
              </div>
              <div className="text-sm text-gray-600">Expected Calibration Error</div>
              <div className="text-xs text-green-600 mt-1">
                {data.ece_score ? 'Well calibrated' : 'Waiting for data'}
              </div>
            </div>
          </div>

          <div className="humain-card">
            <div className="humain-card-content text-center">
              <div className="text-2xl font-bold text-humain-600 mb-2">
                {data.brier_score ? data.brier_score.toFixed(3) : 'N/A'}
              </div>
              <div className="text-sm text-gray-600">Brier Score</div>
              <div className="text-xs text-green-600 mt-1">
                {data.brier_score ? 'Lower is better' : 'Waiting for data'}
              </div>
            </div>
          </div>

          <div className="humain-card">
            <div className="humain-card-content text-center">
              <div className="text-2xl font-bold text-humain-600 mb-2">
                {data.log_loss ? data.log_loss.toFixed(3) : 'N/A'}
              </div>
              <div className="text-sm text-gray-600">Log Loss</div>
              <div className="text-xs text-green-600 mt-1">
                {data.log_loss ? 'Lower is better' : 'Waiting for data'}
              </div>
            </div>
          </div>

          <div className="humain-card">
            <div className="humain-card-content text-center">
              <div className="text-2xl font-bold text-humain-600 mb-2">
                {data.reliability_data.length > 0 ? `${data.reliability_data.length}` : '0'}
              </div>
              <div className="text-sm text-gray-600">Confidence Bins</div>
              <div className="text-xs text-green-600 mt-1">
                {data.reliability_data.length > 0 ? 'Data points' : 'No data yet'}
              </div>
            </div>
          </div>
        </div>

        {/* Reliability Chart */}
        <div className="humain-card">
          <div className="humain-card-content">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Reliability Diagram</h3>
            {data.reliability_data.length > 0 ? (
              <Chart
                data={data.reliability_data}
                lines={[
                  { key: 'accuracy', name: 'Actual Accuracy', color: '#1DB584' },
                  { key: 'confidence', name: 'Perfect Calibration', color: '#6B7280' }
                ]}
                height={300}
              />
            ) : (
              <div className="text-center py-12 text-gray-500">
                <ClockIcon className="h-12 w-12 mx-auto mb-2 text-gray-300" />
                <p>Reliability diagram will appear after collecting confidence data</p>
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