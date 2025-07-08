'use client'

import React, { useState, useEffect } from 'react'
import { ArrowTrendingUpIcon, BeakerIcon, ClockIcon } from '@heroicons/react/24/outline'
import DashboardLayout from '@/components/layout/DashboardLayout'
import Chart from '@/components/charts/Chart'

interface EvolutionData {
  performance_timeline: Array<{
    version: string
    accuracy: number
    user_preference: number
    training_date: string
  }>
  model_versions: Array<{
    version: string
    release_date: string
    improvements: string[]
    performance_delta: number
  }>
  has_data: boolean
  last_updated: string
}

export default function ModelEvolutionPage() {
  const [isLoading, setIsLoading] = useState(true)
  const [data, setData] = useState<EvolutionData | null>(null)
  const [selectedMetric, setSelectedMetric] = useState('accuracy')

  useEffect(() => {
    loadEvolutionData()
  }, [])

  const loadEvolutionData = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/evolution')
      const result = await response.json()
      setData(result)
    } catch (error) {
      console.error('Error loading evolution data:', error)
      setData({
        performance_timeline: [],
        model_versions: [],
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
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Failed to Load Evolution Data</h3>
          <button onClick={loadEvolutionData} className="humain-btn-primary">Retry</button>
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
              <h1 className="text-2xl font-bold text-gray-900">Model Evolution</h1>
              <p className="text-gray-600">Track model improvements and performance over time</p>
            </div>
            <ArrowTrendingUpIcon className="h-8 w-8 text-humain-400" />
          </div>

          {/* Empty State */}
          <div className="text-center py-12">
            <div className="humain-card max-w-md mx-auto">
              <div className="humain-card-content text-center">
                <BeakerIcon className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-gray-900 mb-2">No Model Versions Yet</h3>
                <p className="text-gray-600 mb-4">
                  Model evolution tracking will begin once you train or deploy different model versions.
                </p>
                <button
                  onClick={() => window.location.href = '/'}
                  className="humain-btn-primary"
                >
                  Start Training Models
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
            <h1 className="text-2xl font-bold text-gray-900">Model Evolution</h1>
            <p className="text-gray-600">Track model improvements and performance over time</p>
          </div>
          <ArrowTrendingUpIcon className="h-8 w-8 text-humain-400" />
        </div>

        {/* Metric Toggle */}
        <div className="flex flex-wrap gap-3">
          {['accuracy', 'user_preference'].map((metric) => (
            <button
              key={metric}
              onClick={() => setSelectedMetric(metric)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                selectedMetric === metric
                  ? 'bg-humain-400 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {metric === 'user_preference' ? 'User Preference' : 'Accuracy'}
            </button>
          ))}
        </div>

        {/* Performance Timeline */}
        <div className="humain-card">
          <div className="humain-card-content">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Performance Over Time - {selectedMetric === 'user_preference' ? 'User Preference' : 'Accuracy'}
            </h3>
            {data.performance_timeline.length > 0 ? (
              <Chart
                data={data.performance_timeline}
                lines={[
                  { key: selectedMetric, name: selectedMetric === 'user_preference' ? 'User Preference' : 'Accuracy', color: '#1DB584' }
                ]}
                height={300}
              />
            ) : (
              <div className="text-center py-12 text-gray-500">
                <ClockIcon className="h-12 w-12 mx-auto mb-2 text-gray-300" />
                <p>Performance timeline will appear after model training</p>
              </div>
            )}
          </div>
        </div>

        {/* Model Versions Table */}
        <div className="humain-card">
          <div className="humain-card-content">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Version History</h3>
            {data.model_versions.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Version</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Release Date</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Key Improvements</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Performance</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {data.model_versions.map((version, index) => (
                      <tr key={index} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="font-medium text-gray-900">{version.version}</div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-gray-900">{new Date(version.release_date).toLocaleDateString()}</div>
                        </td>
                        <td className="px-6 py-4">
                          <div className="text-sm text-gray-900">
                            {version.improvements.slice(0, 2).map((improvement, i) => (
                              <div key={i} className="mb-1">• {improvement}</div>
                            ))}
                            {version.improvements.length > 2 && (
                              <div className="text-gray-500">+{version.improvements.length - 2} more...</div>
                            )}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className={`font-medium ${
                            version.performance_delta > 0 ? 'text-green-600' : 
                            version.performance_delta < 0 ? 'text-red-600' : 'text-gray-600'
                          }`}>
                            {version.performance_delta > 0 ? '+' : ''}{(version.performance_delta * 100).toFixed(1)}%
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <p>Model version history will appear after deploying different models</p>
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