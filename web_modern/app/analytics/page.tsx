'use client'

import React, { useState, useEffect } from 'react'
import { ChartBarIcon, ArrowTrendingUpIcon, ArrowTrendingDownIcon, ClockIcon } from '@heroicons/react/24/outline'
import DashboardLayout from '@/components/layout/DashboardLayout'
import Chart from '@/components/charts/Chart'

interface AnalyticsData {
  performance_data: Array<{
    month: string
    accuracy: number
    precision: number
    recall: number
    f1: number
  }>
  domain_data: Array<{
    domain: string
    votes: number
    accuracy: number
    trend: string
  }>
  has_data: boolean
  last_updated: string
}

export default function AnalyticsPage() {
  const [isLoading, setIsLoading] = useState(true)
  const [data, setData] = useState<AnalyticsData | null>(null)
  const [selectedMetric, setSelectedMetric] = useState('accuracy')

  useEffect(() => {
    loadAnalyticsData()
  }, [])

  const loadAnalyticsData = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/analytics')
      const result = await response.json()
      setData(result)
    } catch (error) {
      console.error('Error loading analytics data:', error)
      setData({
        performance_data: [],
        domain_data: [],
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
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Failed to Load Analytics</h3>
          <button onClick={loadAnalyticsData} className="humain-btn-primary">Retry</button>
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
              <h1 className="text-2xl font-bold text-gray-900">Advanced Analytics</h1>
              <p className="text-gray-600">Deep insights into model performance and behavior patterns</p>
            </div>
            <ChartBarIcon className="h-8 w-8 text-humain-400" />
          </div>

          {/* Empty State */}
          <div className="text-center py-12">
            <div className="humain-card max-w-md mx-auto">
              <div className="humain-card-content text-center">
                <ClockIcon className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-gray-900 mb-2">No Analytics Data</h3>
                <p className="text-gray-600 mb-4">
                  Analytics will appear once you have collected human feedback and model predictions.
                </p>
                <button
                  onClick={() => window.location.href = '/annotation'}
                  className="humain-btn-primary"
                >
                  Start Collecting Data
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
            <h1 className="text-2xl font-bold text-gray-900">Advanced Analytics</h1>
            <p className="text-gray-600">Deep insights into model performance and behavior patterns</p>
          </div>
          <ChartBarIcon className="h-8 w-8 text-humain-400" />
        </div>

        {/* Metric Selector */}
        <div className="flex flex-wrap gap-3">
          {['accuracy', 'precision', 'recall', 'f1'].map((metric) => (
            <button
              key={metric}
              onClick={() => setSelectedMetric(metric)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                selectedMetric === metric
                  ? 'bg-humain-400 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {metric.charAt(0).toUpperCase() + metric.slice(1)}
            </button>
          ))}
        </div>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Performance Trends */}
          <div className="humain-card">
            <div className="humain-card-content">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Performance Trends - {selectedMetric.charAt(0).toUpperCase() + selectedMetric.slice(1)}
              </h3>
              {data.performance_data.length > 0 ? (
                <Chart
                  data={data.performance_data}
                  lines={[
                    { key: selectedMetric, name: selectedMetric.charAt(0).toUpperCase() + selectedMetric.slice(1), color: '#1DB584' }
                  ]}
                  height={250}
                />
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <p>Performance trends will appear after data collection</p>
                </div>
              )}
            </div>
          </div>

          {/* Multi-Metric Comparison */}
          <div className="humain-card">
            <div className="humain-card-content">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">All Metrics Comparison</h3>
              {data.performance_data.length > 0 ? (
                <Chart
                  data={data.performance_data}
                  lines={[
                    { key: 'accuracy', name: 'Accuracy', color: '#1DB584' },
                    { key: 'precision', name: 'Precision', color: '#3B82F6' },
                    { key: 'recall', name: 'Recall', color: '#8B5CF6' },
                    { key: 'f1', name: 'F1 Score', color: '#F59E0B' }
                  ]}
                  height={250}
                />
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <p>Metric comparison will appear after data collection</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Domain Performance Table */}
        <div className="humain-card">
          <div className="humain-card-content">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Performance by Domain</h3>
            {data.domain_data.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Domain</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Total Votes</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Accuracy</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Trend</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {data.domain_data.map((domain, index) => (
                      <tr key={index} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="font-medium text-gray-900">{domain.domain}</div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-gray-900">{domain.votes.toLocaleString()}</div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="font-medium text-gray-900">
                            {(domain.accuracy * 100).toFixed(1)}%
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className={`flex items-center ${
                            domain.trend.startsWith('+') ? 'text-green-600' : 'text-red-600'
                          }`}>
                            {domain.trend.startsWith('+') ? (
                              <ArrowTrendingUpIcon className="h-4 w-4 mr-1" />
                            ) : (
                              <ArrowTrendingDownIcon className="h-4 w-4 mr-1" />
                            )}
                            <span className="font-medium">{domain.trend}</span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <p>Domain performance data will appear after annotation collection</p>
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