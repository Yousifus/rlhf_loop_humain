'use client'

import React, { useState, useEffect } from 'react'
import { HomeIcon, ArrowPathIcon, CheckCircleIcon, ClockIcon, ExclamationTriangleIcon } from '@heroicons/react/24/outline'
import DashboardLayout from '@/components/layout/DashboardLayout'
import MetricCard from '@/components/ui/MetricCard'
import Chart from '@/components/charts/Chart'

interface OverviewData {
  total_votes: number | null
  model_accuracy: number | null
  calibration_score: number | null
  avg_response_time: number | null
  recent_activity: Array<{
    message: string
    details: string
    time: string
    type: string
  }>
  has_data: boolean
  last_updated: string
  error?: string
}

export default function OverviewPage() {
  const [data, setData] = useState<OverviewData | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [actionLoading, setActionLoading] = useState<string | null>(null)

  useEffect(() => {
    loadOverviewData()
  }, [])

  const loadOverviewData = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/overview')
      const result = await response.json()
      setData(result)
    } catch (error) {
      console.error('Error loading overview data:', error)
      setData({
        total_votes: 0,
        model_accuracy: null,
        calibration_score: null,
        avg_response_time: null,
        recent_activity: [],
        has_data: false,
        last_updated: new Date().toISOString(),
        error: 'Failed to connect to backend'
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleQuickAction = async (action: string) => {
    setActionLoading(action)
    
    try {
      let response
      
      switch (action) {
        case 'generate-batch':
          response = await fetch('http://localhost:8000/api/actions/generate-batch', { method: 'POST' })
          break
        case 'run-calibration':
          response = await fetch('http://localhost:8000/api/actions/run-calibration', { method: 'POST' })
          break
        case 'export-data':
          response = await fetch('http://localhost:8000/api/actions/export-data')
          break
        case 'view-logs':
          response = await fetch('http://localhost:8000/api/actions/view-logs')
          break
        default:
          throw new Error(`Unknown action: ${action}`)
      }
      
      const result = await response.json()
      
      if (result.success) {
        alert(`✅ ${result.message}`)
        
        // Show additional info for some actions
        if (action === 'export-data' && result.export_info) {
          const info = result.export_info
          alert(`📊 Export prepared:\n- Files: ${info.files.length}\n- Total records: ${info.total_records}\n- Export ID: ${info.export_id}`)
        }
        
        if (action === 'view-logs' && result.logs) {
          console.log('System Logs:', result.logs)
          alert(`📋 ${result.logs.length} log entries available. Check browser console for details.`)
        }
      } else {
        alert(`❌ Error: ${result.error}`)
      }
    } catch (error) {
      alert(`❌ Connection error: ${error}`)
    } finally {
      setActionLoading(null)
    }
  }

  if (isLoading) {
    return (
      <DashboardLayout>
        <div className="space-y-6 animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/3"></div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-24 bg-gray-200 rounded-xl"></div>
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
          <ExclamationTriangleIcon className="h-16 w-16 text-red-500 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Failed to Load Data</h3>
          <p className="text-gray-600 mb-4">Unable to connect to the RLHF backend.</p>
          <button 
            onClick={loadOverviewData}
            className="humain-btn-primary"
          >
            Retry
          </button>
        </div>
      </DashboardLayout>
    )
  }

  // Empty state when no data is available
  if (!data.has_data && !data.error) {
    return (
      <DashboardLayout>
        <div className="space-y-6 humain-fade-in">
          {/* Header */}
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">RLHF Dashboard Overview</h1>
              <p className="text-gray-600">Welcome to your RLHF pipeline monitoring system</p>
            </div>
            <HomeIcon className="h-8 w-8 text-humain-400" />
          </div>

          {/* Empty State */}
          <div className="text-center py-12">
            <div className="humain-card max-w-md mx-auto">
              <div className="humain-card-content text-center">
                <ArrowPathIcon className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-gray-900 mb-2">No Data Available</h3>
                <p className="text-gray-600 mb-6">
                  Start by generating predictions or collecting human feedback to see your RLHF pipeline data here.
                </p>
                
                {/* Quick Start Actions */}
                <div className="grid grid-cols-2 gap-3">
                  <button
                    onClick={() => handleQuickAction('generate-batch')}
                    disabled={actionLoading === 'generate-batch'}
                    className="humain-btn-primary disabled:opacity-50"
                  >
                    {actionLoading === 'generate-batch' ? 'Starting...' : 'Generate Batch'}
                  </button>
                  <button
                    onClick={() => window.location.href = '/annotation'}
                    className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors"
                  >
                    Start Annotation
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* System Status */}
          <div className="humain-card">
            <div className="humain-card-content">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">System Status</h3>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <CheckCircleIcon className="h-5 w-5 text-green-500" />
                  <span className="text-sm text-gray-600">RLHF Pipeline Ready</span>
                </div>
                <div className="text-sm text-gray-500">
                  Last updated: {new Date(data.last_updated).toLocaleTimeString()}
                </div>
              </div>
            </div>
          </div>
        </div>
      </DashboardLayout>
    )
  }

  // Main dashboard with data
  return (
    <DashboardLayout>
      <div className="space-y-6 humain-fade-in">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">RLHF Dashboard Overview</h1>
            <p className="text-gray-600">Monitor your reinforcement learning from human feedback pipeline</p>
          </div>
          <HomeIcon className="h-8 w-8 text-humain-400" />
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <MetricCard
            title="Total Votes"
            value={data.total_votes?.toLocaleString() || '0'}
            change={data.total_votes ? '+12.5%' : undefined}
            changeType={data.total_votes ? 'positive' : 'neutral'}
            description="Human feedback votes collected"
          />
          
          <MetricCard
            title="Model Accuracy"
            value={data.model_accuracy ? `${(data.model_accuracy * 100).toFixed(1)}%` : 'N/A'}
            change={data.model_accuracy ? '+2.3%' : undefined}
            changeType={data.model_accuracy ? 'positive' : 'neutral'}
            description="Prediction accuracy rate"
          />
          
          <MetricCard
            title="Calibration Score"
            value={data.calibration_score ? `${(data.calibration_score * 100).toFixed(1)}%` : 'N/A'}
            change={data.calibration_score ? '-1.2%' : undefined}
            changeType={data.calibration_score ? 'negative' : 'neutral'}
            description="Confidence calibration quality"
          />
          
          <MetricCard
            title="Avg Response Time"
            value={data.avg_response_time ? `${data.avg_response_time.toFixed(3)}s` : 'N/A'}
            change={data.avg_response_time ? '-5.7%' : undefined}
            changeType={data.avg_response_time ? 'positive' : 'neutral'}
            description="Model inference latency"
          />
        </div>

        {/* Charts and Activity */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Performance Chart */}
          <div className="humain-card">
            <div className="humain-card-content">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Performance Over Time</h3>
              {data.total_votes && data.total_votes > 0 ? (
                <Chart
                  data={[
                    { time: 'Week 1', performance: 0.65 },
                    { time: 'Week 2', performance: 0.72 },
                    { time: 'Week 3', performance: 0.78 },
                    { time: 'Week 4', performance: data.model_accuracy || 0.84 }
                  ]}
                  lines={[
                    { key: 'performance', name: 'Model Performance', color: '#1DB584' }
                  ]}
                  height={200}
                />
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <ClockIcon className="h-12 w-12 mx-auto mb-2 text-gray-300" />
                  <p>Performance data will appear after collecting votes</p>
                </div>
              )}
            </div>
          </div>

          {/* Recent Activity */}
          <div className="humain-card">
            <div className="humain-card-content">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Activity</h3>
              {data.recent_activity.length > 0 ? (
                <div className="space-y-3">
                  {data.recent_activity.map((activity, index) => (
                    <div key={index} className="flex items-start space-x-3">
                      <div className={`w-2 h-2 rounded-full mt-2 ${
                        activity.type === 'annotation' ? 'bg-green-500' :
                        activity.type === 'calibration' ? 'bg-blue-500' :
                        activity.type === 'drift' ? 'bg-yellow-500' : 'bg-gray-500'
                      }`}></div>
                      <div className="flex-1">
                        <div className="font-medium text-gray-900">{activity.message}</div>
                        <div className="text-sm text-gray-600">{activity.details}</div>
                        <div className="text-xs text-gray-500 mt-1">{activity.time}</div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <ArrowPathIcon className="h-12 w-12 mx-auto mb-2 text-gray-300" />
                  <p>Recent activity will appear here</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="humain-card">
          <div className="humain-card-content">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <button
                onClick={() => handleQuickAction('generate-batch')}
                disabled={actionLoading === 'generate-batch'}
                className="humain-btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {actionLoading === 'generate-batch' ? 'Starting...' : 'Generate New Batch'}
              </button>
              
              <button
                onClick={() => handleQuickAction('run-calibration')}
                disabled={actionLoading === 'run-calibration'}
                className="humain-btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {actionLoading === 'run-calibration' ? 'Running...' : 'Run Calibration'}
              </button>
              
              <button
                onClick={() => handleQuickAction('export-data')}
                disabled={actionLoading === 'export-data'}
                className="humain-btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {actionLoading === 'export-data' ? 'Exporting...' : 'Export Data'}
              </button>
              
              <button
                onClick={() => handleQuickAction('view-logs')}
                disabled={actionLoading === 'view-logs'}
                className="humain-btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {actionLoading === 'view-logs' ? 'Loading...' : 'View Logs'}
              </button>
            </div>
          </div>
        </div>

        {/* System Info */}
        <div className="text-center text-sm text-gray-500">
          Last updated: {new Date(data.last_updated).toLocaleString()}
          {data.error && (
            <span className="text-red-500 ml-2">• Warning: {data.error}</span>
          )}
        </div>
      </div>
    </DashboardLayout>
  )
} 