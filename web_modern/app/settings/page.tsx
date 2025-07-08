'use client'

import React, { useState, useEffect } from 'react'
import { CogIcon, KeyIcon, BellIcon, UserIcon, ServerIcon, CheckCircleIcon, XCircleIcon } from '@heroicons/react/24/outline'
import DashboardLayout from '@/components/layout/DashboardLayout'

export default function SettingsPage() {
  const [isLoading, setIsLoading] = useState(true)
  const [isSaving, setIsSaving] = useState(false)
  const [testingApi, setTestingApi] = useState<string | null>(null)
  const [apiStatus, setApiStatus] = useState<Record<string, boolean>>({})
  const [settings, setSettings] = useState({
    apiKeys: {
      deepseek: '',
      openai: '',
      lmstudio: 'http://localhost:1234'
    },
    notifications: {
      email: true,
      drift_alerts: true,
      training_complete: true,
      weekly_reports: false
    },
    dashboard: {
      auto_refresh: true,
      refresh_interval: 30,
      theme: 'light',
      show_debug: false
    },
    model: {
      default_provider: 'deepseek',
      temperature: 0.7,
      max_tokens: 500,
      batch_size: 32
    }
  })

  useEffect(() => {
    loadSettings()
  }, [])

  const loadSettings = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/settings')
      if (response.ok) {
        const data = await response.json()
        setSettings(data)
      }
    } catch (error) {
      console.error('Error loading settings:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleSave = async () => {
    setIsSaving(true)
    try {
      const response = await fetch('http://localhost:8000/api/settings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(settings),
      })

      if (response.ok) {
        alert('✅ Settings saved successfully!')
      } else {
        alert('❌ Failed to save settings')
      }
    } catch (error) {
      console.error('Error saving settings:', error)
      alert('❌ Error saving settings')
    } finally {
      setIsSaving(false)
    }
  }

  const testApiConnection = async (provider: string) => {
    setTestingApi(provider)
    try {
      const response = await fetch(`http://localhost:8000/api/model-providers/${provider}/test`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          api_key: settings.apiKeys[provider as keyof typeof settings.apiKeys],
          base_url: provider === 'lmstudio' ? settings.apiKeys.lmstudio : undefined
        }),
      })

      const result = await response.json()
      setApiStatus(prev => ({ ...prev, [provider]: result.success }))
      
      if (result.success) {
        alert(`✅ ${provider.charAt(0).toUpperCase() + provider.slice(1)} API connection successful!`)
      } else {
        alert(`❌ ${provider.charAt(0).toUpperCase() + provider.slice(1)} API connection failed: ${result.error}`)
      }
    } catch (error) {
      console.error(`Error testing ${provider} API:`, error)
      setApiStatus(prev => ({ ...prev, [provider]: false }))
      alert(`❌ Failed to test ${provider} API connection`)
    } finally {
      setTestingApi(null)
    }
  }

  const handleReset = () => {
    if (confirm('Reset all settings to default values?')) {
      setSettings({
        apiKeys: {
          deepseek: '',
          openai: '',
          lmstudio: 'http://localhost:1234'
        },
        notifications: {
          email: true,
          drift_alerts: true,
          training_complete: true,
          weekly_reports: false
        },
        dashboard: {
          auto_refresh: true,
          refresh_interval: 30,
          theme: 'light',
          show_debug: false
        },
        model: {
          default_provider: 'deepseek',
          temperature: 0.7,
          max_tokens: 500,
          batch_size: 32
        }
      })
    }
  }

  if (isLoading) {
    return (
      <DashboardLayout>
        <div className="space-y-6 animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/3"></div>
          <div className="grid grid-cols-1 gap-6">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-64 bg-gray-200 rounded-xl"></div>
            ))}
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
            <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
            <p className="text-gray-600">Configure your RLHF dashboard and model preferences</p>
          </div>
          <CogIcon className="h-8 w-8 text-humain-400" />
        </div>

        {/* API Keys Section */}
        <div className="humain-card">
          <div className="humain-card-content">
            <div className="flex items-center mb-4">
              <KeyIcon className="h-6 w-6 text-humain-400 mr-2" />
              <h3 className="text-lg font-semibold text-gray-900">API Configuration</h3>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="humain-label">DeepSeek API Key</label>
                <div className="flex space-x-2">
                  <input
                    type="password"
                    placeholder="sk-..."
                    value={settings.apiKeys.deepseek}
                    onChange={(e) => setSettings(prev => ({
                      ...prev,
                      apiKeys: { ...prev.apiKeys, deepseek: e.target.value }
                    }))}
                    className="humain-input flex-1"
                  />
                  <button
                    onClick={() => testApiConnection('deepseek')}
                    disabled={!settings.apiKeys.deepseek || testingApi === 'deepseek'}
                    className="px-3 py-2 border border-gray-300 rounded-lg text-sm hover:bg-gray-50 disabled:opacity-50 flex items-center"
                  >
                    {testingApi === 'deepseek' ? (
                      'Testing...'
                    ) : apiStatus.deepseek === true ? (
                      <CheckCircleIcon className="h-4 w-4 text-green-500" />
                    ) : apiStatus.deepseek === false ? (
                      <XCircleIcon className="h-4 w-4 text-red-500" />
                    ) : (
                      'Test'
                    )}
                  </button>
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  Get your key from <a href="https://platform.deepseek.com" className="text-humain-400 hover:text-humain-500">DeepSeek Platform</a>
                </p>
              </div>

              <div>
                <label className="humain-label">OpenAI API Key</label>
                <div className="flex space-x-2">
                  <input
                    type="password"
                    placeholder="sk-..."
                    value={settings.apiKeys.openai}
                    onChange={(e) => setSettings(prev => ({
                      ...prev,
                      apiKeys: { ...prev.apiKeys, openai: e.target.value }
                    }))}
                    className="humain-input flex-1"
                  />
                  <button
                    onClick={() => testApiConnection('openai')}
                    disabled={!settings.apiKeys.openai || testingApi === 'openai'}
                    className="px-3 py-2 border border-gray-300 rounded-lg text-sm hover:bg-gray-50 disabled:opacity-50 flex items-center"
                  >
                    {testingApi === 'openai' ? (
                      'Testing...'
                    ) : apiStatus.openai === true ? (
                      <CheckCircleIcon className="h-4 w-4 text-green-500" />
                    ) : apiStatus.openai === false ? (
                      <XCircleIcon className="h-4 w-4 text-red-500" />
                    ) : (
                      'Test'
                    )}
                  </button>
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  Get your key from <a href="https://platform.openai.com" className="text-humain-400 hover:text-humain-500">OpenAI Platform</a>
                </p>
              </div>

              <div className="md:col-span-2">
                <label className="humain-label">LM Studio Base URL</label>
                <div className="flex space-x-2">
                  <input
                    type="url"
                    value={settings.apiKeys.lmstudio}
                    onChange={(e) => setSettings(prev => ({
                      ...prev,
                      apiKeys: { ...prev.apiKeys, lmstudio: e.target.value }
                    }))}
                    className="humain-input flex-1"
                  />
                  <button
                    onClick={() => testApiConnection('lmstudio')}
                    disabled={!settings.apiKeys.lmstudio || testingApi === 'lmstudio'}
                    className="px-3 py-2 border border-gray-300 rounded-lg text-sm hover:bg-gray-50 disabled:opacity-50 flex items-center"
                  >
                    {testingApi === 'lmstudio' ? (
                      'Testing...'
                    ) : apiStatus.lmstudio === true ? (
                      <CheckCircleIcon className="h-4 w-4 text-green-500" />
                    ) : apiStatus.lmstudio === false ? (
                      <XCircleIcon className="h-4 w-4 text-red-500" />
                    ) : (
                      'Test'
                    )}
                  </button>
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  Local LM Studio server endpoint (no API key required)
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Notifications Section */}
        <div className="humain-card">
          <div className="humain-card-content">
            <div className="flex items-center mb-4">
              <BellIcon className="h-6 w-6 text-humain-400 mr-2" />
              <h3 className="text-lg font-semibold text-gray-900">Notifications</h3>
            </div>
            
            <div className="space-y-4">
              {Object.entries(settings.notifications).map(([key, value]) => (
                <div key={key} className="flex items-center justify-between">
                  <div>
                    <div className="font-medium text-gray-900">
                      {key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
                    </div>
                    <div className="text-sm text-gray-600">
                      {key === 'email' && 'Receive email notifications for important events'}
                      {key === 'drift_alerts' && 'Get notified when model drift is detected'}
                      {key === 'training_complete' && 'Alert when training jobs finish'}
                      {key === 'weekly_reports' && 'Weekly performance summary emails'}
                    </div>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={value}
                      onChange={(e) => setSettings(prev => ({
                        ...prev,
                        notifications: { ...prev.notifications, [key]: e.target.checked }
                      }))}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-humain-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-humain-400"></div>
                  </label>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Dashboard Settings */}
        <div className="humain-card">
          <div className="humain-card-content">
            <div className="flex items-center mb-4">
              <UserIcon className="h-6 w-6 text-humain-400 mr-2" />
              <h3 className="text-lg font-semibold text-gray-900">Dashboard Preferences</h3>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="humain-label">Auto Refresh</label>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.dashboard.auto_refresh}
                    onChange={(e) => setSettings(prev => ({
                      ...prev,
                      dashboard: { ...prev.dashboard, auto_refresh: e.target.checked }
                    }))}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-humain-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-humain-400"></div>
                </label>
              </div>

              <div>
                <label className="humain-label">Refresh Interval (seconds)</label>
                <input
                  type="number"
                  min="10"
                  max="300"
                  value={settings.dashboard.refresh_interval}
                  onChange={(e) => setSettings(prev => ({
                    ...prev,
                    dashboard: { ...prev.dashboard, refresh_interval: parseInt(e.target.value) }
                  }))}
                  className="humain-input"
                />
              </div>

              <div>
                <label className="humain-label">Theme</label>
                <select
                  value={settings.dashboard.theme}
                  onChange={(e) => setSettings(prev => ({
                    ...prev,
                    dashboard: { ...prev.dashboard, theme: e.target.value }
                  }))}
                  className="humain-input"
                >
                  <option value="light">Light</option>
                  <option value="dark">Dark</option>
                  <option value="auto">Auto</option>
                </select>
              </div>

              <div>
                <label className="humain-label">Show Debug Info</label>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.dashboard.show_debug}
                    onChange={(e) => setSettings(prev => ({
                      ...prev,
                      dashboard: { ...prev.dashboard, show_debug: e.target.checked }
                    }))}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-humain-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-humain-400"></div>
                </label>
              </div>
            </div>
          </div>
        </div>

        {/* Model Settings */}
        <div className="humain-card">
          <div className="humain-card-content">
            <div className="flex items-center mb-4">
              <ServerIcon className="h-6 w-6 text-humain-400 mr-2" />
              <h3 className="text-lg font-semibold text-gray-900">Model Configuration</h3>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="humain-label">Default Provider</label>
                <select
                  value={settings.model.default_provider}
                  onChange={(e) => setSettings(prev => ({
                    ...prev,
                    model: { ...prev.model, default_provider: e.target.value }
                  }))}
                  className="humain-input"
                >
                  <option value="deepseek">DeepSeek</option>
                  <option value="openai">OpenAI</option>
                  <option value="lmstudio">LM Studio</option>
                </select>
              </div>

              <div>
                <label className="humain-label">Temperature</label>
                <input
                  type="number"
                  min="0"
                  max="2"
                  step="0.1"
                  value={settings.model.temperature}
                  onChange={(e) => setSettings(prev => ({
                    ...prev,
                    model: { ...prev.model, temperature: parseFloat(e.target.value) }
                  }))}
                  className="humain-input"
                />
              </div>

              <div>
                <label className="humain-label">Max Tokens</label>
                <input
                  type="number"
                  min="100"
                  max="4000"
                  value={settings.model.max_tokens}
                  onChange={(e) => setSettings(prev => ({
                    ...prev,
                    model: { ...prev.model, max_tokens: parseInt(e.target.value) }
                  }))}
                  className="humain-input"
                />
              </div>

              <div>
                <label className="humain-label">Batch Size</label>
                <input
                  type="number"
                  min="1"
                  max="128"
                  value={settings.model.batch_size}
                  onChange={(e) => setSettings(prev => ({
                    ...prev,
                    model: { ...prev.model, batch_size: parseInt(e.target.value) }
                  }))}
                  className="humain-input"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex justify-end space-x-4">
          <button
            onClick={handleReset}
            className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors"
          >
            Reset to Defaults
          </button>
          <button
            onClick={handleSave}
            disabled={isSaving}
            className="humain-btn-primary disabled:opacity-50"
          >
            {isSaving ? 'Saving...' : 'Save Settings'}
          </button>
        </div>
      </div>
    </DashboardLayout>
  )
} 