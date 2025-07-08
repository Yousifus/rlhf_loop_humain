'use client'

import React, { useState, useEffect, useRef } from 'react'
import { ChatBubbleLeftRightIcon, PaperAirplaneIcon, CogIcon, ExclamationTriangleIcon } from '@heroicons/react/24/outline'
import DashboardLayout from '@/components/layout/DashboardLayout'

interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
  provider?: string
  tokens?: number
  error?: boolean
}

interface Provider {
  id: string
  name: string
  icon: string
  available: boolean
  status: string
}

interface Settings {
  apiKeys: {
    deepseek: string
    openai: string
    lmstudio: string
  }
  model: {
    default_provider: string
    temperature: number
    max_tokens: number
  }
}

export default function ChatPage() {
  const [isLoading, setIsLoading] = useState(true)
  const [messages, setMessages] = useState<Message[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [providers, setProviders] = useState<Provider[]>([])
  const [globalSettings, setGlobalSettings] = useState<Settings | null>(null)
  const [chatSettings, setChatSettings] = useState({
    temperature: 0.7,
    max_tokens: 500,
    system_message: 'You are a helpful AI assistant.'
  })
  
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    loadSettings()
    loadProviders()
    // Add welcome message
    setMessages([{
      id: '1',
      role: 'system',
      content: 'Welcome to the RLHF Chat Interface! This uses your configured API provider from Settings.',
      timestamp: new Date()
    }])
    setIsLoading(false)
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const loadSettings = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/settings')
      if (response.ok) {
        const data = await response.json()
        setGlobalSettings(data)
        
        // Update chat settings with global defaults
        setChatSettings(prev => ({
          ...prev,
          temperature: data.model?.temperature || prev.temperature,
          max_tokens: data.model?.max_tokens || prev.max_tokens
        }))
      } else {
        console.warn('Settings endpoint not available, using defaults')
        // Set default settings if endpoint is not available
        setGlobalSettings({
          apiKeys: {
            deepseek: '',
            openai: '',
            lmstudio: 'http://localhost:1234'
          },
          model: {
            default_provider: 'deepseek',
            temperature: 0.7,
            max_tokens: 500
          }
        })
      }
    } catch (error) {
      console.error('Error loading settings:', error)
      // Set default settings on error
      setGlobalSettings({
        apiKeys: {
          deepseek: '',
          openai: '',
          lmstudio: 'http://localhost:1234'
        },
        model: {
          default_provider: 'deepseek',
          temperature: 0.7,
          max_tokens: 500
        }
      })
    }
  }

  const loadProviders = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/model-providers')
      const data = await response.json()
      setProviders(data.providers || [])
    } catch (error) {
      console.error('Error loading providers:', error)
    }
  }

  const sendMessage = async () => {
    if (!inputMessage.trim() || isSending) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputMessage.trim(),
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    const currentMessage = inputMessage.trim()
    setInputMessage('')
    setIsSending(true)

    try {
      // Use the annotation response endpoint which uses configured settings
      const response = await fetch('http://localhost:8000/api/annotations/generate-response', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          prompt: `${chatSettings.system_message}\n\nUser: ${currentMessage}\nAssistant:`
        })
      })

      const data = await response.json()

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.success 
          ? data.response 
          : `❌ Error: ${data.error}\n\n💡 Please check your API configuration in Settings.`,
        timestamp: new Date(),
        provider: data.provider,
        tokens: data.tokens_used,
        error: !data.success
      }

      setMessages(prev => [...prev, assistantMessage])

    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `❌ Connection error: ${error instanceof Error ? error.message : String(error)}\n\n💡 Please check your API configuration in Settings.`,
        timestamp: new Date(),
        error: true
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsSending(false)
    }
  }

  const clearChat = () => {
    setMessages([{
      id: '1',
      role: 'system',
      content: 'Chat cleared. Ready for new conversation.',
      timestamp: new Date()
    }])
  }

  const testProvider = async () => {
    if (!globalSettings?.model?.default_provider) {
      alert('No provider configured. Please configure in Settings.')
      return
    }

    setIsSending(true)
    
    const testMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: '[SYSTEM TEST] Testing provider connection...',
      timestamp: new Date()
    }
    setMessages(prev => [...prev, testMessage])

    try {
      const response = await fetch(`http://localhost:8000/api/model-providers/${globalSettings.model.default_provider}/test`, {
        method: 'POST'
      })
      const data = await response.json()

      const resultMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.success 
          ? `✅ Provider test successful!\n\nProvider: ${data.provider}\nResponse: ${data.response}\nTokens used: ${data.tokens_used || 'N/A'}`
          : `❌ Provider test failed: ${data.error}\n\n💡 Please check your API configuration in Settings.`,
        timestamp: new Date(),
        provider: data.provider,
        error: !data.success
      }

      setMessages(prev => [...prev, resultMessage])
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `❌ Connection error: ${error instanceof Error ? error.message : String(error)}`,
        timestamp: new Date(),
        error: true
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsSending(false)
    }
  }

  const getCurrentProvider = () => {
    if (!globalSettings?.model?.default_provider) return null
    return providers.find(p => p.id === globalSettings.model.default_provider)
  }

  const hasApiConfigured = () => {
    if (!globalSettings?.model?.default_provider) return false
    const provider = globalSettings.model.default_provider
    if (provider === 'lmstudio') return !!globalSettings.apiKeys?.lmstudio
    if (provider === 'deepseek') return !!globalSettings.apiKeys?.deepseek
    if (provider === 'openai') return !!globalSettings.apiKeys?.openai
    return false
  }

  if (isLoading) {
    return (
      <DashboardLayout>
        <div className="space-y-6 animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/3"></div>
          <div className="h-96 bg-gray-200 rounded-xl"></div>
        </div>
      </DashboardLayout>
    )
  }

  const currentProvider = getCurrentProvider()
  const apiConfigured = hasApiConfigured()

  return (
    <DashboardLayout>
      <div className="space-y-6 humain-fade-in">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Model Chat Interface</h1>
            <p className="text-gray-600">
              Test and interact with your configured AI provider: {currentProvider?.name || 'Not configured'}
            </p>
          </div>
          <div className="flex items-center space-x-4">
            <ChatBubbleLeftRightIcon className="h-8 w-8 text-humain-400" />
            <button
              onClick={() => window.location.href = '/settings'}
              className="humain-btn-primary flex items-center space-x-2"
            >
              <CogIcon className="h-4 w-4" />
              <span>Settings</span>
            </button>
          </div>
        </div>

        {/* Configuration Status */}
        {!apiConfigured && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <div className="flex items-center">
              <ExclamationTriangleIcon className="h-5 w-5 text-yellow-400 mr-2" />
              <div className="text-sm">
                <strong className="text-yellow-800">API Not Configured:</strong>
                <span className="text-yellow-700 ml-1">
                  Please configure your API keys in Settings to use the chat interface.
                </span>
                <button
                  onClick={() => window.location.href = '/settings'}
                  className="ml-2 text-yellow-800 underline hover:text-yellow-900"
                >
                  Go to Settings
                </button>
              </div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Chat Interface */}
          <div className="lg:col-span-3">
            <div className="humain-card h-96 flex flex-col">
              {/* Chat Messages */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map((message) => (
                  <div key={message.id} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-xs lg:max-w-md xl:max-w-lg px-4 py-2 rounded-lg ${
                      message.role === 'user' 
                        ? 'bg-humain-400 text-white'
                        : message.role === 'system'
                        ? 'bg-gray-100 text-gray-700 border'
                        : message.error
                        ? 'bg-red-50 text-red-700 border border-red-200'
                        : 'bg-gray-50 text-gray-900 border'
                    }`}>
                      <div className="text-sm leading-relaxed whitespace-pre-wrap">
                        {message.content}
                      </div>
                      
                      {/* Message metadata */}
                      <div className={`text-xs mt-2 opacity-75 ${
                        message.role === 'user' ? 'text-white' : 'text-gray-500'
                      }`}>
                        {message.timestamp.toLocaleTimeString()}
                        {message.provider && ` • ${message.provider}`}
                        {message.tokens && ` • ${message.tokens} tokens`}
                      </div>
                    </div>
                  </div>
                ))}
                
                {isSending && (
                  <div className="flex justify-start">
                    <div className="bg-gray-50 border px-4 py-2 rounded-lg">
                      <div className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                        <span className="text-sm text-gray-500 ml-2">Thinking...</span>
                      </div>
                    </div>
                  </div>
                )}
                
                <div ref={messagesEndRef} />
              </div>
              
              {/* Message Input */}
              <div className="border-t p-4">
                <div className="flex space-x-2">
                  <input
                    type="text"
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                    placeholder={apiConfigured ? "Type your message..." : "Configure API in Settings first..."}
                    className="flex-1 humain-input"
                    disabled={isSending || !apiConfigured}
                  />
                  <button
                    onClick={sendMessage}
                    disabled={!inputMessage.trim() || isSending || !apiConfigured}
                    className="humain-btn-primary disabled:opacity-50 disabled:cursor-not-allowed px-4"
                  >
                    <PaperAirplaneIcon className="h-4 w-4" />
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Settings Panel */}
          <div className="space-y-6">
            {/* Provider Status */}
            <div className="humain-card">
              <div className="humain-card-content">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Current Provider</h3>
                
                <div className="space-y-3">
                  {currentProvider ? (
                    <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center space-x-2">
                        <span className="text-lg">{currentProvider.icon}</span>
                        <span className="font-medium text-gray-900">{currentProvider.name}</span>
                      </div>
                      <div className={`px-2 py-1 rounded-full text-xs font-medium ${
                        apiConfigured
                          ? 'bg-green-100 text-green-800'
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {apiConfigured ? 'Configured' : 'Not Configured'}
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-4 text-gray-500">
                      <p>No provider selected</p>
                      <button
                        onClick={() => window.location.href = '/settings'}
                        className="text-humain-400 hover:text-humain-500 text-sm underline mt-1"
                      >
                        Configure in Settings
                      </button>
                    </div>
                  )}
                  
                  {apiConfigured && (
                    <button
                      onClick={testProvider}
                      disabled={isSending}
                      className="humain-btn-primary w-full disabled:opacity-50"
                    >
                      Test Connection
                    </button>
                  )}
                  
                  <button
                    onClick={loadSettings}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors text-sm"
                  >
                    🔄 Refresh Configuration
                  </button>
                </div>
              </div>
            </div>

            {/* Chat Settings */}
            <div className="humain-card">
              <div className="humain-card-content">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Chat Settings</h3>
                
                <div className="space-y-4">
                  <div>
                    <label className="humain-label">Temperature</label>
                    <input
                      type="range"
                      min="0"
                      max="2"
                      step="0.1"
                      value={chatSettings.temperature}
                      onChange={(e) => setChatSettings(prev => ({ ...prev, temperature: parseFloat(e.target.value) }))}
                      className="w-full"
                    />
                    <div className="text-sm text-gray-600 mt-1">{chatSettings.temperature}</div>
                  </div>
                  
                  <div>
                    <label className="humain-label">Max Tokens</label>
                    <input
                      type="number"
                      min="10"
                      max="2000"
                      value={chatSettings.max_tokens}
                      onChange={(e) => setChatSettings(prev => ({ ...prev, max_tokens: parseInt(e.target.value) }))}
                      className="humain-input"
                    />
                  </div>
                  
                  <div>
                    <label className="humain-label">System Message</label>
                    <textarea
                      value={chatSettings.system_message}
                      onChange={(e) => setChatSettings(prev => ({ ...prev, system_message: e.target.value }))}
                      className="humain-input"
                      rows={3}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Chat Actions */}
            <div className="humain-card">
              <div className="humain-card-content">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Actions</h3>
                
                <div className="space-y-2">
                  <button
                    onClick={clearChat}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors"
                  >
                    Clear Chat
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  )
} 