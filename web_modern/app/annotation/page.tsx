'use client'

import React, { useState, useEffect } from 'react'
import { PencilIcon, CheckIcon, XMarkIcon, ChatBubbleLeftRightIcon, SparklesIcon, PlusIcon } from '@heroicons/react/24/outline'
import DashboardLayout from '@/components/layout/DashboardLayout'

interface Annotation {
  id: string
  timestamp: string
  prompt: string
  response_a: string
  response_b: string
  human_choice?: string
  model_choice?: string
  confidence?: number
  correct?: boolean
}

export default function AnnotationPage() {
  const [isLoading, setIsLoading] = useState(true)
  const [annotations, setAnnotations] = useState<Annotation[]>([])
  const [currentAnnotation, setCurrentAnnotation] = useState<Annotation | null>(null)
  const [selectedChoice, setSelectedChoice] = useState<string>('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  
  // New response generation state
  const [newPrompt, setNewPrompt] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [showPromptInput, setShowPromptInput] = useState(false)

  useEffect(() => {
    loadAnnotations()
  }, [])

  const loadAnnotations = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/annotations')
      const data = await response.json()
      setAnnotations(data.annotations || [])
      
      // Set first unannotated item as current
      const unannotated = data.annotations?.find((ann: Annotation) => !ann.human_choice)
      setCurrentAnnotation(unannotated || data.annotations?.[0] || null)
      
      setIsLoading(false)
    } catch (error) {
      console.error('Error loading annotations:', error)
      setIsLoading(false)
    }
  }

  const generateResponses = async () => {
    if (!newPrompt.trim()) {
      alert('Please enter a prompt')
      return
    }

    setIsGenerating(true)
    
    try {
      // Generate first response
      const response1 = await fetch('http://localhost:8000/api/annotations/generate-response', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: newPrompt }),
      })
      const result1 = await response1.json()

      if (!result1.success) {
        throw new Error(result1.error || 'Failed to generate first response')
      }

      // Generate second response (with slightly different temperature for variation)
      const response2 = await fetch('http://localhost:8000/api/annotations/generate-response', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: newPrompt }),
      })
      const result2 = await response2.json()

      if (!result2.success) {
        throw new Error(result2.error || 'Failed to generate second response')
      }

      // Create new annotation from generated responses
      const newAnnotation: Annotation = {
        id: `generated_${Date.now()}`,
        timestamp: new Date().toISOString(),
        prompt: newPrompt,
        response_a: result1.response,
        response_b: result2.response
      }

      // Add to annotations and set as current
      const updatedAnnotations = [newAnnotation, ...annotations]
      setAnnotations(updatedAnnotations)
      setCurrentAnnotation(newAnnotation)
      
      // Reset form
      setNewPrompt('')
      setShowPromptInput(false)
      setSelectedChoice('')

      alert('✅ Responses generated successfully! Please review and annotate.')

    } catch (error) {
      console.error('Error generating responses:', error)
      alert(`❌ Error generating responses: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      setIsGenerating(false)
    }
  }

  const submitAnnotation = async () => {
    if (!currentAnnotation || !selectedChoice) return

    setIsSubmitting(true)
    
    try {
      // Save annotation to backend
      const response = await fetch('http://localhost:8000/api/annotations/save', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          id: currentAnnotation.id,
          choice: selectedChoice,
          prompt: currentAnnotation.prompt,
          response_a: currentAnnotation.response_a,
          response_b: currentAnnotation.response_b,
          timestamp: new Date().toISOString()
        }),
      })

      const result = await response.json()

      if (!result.success) {
        throw new Error(result.error || 'Failed to save annotation')
      }

      // Update local state
      const updatedAnnotations = annotations.map(ann => 
        ann.id === currentAnnotation.id 
          ? { ...ann, human_choice: selectedChoice }
          : ann
      )
      setAnnotations(updatedAnnotations)

      // Move to next unannotated item
      const nextUnannotated = updatedAnnotations.find(ann => !ann.human_choice)
      setCurrentAnnotation(nextUnannotated || null)
      setSelectedChoice('')

      // Show success message
      alert(`✅ Annotation saved successfully! Added to RLHF training data.`)
      
    } catch (error) {
      console.error('Error submitting annotation:', error)
      alert(`❌ Error saving annotation: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      setIsSubmitting(false)
    }
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

  return (
    <DashboardLayout>
      <div className="space-y-6 humain-fade-in">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Human Feedback Collection</h1>
            <p className="text-gray-600">Generate model responses and provide preference annotations for RLHF training</p>
          </div>
          <div className="flex items-center space-x-4">
            <PencilIcon className="h-8 w-8 text-humain-400" />
            <button
              onClick={() => setShowPromptInput(!showPromptInput)}
              className="humain-btn-primary flex items-center space-x-2"
            >
              <PlusIcon className="h-4 w-4" />
              <span>New Annotation</span>
            </button>
          </div>
        </div>

        {/* Generate New Response Interface */}
        {showPromptInput && (
          <div className="humain-card">
            <div className="humain-card-content">
              <div className="flex items-center mb-4">
                <SparklesIcon className="h-6 w-6 text-humain-400 mr-2" />
                <h3 className="text-lg font-semibold text-gray-900">Generate New Responses</h3>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="humain-label">Enter a prompt to generate model responses</label>
                  <textarea
                    value={newPrompt}
                    onChange={(e) => setNewPrompt(e.target.value)}
                    placeholder="Type your prompt here... (e.g., 'Explain quantum computing in simple terms')"
                    className="humain-input min-h-[100px] resize-y"
                    disabled={isGenerating}
                  />
                </div>
                
                <div className="flex justify-end space-x-3">
                  <button
                    onClick={() => {
                      setShowPromptInput(false)
                      setNewPrompt('')
                    }}
                    className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors"
                    disabled={isGenerating}
                  >
                    Cancel
                  </button>
                  
                  <button
                    onClick={generateResponses}
                    disabled={!newPrompt.trim() || isGenerating}
                    className="humain-btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
                  >
                    <SparklesIcon className="h-4 w-4" />
                    <span>{isGenerating ? 'Generating...' : 'Generate Responses'}</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Progress Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="humain-card">
            <div className="humain-card-content text-center">
              <div className="text-2xl font-bold text-humain-600 mb-2">
                {annotations.length}
              </div>
              <div className="text-sm text-gray-600">Total Annotations</div>
            </div>
          </div>

          <div className="humain-card">
            <div className="humain-card-content text-center">
              <div className="text-2xl font-bold text-humain-600 mb-2">
                {annotations.filter(ann => ann.human_choice).length}
              </div>
              <div className="text-sm text-gray-600">Completed</div>
            </div>
          </div>

          <div className="humain-card">
            <div className="humain-card-content text-center">
              <div className="text-2xl font-bold text-humain-600 mb-2">
                {annotations.filter(ann => !ann.human_choice).length}
              </div>
              <div className="text-sm text-gray-600">Remaining</div>
            </div>
          </div>

          <div className="humain-card">
            <div className="humain-card-content text-center">
              <div className="text-2xl font-bold text-humain-600 mb-2">
                {annotations.length > 0 ? Math.round((annotations.filter(ann => ann.human_choice).length / annotations.length) * 100) : 0}%
              </div>
              <div className="text-sm text-gray-600">Progress</div>
            </div>
          </div>
        </div>

        {/* Annotation Interface */}
        {currentAnnotation ? (
          <div className="humain-card">
            <div className="humain-card-content">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-gray-900">Current Annotation</h3>
                <div className="flex items-center space-x-4">
                  <div className="text-sm text-gray-500">ID: {currentAnnotation.id}</div>
                  {currentAnnotation.id.startsWith('generated_') && (
                    <span className="px-2 py-1 bg-green-100 text-green-800 text-xs font-medium rounded-full">
                      Generated
                    </span>
                  )}
                </div>
              </div>

              {/* Prompt */}
              <div className="mb-6">
                <label className="humain-label">Prompt</label>
                <div className="p-4 bg-gray-50 rounded-lg border">
                  <p className="text-gray-800 leading-relaxed">{currentAnnotation.prompt}</p>
                </div>
              </div>

              {/* Response Comparison */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                {/* Response A */}
                <div className={`p-4 rounded-lg border-2 transition-all cursor-pointer ${
                  selectedChoice === 'A' 
                    ? 'border-humain-400 bg-humain-50' 
                    : 'border-gray-200 hover:border-gray-300'
                }`} onClick={() => setSelectedChoice('A')}>
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-2">
                      <div className={`w-6 h-6 rounded-full flex items-center justify-center text-white font-semibold ${
                        selectedChoice === 'A' ? 'bg-humain-400' : 'bg-gray-400'
                      }`}>
                        A
                      </div>
                      <span className="font-medium text-gray-900">Response A</span>
                    </div>
                    {selectedChoice === 'A' && (
                      <CheckIcon className="h-5 w-5 text-humain-400" />
                    )}
                  </div>
                  <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">{currentAnnotation.response_a}</p>
                </div>

                {/* Response B */}
                <div className={`p-4 rounded-lg border-2 transition-all cursor-pointer ${
                  selectedChoice === 'B' 
                    ? 'border-humain-400 bg-humain-50' 
                    : 'border-gray-200 hover:border-gray-300'
                }`} onClick={() => setSelectedChoice('B')}>
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-2">
                      <div className={`w-6 h-6 rounded-full flex items-center justify-center text-white font-semibold ${
                        selectedChoice === 'B' ? 'bg-humain-400' : 'bg-gray-400'
                      }`}>
                        B
                      </div>
                      <span className="font-medium text-gray-900">Response B</span>
                    </div>
                    {selectedChoice === 'B' && (
                      <CheckIcon className="h-5 w-5 text-humain-400" />
                    )}
                  </div>
                  <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">{currentAnnotation.response_b}</p>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex justify-between items-center">
                <div className="flex items-center space-x-4">
                  <div className="text-sm text-gray-600">
                    Select the better response by clicking on it
                  </div>
                </div>
                
                <div className="flex space-x-3">
                  <button
                    onClick={() => setSelectedChoice('')}
                    className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors"
                    disabled={isSubmitting}
                  >
                    Clear
                  </button>
                  
                  <button
                    onClick={submitAnnotation}
                    disabled={!selectedChoice || isSubmitting}
                    className="humain-btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isSubmitting ? 'Submitting...' : 'Submit Annotation'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="humain-card">
            <div className="humain-card-content text-center py-12">
              {annotations.length === 0 ? (
                <>
                  <SparklesIcon className="h-16 w-16 text-humain-400 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">No Annotations Yet</h3>
                  <p className="text-gray-600 mb-4">
                    Start by generating new model responses to annotate, or check if your API is configured in Settings.
                  </p>
                  <button
                    onClick={() => setShowPromptInput(true)}
                    className="humain-btn-primary"
                  >
                    Generate First Annotation
                  </button>
                </>
              ) : (
                <>
                  <CheckIcon className="h-16 w-16 text-green-500 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">All Annotations Complete!</h3>
                  <p className="text-gray-600 mb-4">
                    You've completed all available annotations. Great work contributing to model improvement!
                  </p>
                  <button
                    onClick={() => setShowPromptInput(true)}
                    className="humain-btn-primary"
                  >
                    Generate More Annotations
                  </button>
                </>
              )}
            </div>
          </div>
        )}

        {/* Recent Annotations History */}
        {annotations.length > 0 && (
          <div className="humain-card">
            <div className="humain-card-content">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Annotations</h3>
              
              <div className="space-y-3">
                {annotations.slice(0, 5).map((annotation) => (
                  <div key={annotation.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="text-sm font-medium text-gray-900">
                        ID: {annotation.id}
                      </div>
                      <div className="text-sm text-gray-600">
                        {new Date(annotation.timestamp).toLocaleString()}
                      </div>
                      {annotation.id.startsWith('generated_') && (
                        <span className="px-2 py-1 bg-green-100 text-green-800 text-xs font-medium rounded-full">
                          Generated
                        </span>
                      )}
                    </div>
                    
                    <div className="flex items-center space-x-3">
                      {annotation.human_choice ? (
                        <div className="flex items-center space-x-2">
                          <CheckIcon className="h-4 w-4 text-green-500" />
                          <span className="text-sm font-medium text-green-600">
                            Choice: {annotation.human_choice}
                          </span>
                        </div>
                      ) : (
                        <div className="flex items-center space-x-2">
                          <XMarkIcon className="h-4 w-4 text-gray-400" />
                          <span className="text-sm text-gray-500">Pending</span>
                        </div>
                      )}
                      
                      {annotation.confidence && (
                        <div className="text-sm text-gray-600">
                          Confidence: {(annotation.confidence * 100).toFixed(1)}%
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  )
} 