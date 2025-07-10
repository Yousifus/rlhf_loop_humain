'use client'

import React, { useState, useEffect } from 'react'
import { PencilIcon, CheckIcon, XMarkIcon, ChatBubbleLeftRightIcon, SparklesIcon, PlusIcon, StarIcon } from '@heroicons/react/24/outline'
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
  annotation_saved?: boolean
  rlhf_generated?: boolean
  prompt_metadata?: any
}

interface QualityRatings {
  accuracy: number
  clarity: number
  completeness: number
  helpfulness: number
  creativity: number
}

interface ChoiceReasons {
  better_explanation: boolean
  more_accurate: boolean
  clearer_structure: boolean
  better_examples: boolean
  more_comprehensive: boolean
  engaging_tone: boolean
  fewer_errors: boolean
}

interface RejectionReasons {
  factual_errors: boolean
  confusing_explanation: boolean
  missing_information: boolean
  poor_organization: boolean
  inappropriate_tone: boolean
  length_issues: boolean
}

interface RichAnnotationData {
  choice_confidence: number
  chosen_quality: QualityRatings
  rejected_quality: QualityRatings
  choice_reasons: ChoiceReasons
  rejection_reasons: RejectionReasons
  additional_feedback: string
}

export default function AnnotationPage() {
  const [isLoading, setIsLoading] = useState(true)
  const [annotations, setAnnotations] = useState<Annotation[]>([])
  const [currentAnnotation, setCurrentAnnotation] = useState<Annotation | null>(null)
  const [selectedChoice, setSelectedChoice] = useState<string>('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  
  // Custom prompt generation state
  const [customPrompt, setCustomPrompt] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [showPromptInput, setShowPromptInput] = useState(false)

  // Rich annotation state
  const [showRichAnnotation, setShowRichAnnotation] = useState(false)
  const [richAnnotation, setRichAnnotation] = useState<RichAnnotationData>({
    choice_confidence: 70,
    chosen_quality: { accuracy: 7, clarity: 7, completeness: 7, helpfulness: 7, creativity: 5 },
    rejected_quality: { accuracy: 5, clarity: 5, completeness: 5, helpfulness: 5, creativity: 5 },
    choice_reasons: {
      better_explanation: false,
      more_accurate: false,
      clearer_structure: false,
      better_examples: false,
      more_comprehensive: false,
      engaging_tone: false,
      fewer_errors: false
    },
    rejection_reasons: {
      factual_errors: false,
      confusing_explanation: false,
      missing_information: false,
      poor_organization: false,
      inappropriate_tone: false,
      length_issues: false
    },
    additional_feedback: ''
  })

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

  const handleChoiceSelection = (choice: string) => {
    setSelectedChoice(choice)
    
    // Reset rich annotation data
    setRichAnnotation({
      choice_confidence: 70,
      chosen_quality: { accuracy: 7, clarity: 7, completeness: 7, helpfulness: 7, creativity: 5 },
      rejected_quality: { accuracy: 5, clarity: 5, completeness: 5, helpfulness: 5, creativity: 5 },
      choice_reasons: {
        better_explanation: false,
        more_accurate: false,
        clearer_structure: false,
        better_examples: false,
        more_comprehensive: false,
        engaging_tone: false,
        fewer_errors: false
      },
      rejection_reasons: {
        factual_errors: false,
        confusing_explanation: false,
        missing_information: false,
        poor_organization: false,
        inappropriate_tone: false,
        length_issues: false
      },
      additional_feedback: ''
    })
    
    // Show rich annotation interface after brief delay
    setTimeout(() => {
      setShowRichAnnotation(true)
    }, 300)
  }

  const updateQuality = (type: 'chosen' | 'rejected', dimension: keyof QualityRatings, value: number) => {
    setRichAnnotation(prev => ({
      ...prev,
      [type === 'chosen' ? 'chosen_quality' : 'rejected_quality']: {
        ...prev[type === 'chosen' ? 'chosen_quality' : 'rejected_quality'],
        [dimension]: value
      }
    }))
  }

  const updateChoiceReason = (reason: keyof ChoiceReasons, checked: boolean) => {
    setRichAnnotation(prev => ({
      ...prev,
      choice_reasons: {
        ...prev.choice_reasons,
        [reason]: checked
      }
    }))
  }

  const updateRejectionReason = (reason: keyof RejectionReasons, checked: boolean) => {
    setRichAnnotation(prev => ({
      ...prev,
      rejection_reasons: {
        ...prev.rejection_reasons,
        [reason]: checked
      }
    }))
  }

  const generateFromCustomPrompt = async () => {
    if (!customPrompt.trim()) {
      alert('Please enter a prompt first!')
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
        body: JSON.stringify({ prompt: customPrompt.trim() }),
      })
      const result1 = await response1.json()

      if (!result1.success) {
        throw new Error(result1.error || 'Failed to generate first response')
      }

      // Generate second response
      const response2 = await fetch('http://localhost:8000/api/annotations/generate-response', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: customPrompt.trim() }),
      })
      const result2 = await response2.json()

      if (!result2.success) {
        throw new Error(result2.error || 'Failed to generate second response')
      }

      // Create new annotation from custom prompt
      const newAnnotation: Annotation = {
        id: `custom_${Date.now()}`,
        timestamp: new Date().toISOString(),
        prompt: customPrompt.trim(),
        response_a: result1.response,
        response_b: result2.response,
        rlhf_generated: false // Mark as custom
      }

      // Add to annotations and set as current
      const updatedAnnotations = [newAnnotation, ...annotations]
      setAnnotations(updatedAnnotations)
      setCurrentAnnotation(newAnnotation)
      
      // Reset state
      setCustomPrompt('')
      setShowPromptInput(false)
      setSelectedChoice('')
      setShowRichAnnotation(false)

      alert(`✅ Custom Prompt Processed! 

🧠 Your prompt: "${customPrompt.trim().substring(0, 50)}${customPrompt.length > 50 ? '...' : ''}"
🤖 Generated two responses from ${result1.provider}
👤 Ready for your detailed annotation

Choose which response is better and provide detailed feedback!`)

    } catch (error) {
      console.error('Error generating responses:', error)
      alert(`❌ Error generating responses: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      setIsGenerating(false)
    }
  }

  const generateFromRLHFSystem = async () => {
    setIsGenerating(true)
    
    try {
      // Step 1: Generate prompt using real RLHF prompt generator
      const promptResponse = await fetch('http://localhost:8000/api/prompts/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          difficulty: 'intermediate',
          count: 1 
        }),
      })
      const promptResult = await promptResponse.json()

      if (!promptResult.success) {
        throw new Error(promptResult.error || 'Failed to generate RLHF prompt')
      }

      const generatedPrompt = promptResult.generated_prompt
      const promptId = promptResult.prompt_id

      // Step 2: Generate first response
      const response1 = await fetch('http://localhost:8000/api/annotations/generate-response', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: generatedPrompt }),
      })
      const result1 = await response1.json()

      if (!result1.success) {
        throw new Error(result1.error || 'Failed to generate first response')
      }

      // Step 3: Generate second response
      const response2 = await fetch('http://localhost:8000/api/annotations/generate-response', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: generatedPrompt }),
      })
      const result2 = await response2.json()

      if (!result2.success) {
        throw new Error(result2.error || 'Failed to generate second response')
      }

      // Step 4: Create new annotation from RLHF-generated data
      const newAnnotation: Annotation = {
        id: promptId,
        timestamp: new Date().toISOString(),
        prompt: generatedPrompt,
        response_a: result1.response,
        response_b: result2.response,
        rlhf_generated: true,
        prompt_metadata: promptResult.prompt
      }

      // Add to annotations and set as current
      const updatedAnnotations = [newAnnotation, ...annotations]
      setAnnotations(updatedAnnotations)
      setCurrentAnnotation(newAnnotation)
      
      // Reset state
      setCustomPrompt('')
      setShowPromptInput(false)
      setSelectedChoice('')
      setShowRichAnnotation(false)

      alert(`✅ RLHF Loop Started! 

🧠 Generated professional prompt
🤖 Generated dual responses from ${result1.provider}
👤 Ready for your detailed preference analysis

This follows the full RLHF pipeline with rich annotation data!`)

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
      // Create enhanced RLHF annotation data with rich feedback
      const annotationData = {
        prompt_id: currentAnnotation.id,
        prompt: currentAnnotation.prompt,
        completion_a: currentAnnotation.response_a,
        completion_b: currentAnnotation.response_b,
        preference: selectedChoice === 'A' ? 'Completion A' : 'Completion B',
        selected_completion: selectedChoice === 'A' ? currentAnnotation.response_a : currentAnnotation.response_b,
        rejected_completion: selectedChoice === 'A' ? currentAnnotation.response_b : currentAnnotation.response_a,
        feedback: richAnnotation.additional_feedback,
        quality_metrics: {
          choice_confidence: richAnnotation.choice_confidence,
          chosen_quality: richAnnotation.chosen_quality,
          rejected_quality: richAnnotation.rejected_quality,
          choice_reasons: richAnnotation.choice_reasons,
          rejection_reasons: richAnnotation.rejection_reasons,
          selection_method: 'rich_annotation_v2'
        },
        is_binary_preference: true,
        timestamp: new Date().toISOString()
      }

      // Save through the real RLHF database system
      const response = await fetch('http://localhost:8000/api/annotations/save', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(annotationData),
      })

      const result = await response.json()

      if (!result.success) {
        throw new Error(result.error || 'Failed to save annotation')
      }

      // Update local state
      const updatedAnnotations = annotations.map(ann => 
        ann.id === currentAnnotation.id 
          ? { ...ann, human_choice: selectedChoice, annotation_saved: true }
          : ann
      )
      setAnnotations(updatedAnnotations)

      // Move to next unannotated item
      const nextUnannotated = updatedAnnotations.find(ann => !ann.human_choice)
      setCurrentAnnotation(nextUnannotated || null)
      setSelectedChoice('')
      setShowRichAnnotation(false)

      // Calculate quality scores for success message
      const avgChosenQuality = Object.values(richAnnotation.chosen_quality).reduce((a, b) => a + b, 0) / 5
      const choiceReasonsCount = Object.values(richAnnotation.choice_reasons).filter(Boolean).length

      // Show success message with rich data context
      alert(`✅ Rich annotation saved to RLHF system! 
      
🎯 Your detailed feedback:
• Confidence: ${richAnnotation.choice_confidence}%
• Chosen quality avg: ${avgChosenQuality.toFixed(1)}/10
• ${choiceReasonsCount} reasons selected
• Additional feedback: ${richAnnotation.additional_feedback ? 'Yes' : 'None'}

This rich data will significantly improve vote predictor training!`)
      
    } catch (error) {
      console.error('Error submitting annotation:', error)
      alert(`❌ Error saving to RLHF system: ${error instanceof Error ? error.message : String(error)}`)
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
            <h1 className="text-2xl font-bold text-gray-900">Enhanced Human Feedback Collection</h1>
            <p className="text-gray-600">Provide detailed preference annotations with quality ratings and reasoning for advanced RLHF training</p>
          </div>
          <div className="flex items-center space-x-4">
            <StarIcon className="h-8 w-8 text-humain-400" />
            <button
              onClick={() => setShowPromptInput(!showPromptInput)}
              className="humain-btn-primary flex items-center space-x-2"
            >
              <PlusIcon className="h-4 w-4" />
              <span>Create Rich Annotation</span>
            </button>
          </div>
        </div>

        {/* Create New Annotation Interface */}
        {showPromptInput && (
          <div className="humain-card">
            <div className="humain-card-content">
              <div className="flex items-center mb-4">
                <SparklesIcon className="h-6 w-6 text-humain-400 mr-2" />
                <h3 className="text-lg font-semibold text-gray-900">Create Enhanced Annotation</h3>
              </div>
              
              <div className="space-y-4">
                {/* Custom Prompt Input */}
                <div>
                  <label className="humain-label">Write Your Custom Prompt</label>
                  <textarea
                    value={customPrompt}
                    onChange={(e) => setCustomPrompt(e.target.value)}
                    placeholder="Enter your prompt here... After generating responses, you'll provide detailed quality ratings and reasoning."
                    className="w-full h-32 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-humain-400 focus:border-transparent resize-none"
                  />
                  <div className="text-sm text-gray-500 mt-1">
                    Enhanced mode: You'll rate quality dimensions and provide detailed reasoning for your choice.
                  </div>
                </div>

                <div className="flex justify-end space-x-3">
                  <button
                    onClick={() => {
                      setShowPromptInput(false)
                      setCustomPrompt('')
                    }}
                    className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors"
                    disabled={isGenerating}
                  >
                    Cancel
                  </button>
                  
                  <button
                    onClick={generateFromCustomPrompt}
                    disabled={isGenerating || !customPrompt.trim()}
                    className="humain-btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
                  >
                    <SparklesIcon className="h-4 w-4" />
                    <span>{isGenerating ? 'Generating Responses...' : 'Generate Responses'}</span>
                  </button>
                  
                  <button
                    onClick={generateFromRLHFSystem}
                    disabled={isGenerating}
                    className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
                  >
                    <ChatBubbleLeftRightIcon className="h-4 w-4" />
                    <span>{isGenerating ? 'Running RLHF Pipeline...' : 'Auto-Generate Prompt'}</span>
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
              <div className="text-sm text-gray-600">Rich Data Collected</div>
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
          <div className="space-y-6">
            {/* Basic Choice Interface */}
            <div className="humain-card">
              <div className="humain-card-content">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-lg font-semibold text-gray-900">Step 1: Choose Better Response</h3>
                  <div className="flex items-center space-x-4">
                    <div className="text-sm text-gray-500">ID: {currentAnnotation.id}</div>
                    {currentAnnotation.rlhf_generated ? (
                      <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs font-medium rounded-full">
                        Auto-Generated
                      </span>
                    ) : (
                      <span className="px-2 py-1 bg-green-100 text-green-800 text-xs font-medium rounded-full">
                        Custom Prompt
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
                  }`} onClick={() => handleChoiceSelection('A')}>
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
                  }`} onClick={() => handleChoiceSelection('B')}>
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

                {!showRichAnnotation && (
                  <div className="text-center text-gray-600">
                    Select the better response to continue with detailed quality analysis
                  </div>
                )}
              </div>
            </div>

            {/* Rich Annotation Interface */}
            {showRichAnnotation && selectedChoice && (
              <div className="humain-card">
                <div className="humain-card-content">
                  <div className="flex items-center mb-6">
                    <StarIcon className="h-6 w-6 text-humain-400 mr-2" />
                    <h3 className="text-lg font-semibold text-gray-900">
                      Step 2: Detailed Quality Analysis (Choice: {selectedChoice})
                    </h3>
                  </div>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* Left Column: Quality Ratings */}
                    <div className="space-y-6">
                      <div>
                        <h4 className="font-semibold text-gray-900 mb-4">Quality Ratings (0-10)</h4>
                        
                        {/* Chosen Response Quality */}
                        <div className="mb-6">
                          <h5 className="text-sm font-medium text-green-600 mb-3">Chosen Response (Quality)</h5>
                          {Object.entries(richAnnotation.chosen_quality).map(([key, value]) => (
                            <div key={key} className="mb-3">
                              <div className="flex justify-between text-sm mb-1">
                                <span className="capitalize">{key.replace('_', ' ')}</span>
                                <span className="font-medium">{value}/10</span>
                              </div>
                              <input
                                type="range"
                                min="0"
                                max="10"
                                value={value}
                                onChange={(e) => updateQuality('chosen', key as keyof QualityRatings, parseInt(e.target.value))}
                                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider-green"
                              />
                            </div>
                          ))}
                        </div>

                        {/* Rejected Response Quality */}
                        <div>
                          <h5 className="text-sm font-medium text-red-600 mb-3">Rejected Response (Quality)</h5>
                          {Object.entries(richAnnotation.rejected_quality).map(([key, value]) => (
                            <div key={key} className="mb-3">
                              <div className="flex justify-between text-sm mb-1">
                                <span className="capitalize">{key.replace('_', ' ')}</span>
                                <span className="font-medium">{value}/10</span>
                              </div>
                              <input
                                type="range"
                                min="0"
                                max="10"
                                value={value}
                                onChange={(e) => updateQuality('rejected', key as keyof QualityRatings, parseInt(e.target.value))}
                                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider-red"
                              />
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Confidence Slider */}
                      <div>
                        <h4 className="font-semibold text-gray-900 mb-4">Choice Confidence</h4>
                        <div className="mb-3">
                          <div className="flex justify-between text-sm mb-1">
                            <span>How confident are you in this choice?</span>
                            <span className="font-medium">{richAnnotation.choice_confidence}%</span>
                          </div>
                          <input
                            type="range"
                            min="0"
                            max="100"
                            value={richAnnotation.choice_confidence}
                            onChange={(e) => setRichAnnotation(prev => ({ ...prev, choice_confidence: parseInt(e.target.value) }))}
                            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider-blue"
                          />
                        </div>
                      </div>
                    </div>

                    {/* Right Column: Reasons and Feedback */}
                    <div className="space-y-6">
                      {/* Choice Reasons */}
                      <div>
                        <h4 className="font-semibold text-gray-900 mb-4">Why did you choose this response?</h4>
                        <div className="space-y-2">
                          {Object.entries(richAnnotation.choice_reasons).map(([key, checked]) => (
                            <label key={key} className="flex items-center">
                              <input
                                type="checkbox"
                                checked={checked}
                                onChange={(e) => updateChoiceReason(key as keyof ChoiceReasons, e.target.checked)}
                                className="rounded border-gray-300 text-humain-600 focus:ring-humain-500"
                              />
                              <span className="ml-2 text-sm text-gray-700 capitalize">
                                {key.replace(/_/g, ' ')}
                              </span>
                            </label>
                          ))}
                        </div>
                      </div>

                      {/* Rejection Reasons */}
                      <div>
                        <h4 className="font-semibold text-gray-900 mb-4">What was wrong with the other response?</h4>
                        <div className="space-y-2">
                          {Object.entries(richAnnotation.rejection_reasons).map(([key, checked]) => (
                            <label key={key} className="flex items-center">
                              <input
                                type="checkbox"
                                checked={checked}
                                onChange={(e) => updateRejectionReason(key as keyof RejectionReasons, e.target.checked)}
                                className="rounded border-gray-300 text-red-600 focus:ring-red-500"
                              />
                              <span className="ml-2 text-sm text-gray-700 capitalize">
                                {key.replace(/_/g, ' ')}
                              </span>
                            </label>
                          ))}
                        </div>
                      </div>

                      {/* Additional Feedback */}
                      <div>
                        <h4 className="font-semibold text-gray-900 mb-4">Additional Feedback (Optional)</h4>
                        <textarea
                          value={richAnnotation.additional_feedback}
                          onChange={(e) => setRichAnnotation(prev => ({ ...prev, additional_feedback: e.target.value }))}
                          placeholder="Any additional thoughts or specific feedback..."
                          className="w-full h-24 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-humain-400 focus:border-transparent resize-none text-sm"
                        />
                      </div>
                    </div>
                  </div>

                  {/* Submit Button */}
                  <div className="flex justify-between items-center mt-8 pt-6 border-t">
                    <div className="text-sm text-gray-600">
                      This rich data will train a more accurate vote predictor model
                    </div>
                    
                    <div className="flex space-x-3">
                      <button
                        onClick={() => {
                          setSelectedChoice('')
                          setShowRichAnnotation(false)
                        }}
                        className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors"
                        disabled={isSubmitting}
                      >
                        Back to Choice
                      </button>
                      
                      <button
                        onClick={submitAnnotation}
                        disabled={isSubmitting}
                        className="humain-btn-primary disabled:opacity-50 disabled:cursor-not-allowed px-6"
                      >
                        {isSubmitting ? 'Saving Rich Data...' : 'Submit Detailed Annotation'}
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="humain-card">
            <div className="humain-card-content text-center py-12">
              {annotations.length === 0 ? (
                <>
                  <StarIcon className="h-16 w-16 text-humain-400 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">No Rich Annotations Yet</h3>
                  <p className="text-gray-600 mb-4">
                    Start by creating custom prompts and providing detailed quality feedback. This will significantly improve vote predictor training.
                  </p>
                  <button
                    onClick={() => setShowPromptInput(true)}
                    className="humain-btn-primary"
                  >
                    Create First Rich Annotation
                  </button>
                </>
              ) : (
                <>
                  <CheckIcon className="h-16 w-16 text-green-500 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">All Detailed Annotations Complete!</h3>
                  <p className="text-gray-600 mb-4">
                    Excellent work! Your detailed feedback will significantly improve the vote predictor model training.
                  </p>
                  <button
                    onClick={() => setShowPromptInput(true)}
                    className="humain-btn-primary"
                  >
                    Create More Rich Annotations
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
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Rich Annotations</h3>
              
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
                      {annotation.rlhf_generated ? (
                        <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs font-medium rounded-full">
                          Auto-Generated
                        </span>
                      ) : (
                        <span className="px-2 py-1 bg-green-100 text-green-800 text-xs font-medium rounded-full">
                          Custom
                        </span>
                      )}
                    </div>
                    
                    <div className="flex items-center space-x-3">
                      {annotation.human_choice ? (
                        <div className="flex items-center space-x-2">
                          <StarIcon className="h-4 w-4 text-yellow-500" />
                          <span className="text-sm font-medium text-green-600">
                            Rich Data: {annotation.human_choice}
                          </span>
                        </div>
                      ) : (
                        <span className="text-sm text-yellow-600">Pending</span>
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