'use client'

import { useState, useEffect } from 'react'
import { X, Brain, BookOpen, Lightbulb, HelpCircle } from 'lucide-react'
import { AgentContext, AgentResponse } from '@/types'

interface AgentAssistantProps {
  selectedText: string
  context: AgentContext
  onClose: () => void
}

export default function AgentAssistant({ selectedText, context, onClose }: AgentAssistantProps) {
  const [response, setResponse] = useState<AgentResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (selectedText) {
      handleAgentRequest()
    }
  }, [selectedText])

  const handleAgentRequest = async () => {
    setIsLoading(true)
    setError(null)
    
    try {
      const response = await fetch('/api/agent', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          selectedText: context.selectedText,
          surroundingText: context.surroundingText,
          documentTitle: context.documentTitle,
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to get AI assistance')
      }

      const data = await response.json()
      setResponse(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
      // For now, provide a mock response for demonstration
      setResponse({
        explanation: `This text appears to discuss "${selectedText}". This could be a complex concept that might benefit from additional context or clarification.`,
        definitions: [
          `${selectedText}: A key concept in this context that may require deeper understanding.`
        ],
        relatedConcepts: [
          'Background knowledge',
          'Context analysis',
          'Conceptual framework'
        ],
        suggestions: [
          'Consider researching the broader context of this topic',
          'Look for examples or case studies related to this concept',
          'Connect this idea to other concepts in the document'
        ]
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-lg shadow-2xl max-w-2xl w-full max-h-[80vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <div className="flex items-center space-x-2">
            <Brain className="w-6 h-6 text-accent" />
            <h2 className="text-xl font-semibold">AI Reading Assistant</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Selected Text */}
        <div className="p-6 bg-gray-50 border-b">
          <p className="text-sm text-gray-600 mb-2">Selected text:</p>
          <p className="font-medium italic">"{selectedText}"</p>
        </div>

        {/* Content */}
        <div className="p-6">
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-accent"></div>
              <span className="ml-3 text-gray-600">Analyzing text...</span>
            </div>
          ) : error ? (
            <div className="text-center py-8">
              <HelpCircle className="w-12 h-12 mx-auto text-gray-400 mb-4" />
              <p className="text-red-600 mb-4">{error}</p>
              <button
                onClick={handleAgentRequest}
                className="px-4 py-2 bg-accent text-white rounded-lg hover:bg-blue-600 transition-colors"
              >
                Try Again
              </button>
            </div>
          ) : response ? (
            <div className="space-y-6">
              {/* Explanation */}
              <div>
                <div className="flex items-center space-x-2 mb-3">
                  <BookOpen className="w-5 h-5 text-blue-600" />
                  <h3 className="font-semibold">Explanation</h3>
                </div>
                <p className="text-gray-700 leading-relaxed">{response.explanation}</p>
              </div>

              {/* Definitions */}
              {response.definitions && response.definitions.length > 0 && (
                <div>
                  <div className="flex items-center space-x-2 mb-3">
                    <HelpCircle className="w-5 h-5 text-green-600" />
                    <h3 className="font-semibold">Key Definitions</h3>
                  </div>
                  <ul className="space-y-2">
                    {response.definitions.map((definition, index) => (
                      <li key={index} className="text-gray-700 bg-green-50 p-3 rounded-lg">
                        {definition}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Related Concepts */}
              {response.relatedConcepts && response.relatedConcepts.length > 0 && (
                <div>
                  <div className="flex items-center space-x-2 mb-3">
                    <Lightbulb className="w-5 h-5 text-yellow-600" />
                    <h3 className="font-semibold">Related Concepts</h3>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {response.relatedConcepts.map((concept, index) => (
                      <span
                        key={index}
                        className="px-3 py-1 bg-yellow-100 text-yellow-800 rounded-full text-sm"
                      >
                        {concept}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Suggestions */}
              {response.suggestions && response.suggestions.length > 0 && (
                <div>
                  <div className="flex items-center space-x-2 mb-3">
                    <Brain className="w-5 h-5 text-purple-600" />
                    <h3 className="font-semibold">Reading Suggestions</h3>
                  </div>
                  <ul className="space-y-2">
                    {response.suggestions.map((suggestion, index) => (
                      <li key={index} className="flex items-start space-x-2">
                        <span className="text-purple-600 mt-1">•</span>
                        <span className="text-gray-700">{suggestion}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ) : null}
        </div>

        {/* Footer */}
        <div className="p-6 border-t bg-gray-50 flex justify-end space-x-3">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-600 hover:text-gray-800 transition-colors"
          >
            Close
          </button>
          <button
            onClick={handleAgentRequest}
            className="px-4 py-2 bg-accent text-white rounded-lg hover:bg-blue-600 transition-colors"
          >
            Ask Again
          </button>
        </div>
      </div>
    </div>
  )
}