import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const { selectedText, surroundingText, documentTitle } = await request.json()

    // For demo purposes, return a mock response
    // In production, integrate with OpenAI or another AI service
    const response = {
      explanation: `The selected text "${selectedText}" appears in the context of "${documentTitle}". This concept may benefit from additional explanation or context to improve understanding.`,
      definitions: selectedText.length > 50 ? [] : [
        `${selectedText}: A key term that requires deeper understanding in this context.`
      ],
      relatedConcepts: [
        'Contextual analysis',
        'Reading comprehension',
        'Knowledge synthesis'
      ],
      suggestions: [
        'Consider the broader context surrounding this passage',
        'Look for connections to other parts of the document',
        'Research related topics for deeper understanding'
      ]
    }

    return NextResponse.json(response)
  } catch (error) {
    console.error('Agent API error:', error)
    return NextResponse.json(
      { error: 'Failed to process request' },
      { status: 500 }
    )
  }
}

// Uncomment and configure when ready to integrate with OpenAI
/*
import OpenAI from 'openai'

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
})

export async function POST(request: NextRequest) {
  try {
    const { selectedText, surroundingText, documentTitle } = await request.json()

    const prompt = `
You are a reading assistant AI. A user is reading "${documentTitle}" and has selected the following text: "${selectedText}".

Here is the surrounding context:
"${surroundingText}"

Please provide:
1. A clear explanation of the selected text
2. Any key definitions if the text contains technical terms
3. Related concepts that might help understanding
4. Suggestions for deeper comprehension

Respond in JSON format with the following structure:
{
  "explanation": "string",
  "definitions": ["string array"],
  "relatedConcepts": ["string array"],
  "suggestions": ["string array"]
}
`

    const completion = await openai.chat.completions.create({
      model: "gpt-4",
      messages: [{ role: "user", content: prompt }],
      temperature: 0.7,
      max_tokens: 1000,
    })

    const content = completion.choices[0]?.message?.content
    if (!content) {
      throw new Error('No response from AI')
    }

    const response = JSON.parse(content)
    return NextResponse.json(response)
  } catch (error) {
    console.error('Agent API error:', error)
    return NextResponse.json(
      { error: 'Failed to process request' },
      { status: 500 }
    )
  }
}
*/