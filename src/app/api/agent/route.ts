import { NextRequest, NextResponse } from 'next/server'
import Anthropic from '@anthropic-ai/sdk'

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
})

export async function POST(request: NextRequest) {
  try {
    const { selectedText, surroundingText, documentTitle } = await request.json()

    const prompt = `You are a reading assistant AI. A user is reading "${documentTitle}" and has selected the following text: "${selectedText}".

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
}`

    const message = await anthropic.messages.create({
      model: "claude-3-5-sonnet-20241022",
      max_tokens: 1000,
      temperature: 0.7,
      messages: [{ role: "user", content: prompt }],
    })

    const content = message.content[0]
    if (!content || content.type !== 'text') {
      throw new Error('No text response from AI')
    }

    const response = JSON.parse(content.text)
    return NextResponse.json(response)
  } catch (error) {
    console.error('Agent API error:', error)
    return NextResponse.json(
      { error: 'Failed to process request' },
      { status: 500 }
    )
  }
}

