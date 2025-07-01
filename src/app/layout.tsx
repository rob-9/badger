import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Curio - Intelligent Reading',
  description: 'An intelligent reading application with AI-powered assistance',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}