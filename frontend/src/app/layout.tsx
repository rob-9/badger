import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'badger',
  description: 'An intelligent reading application with AI assistance',
  icons: {
    icon: [
      { url: '/icons/favicon.ico' },
      { url: '/icons/favicon-96x96.png', sizes: '96x96', type: 'image/png' },
    ],
    apple: '/icons/apple-touch-icon.png',
  },
  manifest: '/site.webmanifest',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Prevent dark mode flash */}
        <script dangerouslySetInnerHTML={{ __html: `
          try {
            const theme = localStorage.getItem('badger-theme')
            if (theme === 'dark' || (!theme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
              document.documentElement.classList.add('dark')
            }
          } catch(e) {}
        ` }} />
      </head>
      <body className="bg-paper dark:bg-[#141414] text-ink dark:text-[#e0e0e0] transition-colors duration-200">
        {children}
      </body>
    </html>
  )
}
