import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Email to Tasks',
  description: 'Gmail to AI-powered Task Board',
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