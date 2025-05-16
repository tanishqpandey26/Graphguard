import type { Metadata } from 'next'
import './globals.css'
import localFont from 'next/font/local'
import {
  ClerkProvider
} from '@clerk/nextjs'

export const metadata: Metadata = {
  title: 'GraphGuard',
  description: 'Fraud Detection',
}


const ageo = localFont({
  src: [
    {
      path: '../../public/font/AgeoPersonalUse.otf',
      weight: '400',
      style: 'normal',
    },
    {
      path: '../../public/font/AgeoPersonalUse-Bold.otf',
      weight: '700',
      style: 'bold',
    },
    {
      path: '../../public/font/AgeoPersonalUse-ExtraBold.otf',
      weight: '850',
      style: 'extra-bold',
    },
    {
      path: '../../public/font/AgeoPersonalUse-Medium.otf',
      weight: '500',
      style: 'medium',
    },
  ],
})


export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (   
    <ClerkProvider>
    <html lang="en">
      <body className={ageo.className} suppressHydrationWarning>{children}</body>
    </html>
    </ClerkProvider>
  )
}
