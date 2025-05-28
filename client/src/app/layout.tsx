import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' }); // Add variable for Tailwind

export const metadata: Metadata = {
  title: 'Facial Emotion Detector',
  description: 'Live facial emotion detection using AI',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} font-sans`}>
      {/* Use variable */}
      <body>{children}</body>
    </html>
  );
}
