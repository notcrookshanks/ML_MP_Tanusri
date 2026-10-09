import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Credit Card Fraud Detection | IEEE Access (2022) Scientific Reproduction',
  description: 'Auditable, faithful reproduction portal of the IEEE Access 2022 paper by Alarfaj et al. evaluating Machine Learning, Deep Learning, and CNN architectures on the European benchmark dataset.',
  keywords: ['Fraud Detection', 'Machine Learning', 'Deep Learning', 'CNN', 'IEEE Access', 'Reproducibility', 'Credit Card Benchmark'],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <link rel="icon" href="/favicon.ico" />
      </head>
      <body>
        {children}
      </body>
    </html>
  );
}
