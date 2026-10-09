import { NextResponse } from 'next/server';
import experimentData from '@/data/experimentData.json';
import sampleTransactions from '@/data/sampleTransactions.json';

export async function GET() {
  return NextResponse.json({
    success: true,
    data: experimentData,
    samples: sampleTransactions,
  });
}
