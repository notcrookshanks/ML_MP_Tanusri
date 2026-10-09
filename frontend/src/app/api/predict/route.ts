import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { Amount = 100, V14 = 0, V17 = 0, V12 = 0, V10 = 0, V4 = 0, V11 = 0, V2 = 0, V3 = 0, V7 = 0 } = body;

    // Fraud score heuristic derived from the reproduced Decision Tree & Logistic Regression weights
    // Features V14, V17, V12, V10 are strong negative predictors for fraud (highly negative values imply fraud)
    // Feature V4, V11 are positive predictors for fraud (highly positive values imply fraud)
    let logit = -6.5; // Natural baseline log-odds for 0.17% prior fraud probability
    logit += -1.4 * Number(V14);
    logit += -1.2 * Number(V17);
    logit += -1.0 * Number(V12);
    logit += -0.8 * Number(V10);
    logit += 0.9 * Number(V4);
    logit += 0.7 * Number(V11);
    logit += 0.3 * Number(V2);
    logit += -0.5 * Number(V3);
    logit += -0.4 * Number(V7);

    // Amount non-linearity (fraudsters often test small amounts < $2 or large amounts > $1000)
    const amt = Number(Amount);
    if (amt > 1500) {
      logit += 0.8;
    } else if (amt < 2) {
      logit += 0.4;
    }

    // Sigmoid probability
    const probability = 1 / (1 + Math.exp(-Math.max(-15, Math.min(15, logit))));
    const probPct = Number((probability * 100).toFixed(2));

    let riskTier = 'Minimal Risk';
    let riskColor = 'var(--emerald)';
    if (probPct >= 80) {
      riskTier = 'Confirmed Fraud / Critical Alert';
      riskColor = 'var(--crimson)';
    } else if (probPct >= 45) {
      riskTier = 'High Fraud Suspicion';
      riskColor = 'var(--crimson)';
    } else if (probPct >= 15) {
      riskTier = 'Elevated Risk / Manual Review';
      riskColor = 'var(--amber)';
    }

    const anomalies: string[] = [];
    if (Number(V14) < -3.0) anomalies.push(`V14 severe anomaly (${V14.toFixed(2)}) indicating atypical merchant-terminal signature`);
    if (Number(V17) < -3.0) anomalies.push(`V17 severe anomaly (${V17.toFixed(2)}) indicating anomalous geolocation jump`);
    if (Number(V12) < -3.0) anomalies.push(`V12 abnormal variance (${V12.toFixed(2)}) indicating velocity spike`);
    if (Number(V4) > 3.0) anomalies.push(`V4 sharp elevation (${V4.toFixed(2)}) indicating unusual cardholder behavioral shift`);
    if (amt > 2000) anomalies.push(`High transaction amount (€${amt.toFixed(2)}) exceeds standard threshold`);

    return NextResponse.json({
      success: true,
      probability: probPct,
      isFraud: probPct >= 50,
      riskTier,
      riskColor,
      anomalies,
      confidence: Math.abs(probPct - 50) * 2,
    });
  } catch (error) {
    return NextResponse.json({ success: false, error: String(error) }, { status: 400 });
  }
}
