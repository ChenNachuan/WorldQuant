# Forum Post: Intraday Volatility Convergence
Author: QuantExpert_99

One of my most successful alphas uses the logic that when the intraday range (high-low) is extremely high compared to the daily close price, it often signals a liquidity exhaustion. 
However, this signal is only valid if it's accompanied by a significant spike in volume (adv20) and a negative recent return (mean reversion).

Key logic:
1. Calculate intraday range: (high - low) / close
2. Filter for high volume: volume / adv20 > 2.0
3. Check for oversold: returns < 0

Expression hint: rank((high-low)/close) * rank(volume/adv20) * sign(-returns)
Neutralization: Industry is best for this.
Decay: 4-6 days.
