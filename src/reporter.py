from dotenv import load_dotenv
import os
from groq import Groq




def generate_report(sharpe, max_drawdown, win_rate, accuracy, cumulative_strategy, cumulative_market):
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    prompt = f"""
You are a quantitative analyst evaluating an ML-based algorithmic 
trading strategy on Nifty 50 index.

PERFORMANCE METRICS:
Sharpe Ratio: {sharpe:.3f} (above 1.0 is good, negative means strategy lost money risk-adjusted)
Max Drawdown: {max_drawdown:.3f} (how much portfolio fell from peak, closer to 0 is better)
Win Rate: {win_rate:.3f} (percentage of profitable trades, above 0.5 is good)
Model Accuracy: {accuracy:.3f} (above 0.51 is meaningful for stock prediction)
Cumulative Strategy Return: {cumulative_strategy:.3f} (starting capital 1.0, ending value)
Cumulative Market Return: {cumulative_market:.3f} (blind buy-and-hold benchmark)
Test period: Aug 2024 to Jul 2026, Nifty 50 daily data.

Provide a structured performance report covering:
1. Metrics interpretation
2. Strategy vs benchmark comparison
3. Key risks and limitations
4. Conclusion

Professional tone, 200-250 words, suitable for academic project report.
"""
    
    # api call

    client = Groq(
    api_key=api_key,
        )

    chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.1-8b-instant",
        )

    return chat_completion.choices[0].message.content