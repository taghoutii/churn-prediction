import os
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """You are helping a telecom retention team understand why a machine
learning model flagged a specific customer as likely to churn (cancel their
subscription) in the next 90 days.

You will be given a table of the customer's top contributing factors, each with:
- feature: the name of the factor (may be a technical column name)
- value: the customer's actual value for that factor
- shap_value: how much that factor pushed the prediction up (positive = toward
  churn) or down (negative = away from churn), and by how much

Write a short (4-6 sentence) plain-language explanation for a non-technical
retention team member. Requirements:
- No jargon: do not say "SHAP", "feature", "model", or reference technical column
  names directly -- translate them into plain business language
  (e.g. "recharge_amount_mean" -> "how much the customer typically recharges")
- Focus on the 3-4 most impactful factors only, not every row
- End with one concrete, actionable suggestion for the retention team
- Be factual and grounded only in the data provided -- do not invent reasons not
  present in the table
- Do not state or imply the factors CAUSE churn -- these are patterns associated
  with churn, not proven causes"""


def format_shap_table_for_prompt(shap_df: pd.DataFrame, top_n: int = 6) -> str:
    top = shap_df.head(top_n)
    lines = [f"- {row.feature}: value={row.value}, impact={row.shap_value:+.3f}" for row in top.itertuples()]
    return "\n".join(lines)


def generate_explanation(shap_df: pd.DataFrame, predicted_proba: float) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Set it before running this function (never hardcode API keys in code)."
        )

    client = Anthropic(api_key=api_key)
    table_str = format_shap_table_for_prompt(shap_df)

    message = client.messages.create(
        model=MODEL,
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"This customer has a predicted churn probability of {predicted_proba:.1%} "
                f"in the next 90 days. Here are the top contributing factors:\n\n{table_str}"
            ),
        }],
    )
    return message.content[0].text