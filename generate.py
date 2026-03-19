from __future__ import annotations

import os
import sqlite3
import pandas as pd
from groq import Groq


# -------------------------------
# Load data from SQLite
# -------------------------------
def load_weather() -> pd.DataFrame:
    conn = sqlite3.connect("data/weather.db")
    df = pd.read_sql("SELECT * FROM weather", conn)
    conn.close()
    return df


# -------------------------------
# Format data for LLM
# -------------------------------
def format_weather_for_prompt(df: pd.DataFrame) -> str:
    text = ""

    for location in df["location"].unique():
        text += f"\nLocation: {location}\n"
        loc_df = df[df["location"] == location]

        for _, row in loc_df.iterrows():
            text += (
                f"{row['period']}: "
                f"{row['temperature']}°C, "
                f"wind {row['wind']} m/s, "
                f"rain {row['precipitation']} mm\n"
            )

    return text


# -------------------------------
# Generate poem using Groq
# -------------------------------

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("Missing GROQ_API_KEY environment variable")


def generate_poem(weather_text: str) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""
    You are a mystical, human-centered poet, deeply inspired by Rumi. 
    Look at this weather data for three cities:
    {weather_text}

    Your task is to write TWO COMPLETELY DIFFERENT poems. DO NOT translate one into the other. They must be unique creations born from the same weather data.

    1. THE ENGLISH POEM:
    Write a flowing, atmospheric poem comparing the three locations. Describe the differences in temperature, wind, rain, and clouds. Connect the weather to human emotions, inner reflection, and humanity. Suggest where the soul would feel most at peace tomorrow. Use philosophical and mystical imagery.

    2. THE NEPALI POEM:
    Write an entirely original, independent poem in Nepali. DO NOT translate the English poem. 
    - Use an authentic Nepali poetic rhythm (like a Muktak or classic Chhanda). 
    - Use rich cultural idioms and words that evoke the true, raw feeling of the weather in the Himalayas versus the flatlands of Europe. 
    - Make it deeply moving, rhythmic, and rooted in Eastern philosophical contemplation.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# -------------------------------
# Generate HTML page
# -------------------------------
def create_html(poem: str, df: pd.DataFrame) -> None:
    os.makedirs("docs", exist_ok=True)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Weather Poem</title>

    <style>
    body {{
        font-family: Arial;
        max-width: 800px;
        margin: auto;
        padding: 20px;
        background: #f5f7fa;
    }}
    h1 {{
        color: #4a6fa5;
    }}
    pre {{
        background: white;
        padding: 15px;
        border-radius: 10px;
        white-space: pre-wrap;
    }}
    </style>

</head>
<body>

    <h1>🌦️ Tomorrow's Weather</h1>

    <h2>📊 Weather Data</h2>
    <pre>{df.to_string(index=False)}</pre>

    <h2>📝 Poem</h2>
    <pre>{poem}</pre>

    <h3>⏱️ Last updated</h3>
    <p>{pd.Timestamp.now()}</p>

</body>
</html>
"""

    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(html)


# -------------------------------
# Main
# -------------------------------
def main():
    df = load_weather()
    weather_text = format_weather_for_prompt(df)

    poem = generate_poem(weather_text)

    create_html(poem, df)

    print("✅ Poem generated and HTML created!")


if __name__ == "__main__":
    main()
