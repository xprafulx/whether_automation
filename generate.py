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
def generate_poem(weather_text: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY environment variable")

    client = Groq(api_key=api_key)

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
# Helper: Dynamic Styling & Emojis
# -------------------------------
def get_weather_styling(row):
    temp = float(row['temperature'])
    rain = float(row['precipitation'])
    clouds = float(row['cloud_cover'])
    wind = float(row['wind'])

    # Determine Emoji
    if rain > 0:
        icon = "🌧️"
    elif clouds > 60:
        icon = "☁️"
    elif clouds > 20:
        icon = "⛅"
    else:
        icon = "☀️"
        
    # Determine Temperature Color Class
    if temp <= 5:
        color_class = "temp-cold" # Blue
    elif temp >= 20:
        color_class = "temp-hot"  # Orange/Red
    else:
        color_class = "temp-mild" # Neutral

    # Determine Wind Alert
    wind_style = "font-weight: bold; color: #7f8c8d;" if wind > 8 else ""

    return icon, color_class, wind_style


# -------------------------------
# Generate HTML page
# -------------------------------
def create_html(poem: str, df: pd.DataFrame) -> None:
    os.makedirs("docs", exist_ok=True)

    # 1. Build the Weather Cards dynamically in Python
    cards_html = ""
    for location in df["location"].unique():
        loc_df = df[df["location"] == location]
        
        cards_html += f"""
        <div class="card">
            <h3>📍 {location}</h3>
            <div class="card-content">
        """
        
        for _, row in loc_df.iterrows():
            icon, temp_class, wind_style = get_weather_styling(row)
            
            cards_html += f"""
                <div class="period-row">
                    <span class="period-name">{row['period'].capitalize()}</span>
                    <span class="weather-data">
                        <span class="temp {temp_class}">{float(row['temperature']):.1f}°C {icon}</span>
                        <span class="wind" style="{wind_style}">💨 {float(row['wind']):.1f} m/s</span>
                        <span class="rain">💧 {float(row['precipitation']):.1f} mm</span>
                    </span>
                </div>
            """
        
        cards_html += """
            </div>
        </div>
        """

    # 2. Inject into the main HTML template
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Weather Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,700;1,400&family=Nunito:wght@400;600;800&display=swap');

    body {{
        font-family: 'Nunito', sans-serif;
        max-width: 1000px;
        margin: 0 auto;
        padding: 40px 20px;
        background: #f4f1ea; 
        color: #2c3e50;
    }}
    h1, h2, h3 {{
        font-family: 'Lora', serif;
        color: #1a252f;
    }}
    h1 {{ text-align: center; margin-bottom: 40px; font-size: 2.5em; }}
    
    /* Grid layout for the Weather Cards */
    .cards-container {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin-bottom: 40px;
    }}
    .card {{
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        overflow: hidden;
    }}
    .card h3 {{
        background: #2c3e50;
        color: white;
        margin: 0;
        padding: 15px 20px;
        font-size: 1.2em;
    }}
    .card-content {{
        padding: 20px;
    }}
    .period-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid #eee;
    }}
    .period-row:last-child {{ border-bottom: none; }}
    .period-name {{ font-weight: 600; color: #7f8c8d; width: 80px; }}
    .weather-data {{ display: flex; gap: 15px; font-size: 0.9em; }}
    
    /* Conditional Colors */
    .temp-cold {{ color: #2980b9; font-weight: bold; }}
    .temp-hot {{ color: #d35400; font-weight: bold; }}
    .temp-mild {{ color: #27ae60; font-weight: bold; }}

    /* Poem Box */
    .poem-box {{
        background: white;
        padding: 40px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        font-family: 'Lora', serif;
        font-size: 1.15em;
        line-height: 1.8;
        white-space: pre-wrap;
        color: #34495e;
        margin-bottom: 30px;
    }}
    
    .footer {{
        text-align: center;
        color: #7f8c8d;
        font-size: 0.85em;
        margin-top: 40px;
    }}
    </style>
</head>
<body>

    <h1>🌤️ Tomorrow's Skies</h1>

    <div class="cards-container">
        {cards_html}
    </div>

    <h2>📜 The Elements, Translated</h2>
    <div class="poem-box">{poem}</div>

    <div class="footer">
        <p>Pipeline updated automatically via GitHub Actions • Last run: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

</body>
</html>
"""

    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(html)


# -------------------------------
# Main
# -------------------------------
def main():
    print("Loading data...")
    df = load_weather()
    weather_text = format_weather_for_prompt(df)

    print("Generating poem via Groq...")
    poem = generate_poem(weather_text)

    print("Building HTML dashboard...")
    create_html(poem, df)

    print("✅ Dashboard generated in docs/index.html!")


if __name__ == "__main__":
    main()
