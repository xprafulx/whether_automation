from __future__ import annotations

import os
import sqlite3
import random
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
                f"{float(row['temperature']):.1f}°C, "
                f"wind {float(row['wind']):.1f} m/s, "
                f"rain {float(row['precipitation']):.1f} mm\n"
            )
    return text

# -------------------------------
# Generate poem using Groq (ROULETTE)
# -------------------------------
def generate_poem(weather_text: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY environment variable")

    client = Groq(api_key=api_key)

    # The Persona Roulette!
    personas = [
        "a mystical, human-centered poet, deeply inspired by Rumi",
        "a cynical but poetic cyberpunk hacker who loves rain and neon lights",
        "a dramatic, melancholy Shakespearean actor walking the moors",
        "an overly enthusiastic, joyful golden retriever who just loves being outside"
    ]
    chosen_persona = random.choice(personas)
    print(f"🎭 Today's AI Persona: {chosen_persona}")

    prompt = f"""
    You are {chosen_persona}. 
    Look at this weather data for three cities:
    {weather_text}

    CRITICAL RULE: YOU MUST NOT USE ANY NUMBERS OR RAW DATA IN THE POEM. 
    Do not write "11 degrees" or "8.9 m/s" or "5.3 mm". 
    Instead, translate the numbers into sensory descriptions (e.g., "a biting frost", "a howling gale", "a gentle drizzle").

    Your task is to write TWO COMPLETELY DIFFERENT poems. DO NOT translate one into the other. They must be unique creations born from the same weather data.

    1. THE ENGLISH POEM:
    Write a flowing, atmospheric poem comparing the three locations. Describe the differences in temperature, wind, rain, and clouds. Stay entirely in character as your persona. Suggest where you would most like to be tomorrow based on your persona's preferences.

    2. THE NEPALI POEM:
    Write an entirely original, independent poem in Nepali. DO NOT translate the English poem. 
    - Use an authentic Nepali poetic rhythm. 
    - Use rich cultural idioms and words that evoke the true feeling of the weather. 
    - Make it deeply moving, rhythmic, and maintain the underlying emotion of your persona.
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
    period = str(row['period']).lower()

    is_night = (period == "night")

    if rain > 0:
        icon = "🌧️"
    elif clouds > 60:
        icon = "☁️"
    elif clouds > 20:
        icon = "☁️" if is_night else "⛅"
    else:
        icon = "🌙" if is_night else "☀️"
        
    if temp <= 5:
        color_class = "temp-cold" 
    elif temp >= 20:
        color_class = "temp-hot"  
    else:
        color_class = "temp-mild" 

    wind_style = "font-weight: bold; color: #7f8c8d;" if wind > 8 else ""

    return icon, color_class, wind_style

# -------------------------------
# Generate HTML page
# -------------------------------
def create_html(poem: str, df: pd.DataFrame) -> None:
    os.makedirs("docs", exist_ok=True)

    # 1. Determine overall atmospheric theme based on average weather
    avg_rain = df['precipitation'].astype(float).mean()
    avg_clouds = df['cloud_cover'].astype(float).mean()

    if avg_rain > 0.5:
        theme_class = "theme-rainy"
    elif avg_clouds > 50:
        theme_class = "theme-cloudy"
    else:
        theme_class = "theme-sunny"

    # 2. Build the Weather Cards dynamically
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
                        <span class="wind" style="{wind_style}">🌬️ {float(row['wind']):.1f} m/s</span>
                        <span class="rain">💧 {float(row['precipitation']):.1f} mm</span>
                    </span>
                </div>
            """
        
        cards_html += """
            </div>
        </div>
        """

    # 3. Inject into the main HTML template
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
        color: #2c3e50;
        transition: background 0.8s ease;
        min-height: 100vh;
    }}
    
    body.theme-sunny {{ background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%); }}
    body.theme-cloudy {{ background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%); }}
    body.theme-rainy {{ background: linear-gradient(135deg, #bdc3c7 0%, #2c3e50 100%); }}
    body.theme-rainy h1, body.theme-rainy h2, body.theme-rainy .footer {{ color: #ffffff; text-shadow: 0 2px 4px rgba(0,0,0,0.3); }}

    h1, h2, h3 {{ font-family: 'Lora', serif; }}
    h1 {{ text-align: center; margin-bottom: 40px; font-size: 2.5em; }}
    
    .cards-container {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin-bottom: 40px;
    }}
    
    @keyframes fadeUp {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .card {{
        background: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.4);
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        overflow: hidden;
        opacity: 0;
        animation: fadeUp 0.6s ease-out forwards;
    }}
    .card:nth-child(1) {{ animation-delay: 0.1s; }}
    .card:nth-child(2) {{ animation-delay: 0.2s; }}
    .card:nth-child(3) {{ animation-delay: 0.3s; }}

    .card h3 {{
        background: rgba(44, 62, 80, 0.9);
        color: white;
        margin: 0;
        padding: 15px 20px;
        font-size: 1.2em;
    }}
    .card-content {{ padding: 20px; }}
    .period-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid rgba(0,0,0,0.05);
    }}
    .period-row:last-child {{ border-bottom: none; }}
    .period-name {{ font-weight: 600; color: #5a6c7d; width: 80px; }}
    .weather-data {{ display: flex; gap: 15px; font-size: 0.9em; }}
    
    .temp-cold {{ color: #2980b9; font-weight: bold; }}
    .temp-hot {{ color: #d35400; font-weight: bold; }}
    .temp-mild {{ color: #27ae60; font-weight: bold; }}

    /* UPGRADED POEM BOX */
    .poem-box {{
        position: relative;
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        padding: 50px 40px;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        font-family: 'Lora', serif;
        font-size: 1.15em;
        line-height: 1.8;
        white-space: pre-wrap;
        color: #2c3e50;
        margin-bottom: 30px;
        opacity: 0;
        animation: fadeUp 0.8s ease-out forwards;
        animation-delay: 0.5s;
        overflow: hidden;
    }}

    /* The Editorial Watermark Quote */
    .poem-box::before {{
        content: "“";
        position: absolute;
        top: -40px;
        left: 20px;
        font-size: 12em;
        color: rgba(44, 62, 80, 0.05);
        font-family: 'Lora', serif;
        line-height: 1;
        pointer-events: none;
    }}

    /* The Blinking Cursor */
    .cursor {{
        display: inline-block;
        width: 3px;
        height: 1.1em;
        background-color: #2c3e50;
        vertical-align: text-bottom;
        margin-left: 4px;
        animation: blink 1s step-end infinite;
    }}
    @keyframes blink {{ 50% {{ opacity: 0; }} }}
    
    .footer {{
        text-align: center;
        font-size: 0.85em;
        margin-top: 40px;
        padding-bottom: 20px;
    }}
    </style>
</head>
<body class="{theme_class}">

    <h1>🌤️ Tomorrow's Skies</h1>

    <div class="cards-container">
        {cards_html}
    </div>

    <h2>📜 The Elements, Translated</h2>
    
    <pre id="raw-poem" style="display: none;">{poem}</pre>
    
    <div class="poem-box">
        <span id="typed-output"></span><span class="cursor"></span>
    </div>

    <div class="footer">
        <p>Pipeline updated automatically via GitHub Actions • Last run: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <script>
        document.addEventListener("DOMContentLoaded", function() {{
            const rawText = document.getElementById('raw-poem').textContent;
            const output = document.getElementById('typed-output');
            let index = 0;
            const typingSpeed = 15; // Lower is faster

            function typeWriter() {{
                if (index < rawText.length) {{
                    output.textContent += rawText.charAt(index);
                    index++;
                    setTimeout(typeWriter, typingSpeed);
                }}
            }}

            // Wait 1.2 seconds for the frosted glass box to finish fading up, then start typing!
            setTimeout(typeWriter, 1200);
        }});
    </script>
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
