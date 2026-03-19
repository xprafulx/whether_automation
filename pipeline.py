from __future__ import annotations

from fetch import main as fetch_main
from generate import main as generate_main


def main() -> None:
    print("🌦️ Starting weather pipeline...")

    # Step 1: Fetch + store weather data
    fetch_main()
    print("✅ Weather data fetched and stored")

    # Step 2: Generate poem + HTML
    generate_main()
    print("✅ Poem generated and HTML updated")

    print("🚀 Pipeline completed successfully!")


if __name__ == "__main__":
    main()