#!/usr/bin/env python3

import sys
import os
import time
import logging

# Add scripts directory to path
sys.path.append("scripts")

from scrape_mnp_data import MNPScraper
from scrape_mnp_selenium import MNPSeleniumScraper

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def debug_hanging_game():
    """Debug the specific game causing hanging at Week 1, Match 6, Round 4, Game 2"""

    # Get the match URL for Week 1, Match 6
    basic_scraper = MNPScraper()
    all_matches = basic_scraper.get_match_links()

    # Week 1 should be first 85 matches (945 total / 14 weeks ≈ 67-68 per week, but let's use 85)
    week1_matches = all_matches[:85]
    match6_url = week1_matches[5]  # Match 6 (0-indexed)

    logger.info(f"Week 1, Match 6 URL: {match6_url}")

    # Get games from this match
    games = basic_scraper.get_game_links_from_match(match6_url)
    logger.info(f"Found {len(games)} games in Match 6")

    # Calculate Round 4, Game 2
    games_per_round = max(1, len(games) // 4)
    round4_start = 3 * games_per_round  # Round 4 (0-indexed)
    round4_games = games[round4_start : round4_start + games_per_round]

    logger.info(f"Round 4 has {len(round4_games)} games")
    for i, game_url in enumerate(round4_games):
        logger.info(f"Round 4, Game {i+1}: {game_url}")

    # Focus on Game 2 which should be causing the hang
    if len(round4_games) >= 2:
        problematic_game = round4_games[1]  # Game 2 (0-indexed)
        logger.info(f"*** TESTING PROBLEMATIC GAME: {problematic_game}")

        # Test this specific game with detailed monitoring
        selenium_scraper = MNPSeleniumScraper()
        try:
            logger.info(f"Starting detailed test of hanging game...")
            start_time = time.time()

            # Add extra monitoring
            logger.info(f"Attempting to load page...")
            game_data = selenium_scraper.scrape_game_with_selenium(problematic_game)

            end_time = time.time()
            duration = end_time - start_time

            logger.info(f"Game scrape completed in {duration:.2f} seconds")
            if game_data:
                logger.info(f"SUCCESS: Found {len(game_data.get('images', []))} images")
                logger.info(
                    f"SUCCESS: Found scores: {game_data.get('js_data', {}).get('scores', 'None')}"
                )
            else:
                logger.warning("ISSUE: No game data returned!")

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            logger.error(f"FAILURE: Game scraping failed after {duration:.2f}s: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
        finally:
            try:
                if hasattr(selenium_scraper, "driver") and selenium_scraper.driver:
                    selenium_scraper.driver.quit()
            except:
                pass
    else:
        logger.error("Not enough games in Round 4 to test Game 2")

    logger.info("Debug complete")


if __name__ == "__main__":
    debug_hanging_game()
