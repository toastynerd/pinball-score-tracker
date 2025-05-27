#!/usr/bin/env python3

import logging
import os
import sys
import time

# Add scripts directory to path
sys.path.append("scripts")

from scrape_mnp_data import MNPScraper
from scrape_mnp_selenium import MNPSeleniumScraper

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def debug_specific_game():
    """Debug the specific game that's causing hanging"""

    # First get the game list for the match we were processing
    basic_scraper = MNPScraper()
    match_url = "https://www.mondaynightpinball.com/matches/mnp-21-1-ADB-JMF"

    logger.info(f"Getting games from match: {match_url}")
    games = basic_scraper.get_game_links_from_match(match_url)
    logger.info(f"Found {len(games)} games total")

    # Print all games with their round/game numbers
    games_per_round = max(1, len(games) // 4)

    for round_num in range(1, 5):
        round_start = (round_num - 1) * games_per_round
        round_end = round_start + games_per_round if round_num < 4 else len(games)
        round_games = games[round_start:round_end]

        logger.info(f"ROUND {round_num}: {len(round_games)} games")
        for game_idx, game_url in enumerate(round_games):
            logger.info(f"  Round {round_num}, Game {game_idx + 1}: {game_url}")

            # Focus on Round 4, Game 2 which should be where it hung
            if round_num == 4 and game_idx == 1:  # Game 2 (0-indexed)
                logger.info(
                    f"*** This is the game that likely caused hanging: {game_url}"
                )

                # Test this specific game
                selenium_scraper = MNPSeleniumScraper()
                try:
                    logger.info(f"Testing problematic game...")
                    start_time = time.time()

                    game_data = selenium_scraper.scrape_game_with_selenium(game_url)

                    end_time = time.time()
                    duration = end_time - start_time

                    logger.info(f"Game scrape completed in {duration:.2f} seconds")
                    if game_data:
                        logger.info(f"Found {len(game_data.get('images', []))} images")
                        logger.info(
                            f"Found scores: {game_data.get('js_data', {}).get('scores', 'None')}"
                        )
                    else:
                        logger.warning("No game data returned!")

                except Exception as e:
                    logger.error(f"Game scraping failed: {e}")
                    import traceback

                    logger.error(f"Traceback: {traceback.format_exc()}")
                finally:
                    if hasattr(selenium_scraper, "driver") and selenium_scraper.driver:
                        selenium_scraper.driver.quit()


if __name__ == "__main__":
    debug_specific_game()
