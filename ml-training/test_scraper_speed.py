#!/usr/bin/env python3

import sys
import os
import time
import logging

# Add scripts directory to path
sys.path.append("scripts")

from scrape_mnp_selenium import MNPSeleniumScraper

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_single_game():
    """Test scraping a single game to see performance"""
    scraper = MNPSeleniumScraper()

    # Test game URL from match 1
    test_url = "https://www.mondaynightpinball.com/games/mnp-21-1-ADB-JMF.1.1"

    logger.info(f"Starting scrape test at {time.strftime('%H:%M:%S')}")
    start_time = time.time()

    try:
        game_data = scraper.scrape_game_with_selenium(test_url)
        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"Scrape completed in {duration:.2f} seconds")
        logger.info(f"Found {len(game_data.get('images', []))} images")
        logger.info(
            f"Found scores: {game_data.get('js_data', {}).get('scores', 'None')}"
        )

        # Test image download
        if game_data.get("images"):
            img_url = game_data["images"][0]
            logger.info(f"Testing image download from: {img_url[:50]}...")

            download_start = time.time()
            local_path = scraper.download_image(
                img_url, "test_download", "test_image.png"
            )
            download_end = time.time()
            download_duration = download_end - download_start

            logger.info(f"Image download completed in {download_duration:.2f} seconds")
            logger.info(f"Saved to: {local_path}")

    except Exception as e:
        logger.error(f"Scrape failed: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        scraper.cleanup()


if __name__ == "__main__":
    test_single_game()
