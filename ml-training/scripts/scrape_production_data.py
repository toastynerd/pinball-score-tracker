#!/usr/bin/env python3
"""
Production data scraping script with conservative request limits
Designed to be respectful to the MNP server while collecting quality training data
"""

import time
import json
import os
import logging
from datetime import datetime
from pathlib import Path

# Import our existing scrapers
from scrape_mnp_data import MNPScraper
from scrape_mnp_selenium import MNPSeleniumScraper
from filter_valid_images import filter_dataset
from format_paddleocr_data import PaddleOCRFormatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionDataCollector:
    def __init__(self, max_matches=3, delay_between_requests=5, max_games_per_match=10):
        """
        Conservative scraper for production data collection

        Args:
            max_matches: Maximum number of matches to scrape (default: 3)
            delay_between_requests: Seconds to wait between requests (default: 5)
            max_games_per_match: Maximum games to scrape per match (default: 10)
        """
        self.max_matches = max_matches
        self.delay_between_requests = delay_between_requests
        self.max_games_per_match = max_games_per_match
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"data/production_dataset_{self.session_id}"

        # Setup directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/images", exist_ok=True)

        logger.info(f"Production data collection session: {self.session_id}")
        logger.info(
            f"Conservative settings: {max_matches} matches, {delay_between_requests}s delay"
        )

    def collect_with_rate_limiting(self):
        """Collect data with conservative rate limiting"""
        basic_scraper = MNPScraper()
        selenium_scraper = MNPSeleniumScraper()
        all_data = []

        try:
            logger.info("Getting match links...")
            # Use basic scraper to get match links
            match_links = basic_scraper.get_match_links()[: self.max_matches]
            logger.info(f"Will process {len(match_links)} matches")

            for match_idx, match_url in enumerate(match_links):
                logger.info(f"Processing match {match_idx + 1}/{len(match_links)}: {match_url}")

                # Get game links for this match
                game_links = basic_scraper.get_game_links_from_match(match_url)[
                    : self.max_games_per_match
                ]
                logger.info(f"Found {len(game_links)} games for this match")

                match_data = []
                for game_idx, game_url in enumerate(game_links):
                    logger.info(f"  Processing game {game_idx + 1}/{len(game_links)}")

                    # Scrape game data with Selenium
                    game_data = selenium_scraper.scrape_game_with_selenium(game_url)

                    if game_data and game_data.get("js_data", {}).get("scores"):
                        # Only keep games with actual score data
                        valid_images = []
                        for img_idx, img_url in enumerate(game_data.get("images", [])):
                            if selenium_scraper.is_game_image(img_url):
                                # Download image
                                filename = f"match_{match_idx + 1}_game_{game_idx + 1}_img_{img_idx + 1}.png"
                                local_path = selenium_scraper.download_image(
                                    img_url, f"{self.output_dir}/images", filename
                                )
                                if local_path:
                                    valid_images.append(local_path)

                        if valid_images:
                            game_data["local_images"] = valid_images
                            match_data.append(game_data)
                            logger.info(
                                f"    ✓ Collected {len(valid_images)} images with scores: {game_data.get('js_data', {}).get('scores', [])}"
                            )
                        else:
                            logger.info(f"    ✗ No valid images found")
                    else:
                        logger.info(f"    ✗ No score data found")

                    # Rate limiting between games
                    if game_idx < len(game_links) - 1:  # Don't delay after last game
                        logger.info(
                            f"    Waiting {self.delay_between_requests}s before next game..."
                        )
                        time.sleep(self.delay_between_requests)

                all_data.extend(match_data)

                # Rate limiting between matches (longer delay)
                if match_idx < len(match_links) - 1:  # Don't delay after last match
                    longer_delay = self.delay_between_requests * 2
                    logger.info(f"Waiting {longer_delay}s before next match...")
                    time.sleep(longer_delay)

        finally:
            selenium_scraper.close()

        # Save raw data
        metadata_file = f"{self.output_dir}/raw_dataset_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(all_data, f, indent=2)

        logger.info(f"Raw data collection complete: {len(all_data)} games collected")
        return all_data, metadata_file

    def process_collected_data(self, raw_metadata_file):
        """Process and filter the collected data"""
        logger.info("Processing collected data...")

        # Filter out blank images
        logger.info("Filtering blank/invalid images...")
        filtered_data = filter_dataset(self.output_dir)

        if not filtered_data:
            logger.warning("No valid data after filtering!")
            return None

        logger.info(f"Filtered data: {len(filtered_data)} valid games")

        # Format for PaddleOCR training
        logger.info("Formatting data for PaddleOCR training...")
        formatter = PaddleOCRFormatter(self.output_dir, f"{self.output_dir}/paddleocr_training")
        num_samples = formatter.process_dataset()

        logger.info(f"Training dataset created with {num_samples} samples")
        return num_samples

    def generate_collection_report(self, num_samples):
        """Generate a report of the data collection session"""
        report = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "settings": {
                "max_matches": self.max_matches,
                "delay_between_requests": self.delay_between_requests,
                "max_games_per_match": self.max_games_per_match,
            },
            "results": {"total_training_samples": num_samples, "output_directory": self.output_dir},
            "files_created": {
                "raw_metadata": f"{self.output_dir}/raw_dataset_metadata.json",
                "filtered_metadata": f"{self.output_dir}/filtered_dataset_metadata.json",
                "training_annotations": f"{self.output_dir}/paddleocr_training/train_list.txt",
                "validation_annotations": f"{self.output_dir}/paddleocr_training/val_list.txt",
                "character_dictionary": f"{self.output_dir}/paddleocr_training/dict.txt",
            },
        }

        report_file = f"{self.output_dir}/collection_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("=" * 60)
        logger.info("DATA COLLECTION REPORT")
        logger.info("=" * 60)
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Training samples collected: {num_samples}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Report saved to: {report_file}")
        logger.info("=" * 60)

        return report

    def run_full_collection(self):
        """Run the complete data collection pipeline"""
        logger.info("Starting production data collection...")

        # Collect raw data
        raw_data, metadata_file = self.collect_with_rate_limiting()

        if not raw_data:
            logger.error("No data collected!")
            return None

        # Process and format data
        num_samples = self.process_collected_data(metadata_file)

        if num_samples is None:
            logger.error("Data processing failed!")
            return None

        # Generate report
        report = self.generate_collection_report(num_samples)

        return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Collect production training data from MNP")
    parser.add_argument(
        "--max-matches",
        type=int,
        default=3,
        help="Maximum number of matches to scrape (default: 3)",
    )
    parser.add_argument(
        "--delay", type=int, default=5, help="Seconds to wait between requests (default: 5)"
    )
    parser.add_argument(
        "--max-games", type=int, default=10, help="Maximum games per match (default: 10)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without actually scraping"
    )

    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN MODE - No actual scraping will be performed")
        logger.info(f"Would scrape {args.max_matches} matches")
        logger.info(f"Would wait {args.delay}s between requests")
        logger.info(f"Would scrape max {args.max_games} games per match")
        total_requests = args.max_matches * args.max_games
        total_time = total_requests * args.delay / 60  # Convert to minutes
        logger.info(f"Estimated total requests: {total_requests}")
        logger.info(f"Estimated total time: {total_time:.1f} minutes")
        return

    # Run collection
    collector = ProductionDataCollector(
        max_matches=args.max_matches,
        delay_between_requests=args.delay,
        max_games_per_match=args.max_games,
    )

    report = collector.run_full_collection()

    if report:
        logger.info("Data collection completed successfully!")
        logger.info(f"Collected {report['results']['total_training_samples']} training samples")
    else:
        logger.error("Data collection failed!")


if __name__ == "__main__":
    main()
