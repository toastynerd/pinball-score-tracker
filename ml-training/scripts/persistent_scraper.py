#!/usr/bin/env python3
"""
Persistent scraper for collecting all MNP data over multiple days
Designed to be respectful with very long delays and resumable sessions
"""

import time
import json
import os
import logging
from datetime import datetime, timedelta
import signal
import sys
from pathlib import Path

from scrape_mnp_data import MNPScraper
from scrape_mnp_selenium import MNPSeleniumScraper
from filter_valid_images import filter_dataset
from format_paddleocr_data import PaddleOCRFormatter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("persistent_scraper.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class PersistentScraper:
    def __init__(self, base_delay=60, max_delay=300, session_name=None):
        """
        Persistent scraper that can run for days

        Args:
            base_delay: Base seconds between requests (default: 60s = 1 minute)
            max_delay: Maximum delay between matches (default: 300s = 5 minutes)
            session_name: Name for this scraping session
        """
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.session_name = session_name or f"persistent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Progress tracking
        self.state_file = f"scraper_state_{self.session_name}.json"
        self.output_dir = f"data/persistent_collection_{self.session_name}"

        # Setup directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/images", exist_ok=True)

        # Initialize or load state
        self.state = self.load_state()

        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        logger.info(f"Persistent scraper initialized: {self.session_name}")
        logger.info(f"Base delay: {base_delay}s, Max delay: {max_delay}s")
        logger.info(f"Output directory: {self.output_dir}")

    def load_state(self):
        """Load previous session state if it exists"""
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                state = json.load(f)
                logger.info(
                    f"Resuming session from: {state.get('last_processed_match', 'beginning')}"
                )
                return state
        else:
            logger.info("Starting new scraping session")
            return {
                "session_start": datetime.now().isoformat(),
                "total_matches_processed": 0,
                "total_games_collected": 0,
                "total_images_downloaded": 0,
                "last_processed_match": None,
                "completed_matches": [],
                "failed_matches": [],
                "statistics": {
                    "games_with_data": 0,
                    "games_without_data": 0,
                    "images_collected": 0,
                    "total_requests_made": 0,
                },
            }

    def save_state(self):
        """Save current session state"""
        self.state["last_updated"] = datetime.now().isoformat()
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def signal_handler(self, signum, frame):
        """Handle graceful shutdown"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.save_state()
        self.cleanup()
        logger.info("Shutdown complete. Session state saved.")
        sys.exit(0)

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, "selenium_scraper"):
            self.selenium_scraper.close()

    def calculate_dynamic_delay(self, request_count):
        """Calculate delay that increases throughout the day"""
        # Increase delay as we make more requests
        dynamic_multiplier = 1 + (request_count * 0.01)  # 1% increase per request
        delay = min(self.base_delay * dynamic_multiplier, self.max_delay)
        return int(delay)

    def should_take_long_break(self):
        """Determine if we should take a longer break (every 50 requests)"""
        total_requests = self.state["statistics"]["total_requests_made"]
        return total_requests > 0 and total_requests % 50 == 0

    def take_long_break(self):
        """Take a longer break to be extra respectful"""
        break_duration = 1800  # 30 minutes
        logger.info(f"Taking extended break: {break_duration/60:.1f} minutes")
        logger.info("This helps keep the server load minimal")

        # Break into smaller chunks so we can still respond to signals
        chunks = 36  # 30 minutes in 50-second chunks
        for i in range(chunks):
            time.sleep(50)
            if i % 6 == 0:  # Log every 5 minutes
                remaining = break_duration - ((i + 1) * 50)
                logger.info(f"Break continues... {remaining/60:.1f} minutes remaining")

    def collect_all_data(self):
        """Main collection loop that processes all matches"""
        basic_scraper = MNPScraper()
        self.selenium_scraper = MNPSeleniumScraper()

        try:
            # Get all match links
            logger.info("Getting complete match list...")
            all_match_links = basic_scraper.get_match_links()
            logger.info(f"Found {len(all_match_links)} total matches on the site")

            # Filter out already processed matches
            remaining_matches = [
                match
                for match in all_match_links
                if match not in self.state["completed_matches"]
                and match not in self.state["failed_matches"]
            ]

            logger.info(f"Remaining matches to process: {len(remaining_matches)}")

            if not remaining_matches:
                logger.info("All matches have been processed!")
                return self.finalize_collection()

            # Process each match
            for match_idx, match_url in enumerate(remaining_matches):
                logger.info(f"Processing match {match_idx + 1}/{len(remaining_matches)}")
                logger.info(f"Match URL: {match_url}")

                try:
                    # Process this match
                    match_success = self.process_match(basic_scraper, match_url, match_idx)

                    if match_success:
                        self.state["completed_matches"].append(match_url)
                        self.state["total_matches_processed"] += 1
                    else:
                        self.state["failed_matches"].append(match_url)

                    # Save progress after each match
                    self.state["last_processed_match"] = match_url
                    self.save_state()

                    # Take long break periodically
                    if self.should_take_long_break():
                        self.take_long_break()

                    # Dynamic delay between matches
                    if match_idx < len(remaining_matches) - 1:
                        match_delay = (
                            self.calculate_dynamic_delay(
                                self.state["statistics"]["total_requests_made"]
                            )
                            * 2
                        )
                        logger.info(f"Waiting {match_delay}s before next match...")
                        time.sleep(match_delay)

                except Exception as e:
                    logger.error(f"Error processing match {match_url}: {e}")
                    self.state["failed_matches"].append(match_url)
                    self.save_state()
                    # Continue with next match after error
                    time.sleep(self.base_delay)

            logger.info("All matches processed!")
            return self.finalize_collection()

        except Exception as e:
            logger.error(f"Fatal error in collection loop: {e}")
            self.save_state()
            raise
        finally:
            self.cleanup()

    def process_match(self, basic_scraper, match_url, match_idx):
        """Process a single match and its games"""
        try:
            # Get game links for this match
            game_links = basic_scraper.get_game_links_from_match(match_url)
            logger.info(f"  Found {len(game_links)} games in this match")

            match_data = []
            games_with_data = 0

            for game_idx, game_url in enumerate(game_links):
                logger.info(f"    Processing game {game_idx + 1}/{len(game_links)}")

                try:
                    # Scrape game data
                    game_data = self.selenium_scraper.scrape_game_with_selenium(game_url)
                    self.state["statistics"]["total_requests_made"] += 1

                    if game_data and game_data.get("js_data", {}).get("scores"):
                        scores = game_data.get("js_data", {}).get("scores", [])
                        logger.info(f"    ✓ Found scores: {scores}")

                        # Download images
                        valid_images = []
                        for img_idx, img_url in enumerate(game_data.get("images", [])):
                            if self.selenium_scraper.is_game_image(img_url):
                                filename = f"match_{self.state['total_matches_processed'] + 1}_game_{game_idx + 1}_img_{img_idx + 1}.png"
                                local_path = self.selenium_scraper.download_image(
                                    img_url, f"{self.output_dir}/images", filename
                                )
                                if local_path:
                                    valid_images.append(local_path)
                                    self.state["statistics"]["images_collected"] += 1

                        if valid_images:
                            game_data["local_images"] = valid_images
                            match_data.append(game_data)
                            games_with_data += 1
                            self.state["statistics"]["games_with_data"] += 1
                            logger.info(f"    ✓ Collected {len(valid_images)} images")
                        else:
                            self.state["statistics"]["games_without_data"] += 1
                    else:
                        self.state["statistics"]["games_without_data"] += 1
                        logger.info(f"    - No score data found")

                    # Delay between games within a match
                    if game_idx < len(game_links) - 1:
                        game_delay = self.calculate_dynamic_delay(
                            self.state["statistics"]["total_requests_made"]
                        )
                        time.sleep(game_delay)

                except Exception as e:
                    logger.error(f"    Error processing game {game_url}: {e}")
                    continue

            # Save match data if we found any
            if match_data:
                match_file = (
                    f"{self.output_dir}/match_{self.state['total_matches_processed'] + 1}_data.json"
                )
                with open(match_file, "w") as f:
                    json.dump(match_data, f, indent=2)

                self.state["total_games_collected"] += len(match_data)
                logger.info(
                    f"  ✓ Match complete: {games_with_data}/{len(game_links)} games had data"
                )
            else:
                logger.info(f"  - Match complete: No games with data found")

            return True

        except Exception as e:
            logger.error(f"Error processing match: {e}")
            return False

    def finalize_collection(self):
        """Finalize the complete collection"""
        logger.info("Finalizing complete dataset...")

        # Combine all match data files
        all_data = []
        match_files = list(Path(self.output_dir).glob("match_*_data.json"))

        for match_file in match_files:
            with open(match_file, "r") as f:
                match_data = json.load(f)
                all_data.extend(match_data)

        # Save complete dataset
        complete_file = f"{self.output_dir}/complete_dataset.json"
        with open(complete_file, "w") as f:
            json.dump(all_data, f, indent=2)

        logger.info(f"Complete dataset saved: {len(all_data)} games with data")

        # Process and format for training
        if all_data:
            logger.info("Processing for PaddleOCR training...")
            filtered_data = filter_dataset(self.output_dir)

            if filtered_data:
                formatter = PaddleOCRFormatter(
                    self.output_dir, f"{self.output_dir}/paddleocr_training"
                )
                num_samples = formatter.process_dataset()
                logger.info(f"Training dataset created: {num_samples} samples")

        # Final report
        self.generate_final_report()
        return all_data

    def generate_final_report(self):
        """Generate final collection report"""
        total_time = datetime.now() - datetime.fromisoformat(self.state["session_start"])

        report = {
            "session_summary": {
                "session_name": self.session_name,
                "total_duration": str(total_time),
                "total_matches_processed": self.state["total_matches_processed"],
                "total_games_collected": self.state["total_games_collected"],
                "total_images_downloaded": self.state["statistics"]["images_collected"],
                "total_requests_made": self.state["statistics"]["total_requests_made"],
            },
            "data_quality": {
                "games_with_data": self.state["statistics"]["games_with_data"],
                "games_without_data": self.state["statistics"]["games_without_data"],
                "success_rate": self.state["statistics"]["games_with_data"]
                / max(
                    1,
                    self.state["statistics"]["games_with_data"]
                    + self.state["statistics"]["games_without_data"],
                ),
            },
            "rate_limiting": {
                "base_delay": self.base_delay,
                "max_delay": self.max_delay,
                "average_delay": self.state["statistics"]["total_requests_made"]
                * self.base_delay
                / max(1, total_time.total_seconds()),
            },
        }

        report_file = f"{self.output_dir}/final_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("=" * 80)
        logger.info("PERSISTENT SCRAPING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Duration: {total_time}")
        logger.info(f"Matches processed: {self.state['total_matches_processed']}")
        logger.info(f"Games with data: {self.state['statistics']['games_with_data']}")
        logger.info(f"Images collected: {self.state['statistics']['images_collected']}")
        logger.info(f"Total requests: {self.state['statistics']['total_requests_made']}")
        logger.info(f"Success rate: {report['data_quality']['success_rate']:.1%}")
        logger.info(f"Final report: {report_file}")
        logger.info("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Persistent scraper for all MNP data")
    parser.add_argument(
        "--base-delay",
        type=int,
        default=60,
        help="Base delay between requests in seconds (default: 60)",
    )
    parser.add_argument(
        "--max-delay",
        type=int,
        default=300,
        help="Maximum delay between matches in seconds (default: 300)",
    )
    parser.add_argument("--session-name", type=str, help="Name for this scraping session")
    parser.add_argument("--resume", type=str, help="Resume a previous session by name")

    args = parser.parse_args()

    session_name = args.resume or args.session_name

    scraper = PersistentScraper(
        base_delay=args.base_delay, max_delay=args.max_delay, session_name=session_name
    )

    try:
        scraper.collect_all_data()
        logger.info("Collection completed successfully!")
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        raise


if __name__ == "__main__":
    main()
