#!/usr/bin/env python3

import sys
import os
import time
import json
import logging
import random
from datetime import datetime

# Add scripts directory to path
sys.path.append("scripts")

from scrape_mnp_data import MNPScraper
from scrape_mnp_selenium import MNPSeleniumScraper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("structured_scraper.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class StructuredMNPScraper:
    def __init__(
        self, base_delay=5, max_delay=30, session_name="structured_collection"
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.session_name = session_name
        self.output_dir = f"data/structured_collection_{session_name}"

        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/images", exist_ok=True)

        # Initialize scrapers
        self.basic_scraper = MNPScraper()
        self.selenium_scraper = MNPSeleniumScraper()

        # State tracking
        self.state = {
            "current_week": 1,
            "current_match_in_week": 0,
            "current_round": 1,
            "current_game": 1,
            "total_images_collected": 0,
            "total_games_processed": 0,
            "weeks_completed": [],
            "start_time": datetime.now().isoformat(),
            "consecutive_failures": 0,
            "max_consecutive_failures": 10,
            "failed_games": [],
            "last_successful_game": None,
        }

        logger.info(f"Structured scraper initialized: {session_name}")
        logger.info(f"Base delay: {base_delay}s, Max delay: {max_delay}s")
        logger.info(f"Output directory: {self.output_dir}")

    def get_week_matches(self, week_num):
        """Get all matches for a specific week"""
        logger.info(f"📅 WEEK {week_num}: Getting match list...")

        # Use the basic scraper to get all matches, then filter by week
        all_matches = self.basic_scraper.get_match_links()

        # Filter matches that contain week indicator (this may need adjustment based on URL patterns)
        week_matches = []
        for match_url in all_matches:
            # This is a simple approach - we may need to refine based on actual URL patterns
            if f"-{week_num}-" in match_url or f"week-{week_num}" in match_url.lower():
                week_matches.append(match_url)

        # If no week-specific filtering works, just take chunks of matches
        if not week_matches and week_num <= 14:
            matches_per_week = len(all_matches) // 14
            start_idx = (week_num - 1) * matches_per_week
            end_idx = (
                start_idx + matches_per_week if week_num < 14 else len(all_matches)
            )
            week_matches = all_matches[start_idx:end_idx]

        logger.info(f"📅 WEEK {week_num}: Found {len(week_matches)} matches")
        return week_matches

    def process_week(self, week_num):
        """Process all matches in a week"""
        logger.info(f"🎯 Starting WEEK {week_num}")

        week_matches = self.get_week_matches(week_num)
        week_images = 0
        week_games = 0

        for match_idx, match_url in enumerate(week_matches, 1):
            logger.info(f"📋 WEEK {week_num} - MATCH {match_idx}/{len(week_matches)}")
            logger.info(f"📋 Match URL: {match_url}")

            match_images, match_games = self.process_match(
                match_url, week_num, match_idx
            )
            week_images += match_images
            week_games += match_games

            # Delay between matches
            if match_idx < len(week_matches):
                delay = random.uniform(self.base_delay, self.max_delay)
                logger.info(f"⏱️  Waiting {delay:.1f}s before next match...")
                time.sleep(delay)

        logger.info(
            f"✅ WEEK {week_num} COMPLETE: {week_images} images, {week_games} games"
        )
        self.state["weeks_completed"].append(week_num)
        return week_images, week_games

    def process_match(self, match_url, week_num, match_idx):
        """Process all rounds and games in a match"""
        match_images = 0
        match_games = 0

        # Get games for this match
        games = self.basic_scraper.get_game_links_from_match(match_url)
        logger.info(f"🎮 Found {len(games)} games in match")

        # Group games by rounds (assuming 4 rounds, ~5 games each)
        games_per_round = max(1, len(games) // 4)

        for round_num in range(1, 5):  # Rounds 1-4
            round_start = (round_num - 1) * games_per_round
            round_end = round_start + games_per_round if round_num < 4 else len(games)
            round_games = games[round_start:round_end]

            if not round_games:
                continue

            logger.info(
                f"🎯 WEEK {week_num} - MATCH {match_idx} - ROUND {round_num}: {len(round_games)} games"
            )

            round_images = self.process_round(
                round_games, week_num, match_idx, round_num
            )
            match_images += round_images
            match_games += len(round_games)

        return match_images, match_games

    def process_round(self, round_games, week_num, match_idx, round_num):
        """Process all games in a round"""
        round_images = 0

        for game_idx, game_url in enumerate(round_games, 1):
            logger.info(
                f"🎲 WEEK {week_num} - MATCH {match_idx} - ROUND {round_num} - GAME {game_idx}/{len(round_games)}"
            )
            logger.info(f"🎲 Game URL: {game_url}")

            game_images = self.process_game(
                game_url, week_num, match_idx, round_num, game_idx
            )
            round_images += game_images

            # Circuit breaker: track failures and attempt recovery
            if game_images == 0:
                self.state["consecutive_failures"] += 1
                self.state["failed_games"].append(
                    {
                        "url": game_url,
                        "week": week_num,
                        "match": match_idx,
                        "round": round_num,
                        "game": game_idx,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                logger.warning(f"⚠️  Game failed: {game_url}")
                logger.warning(
                    f"⚠️  Consecutive failures: {self.state['consecutive_failures']}/{self.state['max_consecutive_failures']}"
                )

                if (
                    self.state["consecutive_failures"]
                    >= self.state["max_consecutive_failures"]
                ):
                    logger.error(
                        f"🛑 CIRCUIT BREAKER TRIGGERED after {self.state['consecutive_failures']} consecutive failures"
                    )
                    logger.error(
                        f"🛑 Failed games: {[game['url'] for game in self.state['failed_games'][-10:]]}"
                    )

                    # Try recovery: recreate selenium driver and skip to next match
                    logger.info(
                        f"🔄 Attempting recovery: recreating driver and skipping to next match..."
                    )
                    try:
                        if (
                            hasattr(self.selenium_scraper, "driver")
                            and self.selenium_scraper.driver
                        ):
                            self.selenium_scraper.driver.quit()
                        self.selenium_scraper.setup_driver()
                        self.state["consecutive_failures"] = (
                            0  # Reset after recovery attempt
                        )
                        logger.info(
                            f"✅ Recovery successful, continuing from next match"
                        )
                        return round_images  # Exit this round, continue to next
                    except Exception as recovery_error:
                        logger.error(f"❌ Recovery failed: {recovery_error}")
                        raise Exception(
                            f"Circuit breaker triggered and recovery failed: {self.state['consecutive_failures']} consecutive failures"
                        )
            else:
                self.state["consecutive_failures"] = 0  # Reset on success
                self.state["last_successful_game"] = {
                    "url": game_url,
                    "week": week_num,
                    "match": match_idx,
                    "round": round_num,
                    "game": game_idx,
                    "timestamp": datetime.now().isoformat(),
                }

            self.state["total_games_processed"] += 1

            # Quick delay between games
            if game_idx < len(round_games):
                delay = random.uniform(2, 5)  # Shorter delay between games
                time.sleep(delay)

        logger.info(f"🎯 ROUND {round_num} COMPLETE: {round_images} images")
        return round_images

    def process_game(self, game_url, week_num, match_idx, round_num, game_idx):
        """Process a single game and download images"""
        try:
            # Scrape game data
            game_data = self.selenium_scraper.scrape_game_with_selenium(game_url)

            if not game_data:
                logger.warning(f"❌ No data returned for game")
                return 0

            # Check for scores
            scores = game_data.get("js_data", {}).get("scores", [])
            if scores:
                logger.info(f"💯 Scores found: {scores}")

            # Download images
            images_list = game_data.get("images", [])
            game_images = 0

            for img_idx, img_url in enumerate(images_list):
                if self.selenium_scraper.is_game_image(img_url):
                    filename = f"w{week_num}_m{match_idx}_r{round_num}_g{game_idx}_img{img_idx + 1}.png"
                    local_path = self.selenium_scraper.download_image(
                        img_url, f"{self.output_dir}/images", filename
                    )
                    if local_path:
                        game_images += 1
                        self.state["total_images_collected"] += 1
                        logger.info(f"📸 Image saved: {filename}")

            logger.info(f"✅ Game complete: {game_images} images")
            return game_images

        except Exception as e:
            logger.error(f"❌ Game failed: {e}")
            return 0

    def save_state(self):
        """Save current state to file"""
        state_file = f"{self.output_dir}/scraper_state.json"
        with open(state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def get_failure_report(self):
        """Generate a report of failed games for debugging"""
        total_failures = len(self.state["failed_games"])
        if total_failures == 0:
            return "No failed games recorded."

        lines = []
        lines.append("📊 FAILURE REPORT:")
        lines.append(f"Total failed games: {total_failures}")
        lines.append(f"Consecutive failures: {self.state['consecutive_failures']}")

        if self.state["last_successful_game"]:
            last_success = self.state["last_successful_game"]
            lines.append(
                f"Last successful game: {last_success['url']} at {last_success['timestamp']}"
            )

        lines.append("Recent failed games:")
        for game in self.state["failed_games"][-5:]:
            lines.append(
                f"  - {game['url']} (W{game['week']} M{game['match']} R{game['round']} G{game['game']})"
            )

        return "\n".join(lines)

    def run_structured_collection(self, start_week=1, end_week=14):
        """Run the structured collection process"""
        logger.info(f"🚀 Starting structured collection: Weeks {start_week}-{end_week}")

        total_images = 0
        total_games = 0

        for week_num in range(start_week, end_week + 1):
            self.state["current_week"] = week_num

            week_images, week_games = self.process_week(week_num)
            total_images += week_images
            total_games += week_games

            self.save_state()

            # Longer break between weeks
            if week_num < end_week:
                logger.info(f"🛌 Taking 5-minute break between weeks...")
                time.sleep(300)  # 5 minutes

        logger.info(f"🎉 COLLECTION COMPLETE!")
        logger.info(
            f"📊 Total: {total_images} images from {total_games} games across {end_week - start_week + 1} weeks"
        )

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self.selenium_scraper, "cleanup"):
            self.selenium_scraper.cleanup()
        elif hasattr(self.selenium_scraper, "driver") and self.selenium_scraper.driver:
            self.selenium_scraper.driver.quit()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Structured MNP scraper following site organization"
    )
    parser.add_argument(
        "--start-week", type=int, default=1, help="Week to start from (1-14)"
    )
    parser.add_argument(
        "--end-week", type=int, default=14, help="Week to end at (1-14)"
    )
    parser.add_argument(
        "--base-delay", type=int, default=5, help="Base delay between requests"
    )
    parser.add_argument(
        "--max-delay", type=int, default=30, help="Max delay between matches"
    )
    parser.add_argument(
        "--session-name", type=str, default="week1to14", help="Session name"
    )

    args = parser.parse_args()

    scraper = StructuredMNPScraper(
        base_delay=args.base_delay,
        max_delay=args.max_delay,
        session_name=args.session_name,
    )

    try:
        scraper.run_structured_collection(args.start_week, args.end_week)
    except KeyboardInterrupt:
        logger.info("🛑 Scraping interrupted by user")
    finally:
        scraper.cleanup()


if __name__ == "__main__":
    main()
