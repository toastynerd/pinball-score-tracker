#!/usr/bin/env python3
"""
Scraper for Monday Night Pinball data
Extracts images and scores for ML training dataset
"""

import json
import logging
import os
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MNPScraper:
    def __init__(self, base_url="https://www.mondaynightpinball.com"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )

    def get_match_links(self):
        """Extract all match links from matches page"""
        url = f"{self.base_url}/matches"
        response = self.session.get(url, timeout=30)
        soup = BeautifulSoup(response.content, "html.parser")

        # Find all match links
        match_links = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "/matches/mnp-" in href:
                full_url = urljoin(self.base_url, href)
                match_links.append(full_url)

        logger.info(f"Found {len(match_links)} match links")
        return match_links

    def get_game_links_from_match(self, match_url):
        """Extract individual game links from a match page"""
        response = self.session.get(match_url, timeout=30)
        soup = BeautifulSoup(response.content, "html.parser")

        game_links = []

        # First try to find explicit game links
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "/games/mnp-" in href:
                full_url = urljoin(self.base_url, href)
                game_links.append(full_url)

        # If no links found, construct them from match URL
        if not game_links:
            # Extract match ID from URL: /matches/mnp-21-1-ADB-JMF
            match_id = match_url.split("/matches/")[-1]

            # Construct game URLs - typically 4 games per match
            for round_num in [1, 2, 3, 4, 5]:  # Try multiple rounds
                for game_num in [1, 2, 3, 4]:  # Typically 4 games per round
                    game_url = (
                        f"{self.base_url}/games/{match_id}.{round_num}.{game_num}"
                    )
                    game_links.append(game_url)

        logger.info(f"Found/constructed {len(game_links)} game links for match")
        return game_links

    def scrape_game_data(self, game_url):
        """Extract images and scores from a game page"""
        try:
            response = self.session.get(game_url, timeout=30)

            # Check if page exists
            if response.status_code != 200:
                logger.warning(f"Page not found: {game_url}")
                return None

            soup = BeautifulSoup(response.content, "html.parser")

            game_data = {
                "url": game_url,
                "images": [],
                "scores": [],
                "players": [],
                "machine": None,
                "status": "success",
            }

            # Extract all images
            for img in soup.find_all("img"):
                if img.get("src"):
                    img_url = urljoin(self.base_url, img["src"])
                    # Include all images for now, filter later
                    game_data["images"].append(img_url)

            # Look for data in various input types
            inputs = soup.find_all("input")
            for input_elem in inputs:
                input_type = input_elem.get("type", "")
                input_value = input_elem.get("value", "")
                input_name = input_elem.get("name", "")

                if input_value and input_value.isdigit():
                    game_data["scores"].append(
                        {"value": input_value, "type": input_type, "name": input_name}
                    )

            # Extract text content that might contain scores
            text_content = soup.get_text()

            # Look for machine name in title or headings
            title = soup.find("title")
            if title:
                game_data["machine"] = title.get_text().strip()

            logger.info(
                f"Game {game_url}: {len(game_data['images'])} images, {len(game_data['scores'])} score fields"
            )
            return game_data

        except Exception as e:
            logger.error(f"Error scraping {game_url}: {e}")
            return None

    def download_image(self, img_url, save_dir):
        """Download an image from URL"""
        try:
            response = self.session.get(img_url, timeout=30)
            response.raise_for_status()

            # Generate filename from URL
            parsed_url = urlparse(img_url)
            filename = os.path.basename(parsed_url.path)
            if not filename:
                filename = f"image_{hash(img_url)}.jpg"

            filepath = os.path.join(save_dir, filename)

            with open(filepath, "wb") as f:
                f.write(response.content)

            return filepath

        except Exception as e:
            logger.error(f"Failed to download {img_url}: {e}")
            return None

    def scrape_all_data(self, output_dir="data/mnp_dataset", max_matches=None):
        """Main scraping function"""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/images", exist_ok=True)

        # Get all match links
        match_links = self.get_match_links()

        if max_matches:
            match_links = match_links[:max_matches]

        all_game_data = []

        for i, match_url in enumerate(match_links):
            logger.info(f"Processing match {i+1}/{len(match_links)}: {match_url}")

            # Get game links from this match
            game_links = self.get_game_links_from_match(match_url)

            for game_url in game_links[:5]:  # Limit to first 5 games for testing
                logger.info(f"Processing game: {game_url}")

                # Scrape game data
                game_data = self.scrape_game_data(game_url)

                if game_data is None:
                    continue

                # Download images
                for img_url in game_data["images"]:
                    local_path = self.download_image(img_url, f"{output_dir}/images")
                    if local_path:
                        game_data["local_images"] = game_data.get("local_images", [])
                        game_data["local_images"].append(local_path)

                all_game_data.append(game_data)

                # Be respectful with requests
                time.sleep(1)

        # Save metadata
        with open(f"{output_dir}/dataset_metadata.json", "w") as f:
            json.dump(all_game_data, f, indent=2)

        logger.info(f"Scraping complete. Collected {len(all_game_data)} games")
        return all_game_data


if __name__ == "__main__":
    scraper = MNPScraper()

    # Test with just 1 match to get ~10 photos
    data = scraper.scrape_all_data(max_matches=1)

    print(f"Collected data for {len(data)} games")

    # Print summary
    total_images = sum(len(game.get("images", [])) for game in data)
    print(f"Total images found: {total_images}")

    for i, game in enumerate(data[:3]):  # Show first 3 games
        print(
            f"Game {i+1}: {len(game.get('images', []))} images, {len(game.get('scores', []))} scores"
        )
