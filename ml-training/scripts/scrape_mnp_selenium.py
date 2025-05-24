#!/usr/bin/env python3
"""
Selenium-based scraper for Monday Night Pinball data
Handles dynamic JavaScript content to extract actual game photos
"""

import json
import logging
import os
import time
from urllib.parse import urljoin, urlparse

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MNPSeleniumScraper:
    def __init__(self):
        self.setup_driver()

    def setup_driver(self):
        """Initialize Chrome driver with appropriate options"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in background
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")

        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.implicitly_wait(10)

    def scrape_game_with_selenium(self, game_url):
        """Scrape game data using Selenium to handle JavaScript"""
        try:
            logger.info(f"Loading page: {game_url}")
            self.driver.get(game_url)

            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # Additional wait for dynamic content
            time.sleep(3)

            game_data = {
                "url": game_url,
                "images": [],
                "scores": [],
                "players": [],
                "machine": None,
                "status": "success",
            }

            # Get page title
            game_data["machine"] = self.driver.title

            # Find all images on the page
            img_elements = self.driver.find_elements(By.TAG_NAME, "img")
            for img in img_elements:
                src = img.get_attribute("src")
                if src and self.is_game_image(src):
                    game_data["images"].append(src)

            # Look for canvas elements (might contain uploaded images)
            canvas_elements = self.driver.find_elements(By.TAG_NAME, "canvas")
            for canvas in canvas_elements:
                # Try to get canvas as image data
                canvas_data = self.driver.execute_script(
                    "return arguments[0].toDataURL('image/png');", canvas
                )
                if canvas_data and canvas_data != "data:,":
                    game_data["images"].append(canvas_data)

            # Look for score inputs
            score_inputs = self.driver.find_elements(
                By.CSS_SELECTOR, "input[type='number'], input[name*='score']"
            )
            for input_elem in score_inputs:
                value = input_elem.get_attribute("value")
                name = input_elem.get_attribute("name") or input_elem.get_attribute("id")
                if value and value.isdigit():
                    game_data["scores"].append({"value": value, "name": name, "type": "number"})

            # Look for player names
            player_inputs = self.driver.find_elements(
                By.CSS_SELECTOR, "input[name*='player'], select[name*='player']"
            )
            for input_elem in player_inputs:
                value = input_elem.get_attribute("value")
                if value:
                    game_data["players"].append(value)

            # Check for any data in JavaScript variables
            js_data = self.driver.execute_script(
                """
                var gameData = {};
                if (typeof gameId !== 'undefined') gameData.gameId = gameId;
                if (typeof players !== 'undefined') gameData.players = players;
                if (typeof scores !== 'undefined') gameData.scores = scores;
                if (typeof machineId !== 'undefined') gameData.machineId = machineId;
                return gameData;
            """
            )

            if js_data:
                game_data["js_data"] = js_data

            logger.info(
                f"Found {len(game_data['images'])} images, {len(game_data['scores'])} scores"
            )
            return game_data

        except Exception as e:
            logger.error(f"Error scraping {game_url}: {e}")
            return None

    def is_game_image(self, img_src):
        """Filter out UI images, keep only actual game photos"""
        if not img_src:
            return False

        # IMPORTANT: Check data URLs FIRST before other patterns
        # Canvas-rendered game images are always data URLs
        if img_src.startswith("data:image"):
            return True

        # Skip common UI images
        ui_patterns = ["icon", "logo", "check", "button", "ui", "header", "footer"]
        src_lower = img_src.lower()

        if any(pattern in src_lower for pattern in ui_patterns):
            return False

        # Look for game/upload patterns
        game_patterns = ["upload", "game", "photo", "score"]
        if any(pattern in src_lower for pattern in game_patterns):
            return True

        return False

    def download_image(self, img_url, save_dir, filename=None):
        """Download image from URL or save data URL"""
        try:
            # Ensure save directory exists
            os.makedirs(save_dir, exist_ok=True)
            
            if img_url.startswith("data:image"):
                # Handle data URLs
                if not filename:
                    filename = f"canvas_image_{hash(img_url)}.png"

                filepath = os.path.join(save_dir, filename)

                # Extract base64 data
                header, data = img_url.split(",", 1)
                import base64

                with open(filepath, "wb") as f:
                    f.write(base64.b64decode(data))

                return filepath
            else:
                # Handle regular URLs
                response = requests.get(img_url)
                response.raise_for_status()

                if not filename:
                    parsed_url = urlparse(img_url)
                    filename = os.path.basename(parsed_url.path) or f"image_{hash(img_url)}.jpg"

                filepath = os.path.join(save_dir, filename)

                with open(filepath, "wb") as f:
                    f.write(response.content)

                return filepath

        except Exception as e:
            logger.error(f"Failed to download {img_url}: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def scrape_sample_games(self, game_urls, output_dir="data/selenium_dataset"):
        """Scrape a list of specific game URLs"""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/images", exist_ok=True)

        all_game_data = []

        for i, game_url in enumerate(game_urls):
            logger.info(f"Processing game {i+1}/{len(game_urls)}: {game_url}")

            game_data = self.scrape_game_with_selenium(game_url)
            if not game_data:
                continue

            # Download images
            for j, img_url in enumerate(game_data["images"]):
                filename = f"game_{i+1}_img_{j+1}.png"
                local_path = self.download_image(img_url, f"{output_dir}/images", filename)
                if local_path:
                    game_data["local_images"] = game_data.get("local_images", [])
                    game_data["local_images"].append(local_path)

            all_game_data.append(game_data)

            # Be respectful with requests
            time.sleep(2)

        # Save metadata
        with open(f"{output_dir}/dataset_metadata.json", "w") as f:
            json.dump(all_game_data, f, indent=2)

        logger.info(f"Scraping complete. Collected {len(all_game_data)} games")
        return all_game_data

    def close(self):
        """Clean up the driver"""
        if hasattr(self, "driver"):
            self.driver.quit()


if __name__ == "__main__":
    # Test URLs - the one you mentioned plus a few others
    test_urls = [
        "https://www.mondaynightpinball.com/games/mnp-21-14-PBR-NLT.4.2",
        "https://www.mondaynightpinball.com/games/mnp-21-14-PBR-NLT.4.1",
        "https://www.mondaynightpinball.com/games/mnp-21-14-PBR-NLT.3.2",
    ]

    scraper = MNPSeleniumScraper()

    try:
        data = scraper.scrape_sample_games(test_urls)

        print(f"Collected data for {len(data)} games")
        total_images = sum(len(game.get("local_images", [])) for game in data)
        print(f"Total images downloaded: {total_images}")

        for i, game in enumerate(data):
            print(
                f"Game {i+1}: {len(game.get('local_images', []))} images, {len(game.get('scores', []))} scores"
            )

    finally:
        scraper.close()
