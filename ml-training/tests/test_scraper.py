#!/usr/bin/env python3
"""
Tests for web scraping functionality
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

class TestMNPScraper:
    
    def test_scraper_initialization(self):
        """Test basic scraper initialization"""
        from scrape_mnp_data import MNPScraper
        
        scraper = MNPScraper()
        assert scraper.base_url == "https://www.mondaynightpinball.com"
        assert hasattr(scraper, 'session')
    
    @patch('scrape_mnp_data.requests.Session.get')
    def test_get_match_links(self, mock_get):
        """Test match link extraction"""
        from scrape_mnp_data import MNPScraper
        
        # Mock response
        mock_response = Mock()
        mock_response.content = b'''
        <html>
            <a href="/matches/mnp-21-1-ABC-DEF">Match 1</a>
            <a href="/matches/mnp-21-2-GHI-JKL">Match 2</a>
            <a href="/other-link">Other</a>
        </html>
        '''
        mock_get.return_value = mock_response
        
        scraper = MNPScraper()
        links = scraper.get_match_links()
        
        assert len(links) == 2
        assert "mnp-21-1-ABC-DEF" in links[0]
        assert "mnp-21-2-GHI-JKL" in links[1]
    
    def test_game_link_construction(self):
        """Test game link construction from match URL"""
        from scrape_mnp_data import MNPScraper
        
        scraper = MNPScraper()
        
        # Test with mock response (no explicit game links)
        with patch.object(scraper.session, 'get') as mock_get:
            mock_response = Mock()
            mock_response.content = b'<html>No game links</html>'
            mock_get.return_value = mock_response
            
            match_url = "https://www.mondaynightpinball.com/matches/mnp-21-1-ABC-DEF"
            game_links = scraper.get_game_links_from_match(match_url)
            
            # Should construct 20 game links (5 rounds × 4 games)
            assert len(game_links) == 20
            assert "mnp-21-1-ABC-DEF.1.1" in game_links[0]
            assert "mnp-21-1-ABC-DEF.5.4" in game_links[-1]

class TestSeleniumScraper:
    
    @pytest.fixture
    def mock_driver(self):
        """Mock Selenium WebDriver"""
        with patch('scrape_mnp_selenium.webdriver.Chrome') as mock_chrome:
            driver = Mock()
            mock_chrome.return_value = driver
            yield driver
    
    def test_is_game_image_filtering(self):
        """Test game image filtering logic"""
        from scrape_mnp_selenium import MNPSeleniumScraper
        
        scraper = MNPSeleniumScraper()
        
        # Should filter out UI images
        assert not scraper.is_game_image("https://example.com/icon.png")
        assert not scraper.is_game_image("https://example.com/logo.jpg")
        assert not scraper.is_game_image("https://example.com/button.png")
        
        # Should keep game images
        assert scraper.is_game_image("https://example.com/upload/game123.jpg")
        assert scraper.is_game_image("https://example.com/photo/score.png")
        assert scraper.is_game_image("data:image/png;base64,iVBOR...")
        
        scraper.close()
    
    def test_scraper_initialization_mock(self):
        """Test Selenium scraper initialization with mocked driver"""
        with patch('scrape_mnp_selenium.ChromeDriverManager') as mock_cdm, \
             patch('scrape_mnp_selenium.webdriver.Chrome') as mock_chrome:
            
            mock_cdm.return_value.install.return_value = "/path/to/driver"
            mock_driver = Mock()
            mock_chrome.return_value = mock_driver
            
            from scrape_mnp_selenium import MNPSeleniumScraper
            scraper = MNPSeleniumScraper()
            
            assert hasattr(scraper, 'driver')
            mock_driver.implicitly_wait.assert_called_with(10)
            
            scraper.close()

class TestImageDownloading:
    
    @patch('scrape_mnp_data.requests.Session.get')
    def test_download_regular_image(self, mock_get):
        """Test downloading regular HTTP images"""
        from scrape_mnp_data import MNPScraper
        
        # Mock response
        mock_response = Mock()
        mock_response.content = b"fake image data"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        scraper = MNPScraper()
        
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            result = scraper.download_image("https://example.com/image.jpg", temp_dir)
            
            assert result is not None
            assert "image.jpg" in result
            assert os.path.exists(result)
    
    def test_download_data_url_image(self):
        """Test downloading data URL images (canvas)"""
        from scrape_mnp_selenium import MNPSeleniumScraper
        
        scraper = MNPSeleniumScraper()
        
        # Mock data URL
        data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            result = scraper.download_image(data_url, temp_dir, "test.png")
            
            assert result is not None
            assert os.path.exists(result)
            
        scraper.close()