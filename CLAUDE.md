# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pinball Score Tracker is an AI-powered application for tracking pinball scores. The application aims to:
- Automatically extract scores using AI (no manual entry required)
- Track user location for comparing progress with other players locally and globally
- Provide score tracking and comparison features

## Current Status

This is a new project with minimal initial setup. The codebase structure and technology stack are yet to be determined.

## ML Model Development

**Technology Stack:**
- PyTorch for ML model training and inference
- Hugging Face models as base/pretrained models
- AWS for containerized training infrastructure

**Model Architecture:**
- PaddleOCR for end-to-end text detection and recognition from pinball score pictures
- Handles multiple numbers in single image (detection + recognition pipeline)
- Fine-tuning on pinball-specific score images
- Training pipeline designed for AWS Batch execution

## Data Collection System

**Web Scrapers for Training Data:**

1. **Basic HTTP Scraper** (`scrape_mnp_data.py`)
   - Static content scraping with requests/BeautifulSoup
   - Game URL construction algorithm
   - Image download with error handling
   - Used for extracting match and game structure

2. **Selenium-based Scraper** (`scrape_mnp_selenium.py`)
   - **Primary scraper** for actual training data
   - Handles JavaScript-rendered content
   - Captures canvas images (actual pinball score photos)
   - Extracts ground truth scores from JS variables
   - Image filtering logic to separate game photos from UI elements
   - Headless Chrome support for automation

**Data Processing Pipeline:**
- `filter_valid_images.py`: Removes blank/corrupted images using file size and variance analysis
- `format_paddleocr_data.py`: Creates PaddleOCR training annotations with bounding boxes

**Testing Strategy:**
- All scrapers use comprehensive mocking to prevent real network requests during testing
- 15 test cases covering scraping logic, image filtering, and data formatting
- CI/CD with GitHub Actions ensuring code quality

## Development Notes

When setting up this project, consider:
- Technology stack selection (web app, mobile app, or both)
- Location services integration  
- Database design for scores, users, and locations
- Authentication system for user accounts