# Pinball Score Tracker

[![Run Tests](https://github.com/toasty/pinball-score-tracker/actions/workflows/test.yml/badge.svg)](https://github.com/toasty/pinball-score-tracker/actions/workflows/test.yml)
[![Code Quality](https://github.com/toasty/pinball-score-tracker/actions/workflows/lint.yml/badge.svg)](https://github.com/toasty/pinball-score-tracker/actions/workflows/lint.yml)

A way to track your pinball scores using AI-powered computer vision! This system automatically extracts scores from pinball machine photos without requiring manual entry.

## Features

- 🤖 **AI-powered score extraction** using PaddleOCR
- 📸 **Automatic photo processing** from pinball machine displays
- 🌍 **Location-based comparisons** with other players
- 🔧 **Containerized training** for AWS deployment
- 🧪 **Comprehensive testing** with 17 test cases

## Project Structure

```
pinball-score-tracker/
├── ml-training/           # ML training pipeline
│   ├── scripts/          # Data scraping and formatting tools
│   ├── tests/            # Comprehensive test suite (17 tests)
│   ├── data/             # Training datasets (gitignored)
│   └── requirements.txt  # Python dependencies
├── .github/workflows/    # CI/CD pipelines
├── CLAUDE.md            # Development guidance
└── README.md           # This file
```

## Getting Started

### Prerequisites

- Python 3.9+
- Chrome browser (for web scraping)
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pinball-score-tracker.git
cd pinball-score-tracker
```

2. Set up the ML training environment:
```bash
cd ml-training
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running Tests

```bash
cd ml-training
source venv/bin/activate
python -m pytest tests/ -v
```

## ML Training Pipeline

### 1. Data Collection

Collect pinball score photos with ground truth data:

```bash
# Scrape data from Monday Night Pinball (with Selenium for dynamic content)
python scripts/scrape_mnp_selenium.py

# Filter out blank/invalid images
python scripts/filter_valid_images.py
```

### 2. Data Formatting

Format collected data for PaddleOCR training:

```bash
# Create training annotations and datasets
python scripts/format_paddleocr_data.py
```

### 3. Model Training

Train PaddleOCR model on pinball-specific data:

```bash
# TODO: Implement training script
python scripts/train_model.py
```

## Architecture

### Technology Stack

- **ML Framework**: PaddleOCR for end-to-end text detection and recognition
- **Data Collection**: Selenium WebDriver for dynamic content scraping
- **Training Infrastructure**: Docker containers deployed on AWS Batch
- **Testing**: pytest with comprehensive mocking (no real network requests)

### Model Approach

1. **Text Detection**: Locate score regions in pinball photos
2. **Text Recognition**: Extract numerical scores using OCR
3. **Post-processing**: Format and validate extracted scores

### Data Flow

```
Pinball Photos → PaddleOCR → Score Detection → Validation → Database
```

## Development

### Code Quality

- **Testing**: 17 comprehensive tests with full mocking
- **CI/CD**: GitHub Actions for automated testing and linting
- **Code Style**: Black formatting, isort imports, flake8 linting

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Testing Strategy

All tests are fully mocked to prevent real network requests:

- **Web Scraping Tests**: Mock HTTP requests and Selenium WebDriver
- **Image Processing Tests**: Use temporary files and synthetic data
- **Data Formatting Tests**: Test annotation generation and file operations

```bash
# Run specific test categories
python -m pytest tests/test_scraper.py -v       # Web scraping
python -m pytest tests/test_filter_images.py -v # Image filtering
python -m pytest tests/test_format_data.py -v   # Data formatting
```

## Deployment

### AWS Training Pipeline

1. **Containerization**: Docker images with PaddleOCR and dependencies
2. **Batch Jobs**: AWS Batch for scalable training
3. **Model Storage**: S3 for trained model artifacts
4. **Monitoring**: CloudWatch for training metrics

### Local Development

```bash
# Install development dependencies
pip install pytest pytest-mock black isort flake8

# Run tests
python -m pytest tests/ -v

# Format code
black ml-training/
isort ml-training/

# Lint code
flake8 ml-training/
```

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Monday Night Pinball for providing real-world data
- PaddleOCR team for the excellent OCR framework
- Claude Code for development assistance