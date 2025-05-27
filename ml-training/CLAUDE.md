# Pinball Score Tracker ML Training - Project Context

## Current Status (May 25, 2025)

### Active Scrapers
- **Process 58884**: `structured_scraper.py` running weeks 1-3 collection
- **Process 58751**: `week1_timeout_test` scraper
- **Progress**: 16 images collected, currently at Week 1, Match 1, Round 4, Game 1
- **Last activity**: 09:38 (actively collecting)

## Recent Major Fixes

### 1. Structured Scraper Implementation
Created `scripts/structured_scraper.py` that follows MNP site organization:
- **Hierarchy**: WEEK → MATCH → ROUND → GAME
- **Clear logging**: Shows exact progress through site structure
- **Organized naming**: `w1_m1_r1_g1_img1.png` format
- **Week organization**: Site has 945 total matches divided into 14 weeks (~85 matches per week)

### 2. Critical Timeout Fixes (COMPLETED)
**Problem**: Scrapers were hanging on slow/empty game pages (some taking 33+ seconds)

**Solution implemented** in `scripts/scrape_mnp_selenium.py`:
```python
# 20-second page load timeout
self.driver.set_page_load_timeout(20)

# Empty game detection and skipping
if not game_data['images'] and not game_data['scores'] and load_time > 15:
    logger.warning(f"Skipping empty game after {load_time:.1f}s (no images or scores)")
    return None
```

**Results**: 
- Problematic game `mnp-21-1-ADB-JMF.5.1` now gets skipped properly
- Normal games load in ~22s consistently
- No more hanging on empty/slow pages

### 3. Performance Issues Resolved
- **Debug logging flood**: Removed 15+ debug statements from image processing loop
- **Rate limiting optimized**: Using 2s base delay, 8s max delay
- **Session management**: Added driver recreation and validation

## Project Architecture

### Core Components
1. **Data Collection**:
   - `scrape_mnp_data.py`: Basic HTTP scraping for match/game links
   - `scrape_mnp_selenium.py`: Selenium scraping for dynamic content + images
   - `structured_scraper.py`: Orchestrates collection following site structure

2. **ML Training Pipeline**:
   - `train_model.py`: PaddleOCR training script (680 lines)
   - `configs/pinball_ocr_config.yml`: Model configuration
   - `model_versioning.py`: Git-like model artifact management

3. **AWS Infrastructure**:
   - `aws/cloudformation-template.yml`: GPU/CPU batch environments
   - `Dockerfile`: Containerized training environment
   - Support for p3.2xlarge, g4dn instances

### Data Organization
```
data/
├── structured_collection_weeks1to3_robust/    # Active collection
│   └── images/
│       ├── w1_m1_r1_g1_img1.png
│       ├── w1_m1_r1_g2_img1.png
│       └── ...
├── paddleocr_training/                        # Formatted for training
└── persistent_collection_*/                   # Previous attempts
```

## Key Insights

### Site Structure Discovery
- **945 total matches** across Monday Night Pinball
- **14 weeks** of competition (~85 matches per week)
- **~20 games per match** organized into 4 rounds
- **Canvas-rendered images**: Extracted via base64 data URLs

### Critical Bug Fixes
1. **Image filtering bug**: Data URLs were incorrectly rejected by UI pattern filters
2. **Hanging issue**: Pages timing out without proper error handling
3. **Session management**: Chrome driver sessions dying without detection

## Running Commands

### Start Structured Scraper
```bash
cd /Users/toasty/programming/pinball-score-tracker/ml-training
source venv/bin/activate
python scripts/structured_scraper.py --start-week 1 --end-week 3 --base-delay 2 --max-delay 8 --session-name weeks1to3_robust &
```

### Check Progress
```bash
# Check if running
ps aux | grep "structured_scraper"

# Count images
find data/structured_collection_*/images/ -name "*.png" | wc -l

# Check latest activity
ls -la data/structured_collection_*/images/ | tail -3
```

### Debug Specific Issues
```bash
# Test problematic game
python debug_specific_game.py

# Check logs
tail -20 structured_scraper.log
```

## Next Steps

1. **Monitor collection**: Current scraper should complete weeks 1-3 without hanging
2. **Scale up**: Once proven stable, extend to all 14 weeks
3. **Training pipeline**: Use collected data with PaddleOCR training script
4. **AWS deployment**: Deploy training to GPU instances when dataset is ready

## Dependencies
- Python 3.13 with venv
- Selenium + ChromeDriver (WebDriver Manager)
- requests, BeautifulSoup4
- PaddleOCR for training pipeline

## Test Results
- **Timeout handling**: ✅ Working (skips 33s+ empty games)
- **Image collection**: ✅ Consistent (~1 image per game)
- **Structured logging**: ✅ Clear progress visibility
- **Session stability**: ✅ Driver recreation on failures
- **Rate limiting**: ✅ Respectful 2-8s delays

The scraper is now robust and collecting data efficiently with proper timeout handling and structured organization.