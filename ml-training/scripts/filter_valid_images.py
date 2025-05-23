#!/usr/bin/env python3
"""
Filter out blank/empty images from scraped dataset
"""

import os
import json
from PIL import Image
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_blank_image(image_path, threshold=0.95):
    """
    Check if an image is blank/empty based on:
    1. File size
    2. Color variance
    3. Edge detection
    """
    try:
        # Check file size first
        file_size = os.path.getsize(image_path)
        if file_size < 2000:  # Less than 2KB likely blank
            return True
            
        # Load image
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Check if image is mostly single color
        if len(img_array.shape) == 3:
            # Color image - check variance across channels
            variance = np.var(img_array, axis=(0, 1))
            mean_variance = np.mean(variance)
        else:
            # Grayscale image
            variance = np.var(img_array)
            mean_variance = variance
            
        # Very low variance indicates blank image
        if mean_variance < 10:
            return True
            
        # Check if image is mostly transparent (if has alpha channel)
        if img.mode == 'RGBA':
            alpha_channel = img_array[:, :, 3]
            non_transparent_pixels = np.sum(alpha_channel > 0)
            total_pixels = alpha_channel.size
            
            if non_transparent_pixels / total_pixels < 0.1:  # Less than 10% visible
                return True
        
        # Check for mostly uniform color
        if len(img_array.shape) == 3:
            # Convert to grayscale for edge detection
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
            
        # Simple edge detection - count significant gradients
        dx = np.abs(np.diff(gray, axis=1))
        dy = np.abs(np.diff(gray, axis=0))
        
        edge_pixels = np.sum(dx > 10) + np.sum(dy > 10)
        total_possible_edges = gray.size
        
        edge_ratio = edge_pixels / total_possible_edges
        
        # If very few edges, likely blank
        if edge_ratio < 0.01:  # Less than 1% edges
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"Error checking image {image_path}: {e}")
        return True  # Assume blank if can't process

def filter_dataset(dataset_dir):
    """Filter out blank images from dataset"""
    metadata_file = os.path.join(dataset_dir, 'dataset_metadata.json')
    
    if not os.path.exists(metadata_file):
        logger.error(f"Metadata file not found: {metadata_file}")
        return
        
    # Load metadata
    with open(metadata_file, 'r') as f:
        games_data = json.load(f)
    
    filtered_games = []
    total_images = 0
    valid_images = 0
    
    for game in games_data:
        filtered_game = game.copy()
        filtered_game['local_images'] = []
        
        for img_path in game.get('local_images', []):
            total_images += 1
            # Handle relative paths correctly
            if os.path.isabs(img_path):
                full_path = img_path
            else:
                # Remove dataset_dir from path if it's duplicated
                clean_path = img_path.replace(f'{dataset_dir}/', '').replace(dataset_dir, '')
                full_path = os.path.join(dataset_dir, clean_path)
            
            if os.path.exists(full_path):
                if not is_blank_image(full_path):
                    filtered_game['local_images'].append(img_path)
                    valid_images += 1
                    logger.info(f"Valid image: {img_path}")
                else:
                    logger.info(f"Blank image removed: {img_path}")
            else:
                logger.warning(f"Image not found: {full_path}")
        
        # Only keep games with valid images and scores
        if filtered_game['local_images'] and game.get('js_data', {}).get('scores'):
            filtered_games.append(filtered_game)
    
    # Save filtered metadata
    filtered_file = os.path.join(dataset_dir, 'filtered_dataset_metadata.json')
    with open(filtered_file, 'w') as f:
        json.dump(filtered_games, f, indent=2)
    
    logger.info(f"Dataset filtering complete:")
    logger.info(f"  Total images: {total_images}")
    logger.info(f"  Valid images: {valid_images}")
    logger.info(f"  Games with valid data: {len(filtered_games)}")
    logger.info(f"  Filtered metadata saved to: {filtered_file}")
    
    return filtered_games

if __name__ == "__main__":
    dataset_dir = "data/selenium_dataset"
    filtered_data = filter_dataset(dataset_dir)
    
    print(f"Filtered dataset contains {len(filtered_data)} valid games")
    for i, game in enumerate(filtered_data):
        scores = game.get('js_data', {}).get('scores', [])
        print(f"Game {i+1}: {len(game['local_images'])} images, scores: {scores}")