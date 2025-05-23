#!/usr/bin/env python3
"""
Tests for image filtering functionality
"""

import pytest
import tempfile
import os
import json
from PIL import Image
import numpy as np
import sys

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from filter_valid_images import is_blank_image, filter_dataset

class TestImageFiltering:
    
    def create_test_image(self, width=100, height=100, color=(255, 255, 255), save_path=None):
        """Create a test image"""
        img = Image.new('RGB', (width, height), color)
        if save_path:
            img.save(save_path)
        return img
    
    def create_blank_image(self, save_path):
        """Create a blank/empty image"""
        return self.create_test_image(10, 10, (0, 0, 0), save_path)
    
    def create_complex_image(self, save_path):
        """Create an image with variation (not blank)"""
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(save_path)
        return img
    
    def test_is_blank_image_with_small_file(self):
        """Test detection of small/empty files"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            # Create tiny file
            f.write(b'small')
            f.flush()
            
            result = is_blank_image(f.name)
            assert result == True
            
            os.unlink(f.name)
    
    def test_is_blank_image_with_uniform_color(self):
        """Test detection of uniform color images"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            self.create_blank_image(f.name)
            
            result = is_blank_image(f.name)
            assert result == True
            
            os.unlink(f.name)
    
    def test_is_blank_image_with_complex_image(self):
        """Test that complex images are not marked as blank"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            self.create_complex_image(f.name)
            
            result = is_blank_image(f.name)
            assert result == False
            
            os.unlink(f.name)
    
    def test_filter_dataset(self):
        """Test full dataset filtering"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test dataset structure
            images_dir = os.path.join(temp_dir, 'images')
            os.makedirs(images_dir)
            
            # Create test images
            blank_img = os.path.join(images_dir, 'blank.png')
            complex_img = os.path.join(images_dir, 'complex.png')
            
            self.create_blank_image(blank_img)
            self.create_complex_image(complex_img)
            
            # Create test metadata
            metadata = [
                {
                    'url': 'test1',
                    'local_images': ['images/blank.png'],
                    'js_data': {'scores': [100, 200]}
                },
                {
                    'url': 'test2', 
                    'local_images': ['images/complex.png'],
                    'js_data': {'scores': [300, 400]}
                }
            ]
            
            metadata_file = os.path.join(temp_dir, 'dataset_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            
            # Test filtering
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
            from filter_valid_images import filter_dataset
            
            # Mock the function to use our temp directory
            import filter_valid_images
            original_func = filter_valid_images.filter_dataset
            
            def mock_filter(dataset_dir):
                return original_func(temp_dir)
            
            result = mock_filter(temp_dir)
            
            # Should only keep the complex image
            assert len(result) == 1
            assert 'complex.png' in result[0]['local_images'][0]