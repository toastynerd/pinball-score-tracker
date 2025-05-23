#!/usr/bin/env python3
"""
Tests for image filtering functionality
"""

import json
import os
import sys
import tempfile
from unittest.mock import Mock, patch

import pytest

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))


class TestImageFiltering:

    def test_is_blank_image_file_size_logic(self):
        """Test file size based filtering logic"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"small")  # Very small file
            f.flush()

            # Mock the is_blank_image function to focus on file size logic
            with patch("filter_valid_images.os.path.getsize") as mock_getsize:
                mock_getsize.return_value = 100  # Small file

                from filter_valid_images import is_blank_image

                result = is_blank_image(f.name)
                assert result == True

            os.unlink(f.name)

    def test_filter_dataset_logic(self):
        """Test dataset filtering logic"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create filtered metadata file
            metadata = [
                {
                    "url": "test1",
                    "local_images": ["images/test.png"],
                    "js_data": {"scores": [100, 200]},
                }
            ]

            metadata_file = os.path.join(temp_dir, "dataset_metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f)

            # Create dummy image
            images_dir = os.path.join(temp_dir, "images")
            os.makedirs(images_dir)
            img_path = os.path.join(images_dir, "test.png")
            with open(img_path, "wb") as f:
                f.write(b"0" * 5000)  # Large enough file

            # Mock is_blank_image to return False (not blank)
            with patch("filter_valid_images.is_blank_image") as mock_is_blank:
                mock_is_blank.return_value = False

                from filter_valid_images import filter_dataset

                result = filter_dataset(temp_dir)

                assert len(result) == 1
                assert result[0]["js_data"]["scores"] == [100, 200]
