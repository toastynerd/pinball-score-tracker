#!/usr/bin/env python3
"""
Tests for PaddleOCR data formatting
"""

import json
import os
import sys
import tempfile

import pytest

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
from format_paddleocr_data import PaddleOCRFormatter


class TestPaddleOCRFormatter:

    def test_format_scores_as_text(self):
        """Test score formatting with commas"""
        formatter = PaddleOCRFormatter("dummy", "dummy")

        scores = [1000000, 500000, 0, 250000]
        result = formatter.format_scores_as_text(scores)

        expected = ["1,000,000", "500,000", "250,000"]
        assert result == expected

    def test_create_simple_text_annotation(self):
        """Test simple text annotation creation"""
        formatter = PaddleOCRFormatter("dummy", "dummy")

        scores = [1000, 2000, 3000]
        result = formatter.create_simple_text_annotation(scores)

        assert result == "1,000 2,000 3,000"

    def test_create_paddleocr_annotation(self):
        """Test PaddleOCR annotation format"""
        formatter = PaddleOCRFormatter("dummy", "dummy")

        scores = [100000, 200000]
        result = formatter.create_paddleocr_annotation("test.png", scores, "test.png")

        # Should contain image path and JSON annotations
        assert "test.png\t" in result
        assert "100,000" in result
        assert "200,000" in result
        assert "points" in result
        assert "transcription" in result

    def test_create_label_dict(self):
        """Test character dictionary creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            formatter = PaddleOCRFormatter("dummy", temp_dir)
            formatter.create_label_dict()

            dict_file = os.path.join(temp_dir, "dict.txt")
            assert os.path.exists(dict_file)

            with open(dict_file, "r") as f:
                content = f.read()
                assert "0" in content
                assert "9" in content
                assert "," in content

    def test_setup_directories(self):
        """Test directory structure creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "output")
            formatter = PaddleOCRFormatter("dummy", output_dir)

            assert os.path.exists(output_dir)
            assert os.path.exists(os.path.join(output_dir, "images"))
            assert os.path.exists(os.path.join(output_dir, "labels"))

    def test_process_dataset_integration(self):
        """Test full dataset processing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create input structure
            input_dir = os.path.join(temp_dir, "input")
            output_dir = os.path.join(temp_dir, "output")
            images_dir = os.path.join(input_dir, "images")
            os.makedirs(images_dir)

            # Create dummy image file
            img_path = os.path.join(images_dir, "test.png")
            with open(img_path, "wb") as f:
                # Write a minimal PNG file
                png_header = b"\x89PNG\r\n\x1a\n"
                f.write(png_header + b"0" * 100)

            # Create filtered metadata
            metadata = [
                {
                    "local_images": ["images/test.png"],
                    "js_data": {"scores": [100000, 200000]},
                }
            ]

            metadata_file = os.path.join(input_dir, "filtered_dataset_metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f)

            # Process dataset
            formatter = PaddleOCRFormatter(input_dir, output_dir)
            result = formatter.process_dataset()

            # Check outputs
            assert result == 1
            assert os.path.exists(os.path.join(output_dir, "train_list.txt"))
            assert os.path.exists(os.path.join(output_dir, "val_list.txt"))
            assert os.path.exists(os.path.join(output_dir, "simple_annotations.txt"))
            assert os.path.exists(os.path.join(output_dir, "dict.txt"))
            assert os.path.exists(
                os.path.join(output_dir, "images", "game_1_img_1.png")
            )
