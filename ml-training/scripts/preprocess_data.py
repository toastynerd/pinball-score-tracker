#!/usr/bin/env python3
"""
Data preprocessing pipeline for pinball score pictures
Prepares images for optimal PaddleOCR training
"""

import cv2
import numpy as np
import os
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PinballImagePreprocessor:
    """Preprocesses pinball score images for optimal OCR training"""

    def __init__(self, target_height: int = 32, target_width: Optional[int] = None):
        """
        Initialize preprocessor

        Args:
            target_height: Standard height for OCR training (PaddleOCR uses 32)
            target_width: Target width (None for maintaining aspect ratio)
        """
        self.target_height = target_height
        self.target_width = target_width

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE for contrast enhancement"""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)

            # Merge channels and convert back
            enhanced = cv2.merge([l, a, b])
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)

    def normalize_size(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target dimensions while maintaining aspect ratio"""
        h, w = image.shape[:2]

        if self.target_width:
            # Fixed width and height
            return cv2.resize(image, (self.target_width, self.target_height))
        else:
            # Maintain aspect ratio, fix height
            aspect_ratio = w / h
            new_width = int(self.target_height * aspect_ratio)
            return cv2.resize(image, (new_width, self.target_height))

    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Apply noise reduction"""
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

    def sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """Apply sharpening filter for better text clarity"""
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)

    def preprocess_single_image(
        self,
        image_path: str,
        apply_denoising: bool = True,
        apply_sharpening: bool = True,
    ) -> np.ndarray:
        """
        Preprocess a single image

        Args:
            image_path: Path to input image
            apply_denoising: Whether to apply noise reduction
            apply_sharpening: Whether to apply sharpening

        Returns:
            Preprocessed image as numpy array
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Apply preprocessing steps
        logger.debug(f"Processing {image_path}")

        # 1. Enhance contrast
        image = self.enhance_contrast(image)

        # 2. Denoise if requested
        if apply_denoising:
            image = self.denoise_image(image)

        # 3. Sharpen if requested
        if apply_sharpening:
            image = self.sharpen_image(image)

        # 4. Normalize size
        image = self.normalize_size(image)

        return image

    def preprocess_dataset(
        self, input_dir: str, output_dir: str, metadata_file: Optional[str] = None
    ) -> Dict:
        """
        Preprocess entire dataset

        Args:
            input_dir: Directory containing input images
            output_dir: Directory for preprocessed images
            metadata_file: Optional metadata file to filter images

        Returns:
            Dictionary with preprocessing statistics
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load metadata if provided
        metadata = {}
        if metadata_file and os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

        # Get image files
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_files = [
            f for f in input_path.iterdir() if f.suffix.lower() in image_extensions
        ]

        stats = {
            "total_images": len(image_files),
            "processed_successfully": 0,
            "failed_images": [],
            "preprocessing_params": {
                "target_height": self.target_height,
                "target_width": self.target_width,
            },
        }

        logger.info(f"Processing {len(image_files)} images...")

        for image_file in image_files:
            try:
                # Check if image has valid metadata (if metadata provided)
                if metadata and image_file.name not in metadata:
                    logger.debug(f"Skipping {image_file.name} - no metadata")
                    continue

                # Preprocess image
                processed_image = self.preprocess_single_image(str(image_file))

                # Save processed image
                output_file = output_path / image_file.name
                cv2.imwrite(str(output_file), processed_image)

                stats["processed_successfully"] += 1

                if stats["processed_successfully"] % 100 == 0:
                    logger.info(
                        f"Processed {stats['processed_successfully']} images..."
                    )

            except Exception as e:
                logger.error(f"Failed to process {image_file.name}: {e}")
                stats["failed_images"].append(
                    {"filename": image_file.name, "error": str(e)}
                )

        logger.info(
            f"Preprocessing complete: {stats['processed_successfully']}/{stats['total_images']} successful"
        )

        # Save preprocessing statistics
        stats_file = output_path / "preprocessing_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess pinball score images for OCR training"
    )
    parser.add_argument("input_dir", help="Input directory containing images")
    parser.add_argument("output_dir", help="Output directory for preprocessed images")
    parser.add_argument("--metadata", help="Path to metadata JSON file")
    parser.add_argument(
        "--height", type=int, default=32, help="Target height (default: 32)"
    )
    parser.add_argument(
        "--width", type=int, help="Target width (maintains aspect ratio if not set)"
    )
    parser.add_argument("--no-denoise", action="store_true", help="Skip denoising step")
    parser.add_argument(
        "--no-sharpen", action="store_true", help="Skip sharpening step"
    )

    args = parser.parse_args()

    # Initialize preprocessor
    preprocessor = PinballImagePreprocessor(
        target_height=args.height, target_width=args.width
    )

    # Process dataset
    stats = preprocessor.preprocess_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        metadata_file=args.metadata,
    )

    print(f"Preprocessing complete!")
    print(f"Successfully processed: {stats['processed_successfully']}")
    print(f"Failed images: {len(stats['failed_images'])}")

    if stats["failed_images"]:
        print("Failed images:")
        for failed in stats["failed_images"]:
            print(f"  - {failed['filename']}: {failed['error']}")


if __name__ == "__main__":
    main()
