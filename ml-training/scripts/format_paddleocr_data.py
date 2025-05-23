#!/usr/bin/env python3
"""
Format scraped pinball data for PaddleOCR training
Creates training annotations with images and ground truth text
"""

import json
import logging
import os
import shutil

from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaddleOCRFormatter:
    def __init__(self, input_dataset_dir, output_dir="data/paddleocr_training"):
        self.input_dir = input_dataset_dir
        self.output_dir = output_dir
        self.setup_directories()

    def setup_directories(self):
        """Create PaddleOCR training directory structure"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/images", exist_ok=True)
        os.makedirs(f"{self.output_dir}/labels", exist_ok=True)

    def format_scores_as_text(self, scores):
        """Convert score array to readable text format"""
        # Format scores with commas for readability
        formatted_scores = []
        for score in scores:
            if score > 0:
                # Add commas to large numbers
                formatted = f"{score:,}"
                formatted_scores.append(formatted)

        return formatted_scores

    def create_paddleocr_annotation(self, image_path, scores, output_name):
        """
        Create PaddleOCR format annotation
        Format: image_path\t[{"transcription": "text", "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]}]
        """
        # For now, we'll create a simple text-only annotation
        # In a real scenario, you'd need bounding boxes around each score
        formatted_scores = self.format_scores_as_text(scores)

        # Create annotation for each score
        annotations = []
        for i, score_text in enumerate(formatted_scores):
            # Placeholder bounding boxes - would need manual annotation or detection
            # For training, these would be real coordinates of score locations
            annotation = {
                "transcription": score_text,
                "points": [
                    [100 + i * 200, 100],  # Top-left
                    [200 + i * 200, 100],  # Top-right
                    [200 + i * 200, 150],  # Bottom-right
                    [100 + i * 200, 150],  # Bottom-left
                ],
                "difficult": False,
            }
            annotations.append(annotation)

        # PaddleOCR format: image_path\tannotations_json
        annotation_line = f"{output_name}\t{json.dumps(annotations, ensure_ascii=False)}\n"
        return annotation_line

    def create_simple_text_annotation(self, scores):
        """Create simple text file with all scores for basic OCR training"""
        formatted_scores = self.format_scores_as_text(scores)
        return " ".join(formatted_scores)

    def process_dataset(self):
        """Process filtered dataset and create PaddleOCR training format"""
        # Load filtered metadata
        metadata_file = os.path.join(self.input_dir, "filtered_dataset_metadata.json")

        if not os.path.exists(metadata_file):
            logger.error(f"Filtered metadata not found: {metadata_file}")
            return

        with open(metadata_file, "r") as f:
            games_data = json.load(f)

        train_annotations = []
        simple_annotations = []

        for game_idx, game in enumerate(games_data):
            scores = game.get("js_data", {}).get("scores", [])

            for img_idx, img_path in enumerate(game.get("local_images", [])):
                # Copy image to training directory
                src_path = (
                    os.path.join(self.input_dir, img_path)
                    if not os.path.isabs(img_path)
                    else img_path
                )
                if not os.path.exists(src_path):
                    # Try alternative path
                    clean_path = img_path.replace(f"{self.input_dir}/", "").replace(
                        self.input_dir, ""
                    )
                    src_path = os.path.join(self.input_dir, clean_path)

                if not os.path.exists(src_path):
                    logger.warning(f"Source image not found: {src_path}")
                    continue

                # Generate output filename
                output_name = f"game_{game_idx+1}_img_{img_idx+1}.png"
                dst_path = os.path.join(self.output_dir, "images", output_name)

                # Copy image
                shutil.copy2(src_path, dst_path)

                # Create PaddleOCR annotation
                annotation = self.create_paddleocr_annotation(
                    f"images/{output_name}", scores, output_name
                )
                train_annotations.append(annotation)

                # Create simple text annotation
                simple_text = self.create_simple_text_annotation(scores)
                simple_annotations.append(f"{output_name}\t{simple_text}\n")

                logger.info(f"Processed: {output_name} with scores: {scores}")

        # Save training annotations
        train_file = os.path.join(self.output_dir, "train_list.txt")
        with open(train_file, "w", encoding="utf-8") as f:
            f.writelines(train_annotations)

        # Save simple text annotations
        simple_file = os.path.join(self.output_dir, "simple_annotations.txt")
        with open(simple_file, "w", encoding="utf-8") as f:
            f.writelines(simple_annotations)

        # Create validation set (copy of training for now)
        val_file = os.path.join(self.output_dir, "val_list.txt")
        with open(val_file, "w", encoding="utf-8") as f:
            f.writelines(train_annotations)

        # Create label dictionary
        self.create_label_dict()

        logger.info(f"PaddleOCR training data created:")
        logger.info(f"  Images: {len(train_annotations)}")
        logger.info(f"  Training annotations: {train_file}")
        logger.info(f"  Validation annotations: {val_file}")
        logger.info(f"  Simple annotations: {simple_file}")

        return len(train_annotations)

    def create_label_dict(self):
        """Create character dictionary for PaddleOCR"""
        # Characters that appear in pinball scores
        chars = set()
        chars.update("0123456789")  # Digits
        chars.update(",")  # Thousands separator
        chars.update(" ")  # Space

        # Save character dictionary
        dict_file = os.path.join(self.output_dir, "dict.txt")
        with open(dict_file, "w", encoding="utf-8") as f:
            for char in sorted(chars):
                f.write(f"{char}\n")

        logger.info(f"Character dictionary saved: {dict_file}")


if __name__ == "__main__":
    formatter = PaddleOCRFormatter("data/selenium_dataset")
    num_samples = formatter.process_dataset()

    print(f"Created PaddleOCR training dataset with {num_samples} samples")
    print("Training data ready for PaddleOCR fine-tuning!")
