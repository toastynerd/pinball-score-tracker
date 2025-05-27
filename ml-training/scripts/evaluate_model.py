#!/usr/bin/env python3
"""
Model evaluation and validation scripts for pinball score OCR
"""

import os
import json
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from dataclasses import dataclass, asdict
import re
from difflib import SequenceMatcher

try:
    from paddleocr import PaddleOCR
except ImportError:
    print("PaddleOCR not installed. Install with: pip install paddleocr")
    exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""

    exact_match_accuracy: float
    character_accuracy: float
    word_accuracy: float
    edit_distance_avg: float
    confidence_avg: float
    total_samples: int
    correct_exact_matches: int
    total_characters: int
    correct_characters: int


class PinballOCREvaluator:
    """Evaluates trained PaddleOCR model on pinball score data"""

    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = True):
        """
        Initialize evaluator

        Args:
            model_path: Path to trained model (None for pretrained)
            use_gpu: Whether to use GPU for inference
        """
        self.model_path = model_path
        self.use_gpu = use_gpu

        # Initialize OCR
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading custom model from {model_path}")
            self.ocr = PaddleOCR(
                det_model_dir=os.path.join(model_path, "det"),
                rec_model_dir=os.path.join(model_path, "rec"),
                cls_model_dir=os.path.join(model_path, "cls"),
                use_angle_cls=True,
                lang="en",
                use_gpu=use_gpu,
            )
        else:
            logger.info("Using pretrained PaddleOCR model")
            self.ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=use_gpu)

    def normalize_score(self, text: str) -> str:
        """Normalize score text for comparison"""
        # Remove non-numeric characters except spaces
        text = re.sub(r"[^\d\s]", "", text)
        # Split by whitespace and rejoin
        numbers = text.split()
        return " ".join(numbers)

    def calculate_edit_distance(self, pred: str, true: str) -> int:
        """Calculate Levenshtein distance between predictions and ground truth"""
        return (
            len(pred)
            + len(true)
            - 2 * len(SequenceMatcher(None, pred, true).get_matching_blocks())
        )

    def evaluate_single_image(self, image_path: str, ground_truth: str) -> Dict:
        """
        Evaluate model on a single image

        Args:
            image_path: Path to test image
            ground_truth: Expected text output

        Returns:
            Dictionary with evaluation results
        """
        try:
            # Run OCR
            result = self.ocr.ocr(image_path, cls=True)

            if not result or not result[0]:
                return {
                    "predicted_text": "",
                    "ground_truth": ground_truth,
                    "exact_match": False,
                    "character_accuracy": 0.0,
                    "edit_distance": len(ground_truth),
                    "confidence": 0.0,
                    "error": "No text detected",
                }

            # Extract text and confidence
            predicted_texts = []
            confidences = []

            for line in result[0]:
                text = line[1][0]
                confidence = line[1][1]
                predicted_texts.append(text)
                confidences.append(confidence)

            predicted_text = " ".join(predicted_texts)
            avg_confidence = np.mean(confidences) if confidences else 0.0

            # Normalize for comparison
            pred_normalized = self.normalize_score(predicted_text)
            true_normalized = self.normalize_score(ground_truth)

            # Calculate metrics
            exact_match = pred_normalized == true_normalized
            edit_distance = self.calculate_edit_distance(
                pred_normalized, true_normalized
            )

            # Character-level accuracy
            char_accuracy = 0.0
            if true_normalized:
                correct_chars = sum(
                    1 for p, t in zip(pred_normalized, true_normalized) if p == t
                )
                char_accuracy = correct_chars / len(true_normalized)

            return {
                "predicted_text": predicted_text,
                "predicted_normalized": pred_normalized,
                "ground_truth": ground_truth,
                "ground_truth_normalized": true_normalized,
                "exact_match": exact_match,
                "character_accuracy": char_accuracy,
                "edit_distance": edit_distance,
                "confidence": avg_confidence,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {
                "predicted_text": "",
                "ground_truth": ground_truth,
                "exact_match": False,
                "character_accuracy": 0.0,
                "edit_distance": len(ground_truth),
                "confidence": 0.0,
                "error": str(e),
            }

    def evaluate_dataset(
        self, test_data_path: str, metadata_file: str
    ) -> EvaluationMetrics:
        """
        Evaluate model on entire test dataset

        Args:
            test_data_path: Path to test images directory
            metadata_file: Path to metadata file with ground truth

        Returns:
            EvaluationMetrics object with aggregate results
        """
        # Load metadata
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        results = []
        test_path = Path(test_data_path)

        logger.info(f"Evaluating on {len(metadata)} samples...")

        for i, (image_name, data) in enumerate(metadata.items()):
            image_path = test_path / image_name

            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue

            ground_truth = " ".join(map(str, data.get("scores", [])))
            result = self.evaluate_single_image(str(image_path), ground_truth)
            results.append(result)

            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(metadata)} images...")

        # Calculate aggregate metrics
        total_samples = len(results)
        exact_matches = sum(1 for r in results if r["exact_match"])
        total_char_accuracy = sum(r["character_accuracy"] for r in results)
        total_edit_distance = sum(r["edit_distance"] for r in results)
        total_confidence = sum(r["confidence"] for r in results)

        # Character-level metrics
        total_characters = sum(len(r["ground_truth_normalized"]) for r in results)
        correct_characters = sum(
            len(r["ground_truth_normalized"]) * r["character_accuracy"] for r in results
        )

        metrics = EvaluationMetrics(
            exact_match_accuracy=(
                exact_matches / total_samples if total_samples > 0 else 0.0
            ),
            character_accuracy=(
                correct_characters / total_characters if total_characters > 0 else 0.0
            ),
            word_accuracy=exact_matches / total_samples if total_samples > 0 else 0.0,
            edit_distance_avg=(
                total_edit_distance / total_samples if total_samples > 0 else 0.0
            ),
            confidence_avg=(
                total_confidence / total_samples if total_samples > 0 else 0.0
            ),
            total_samples=total_samples,
            correct_exact_matches=exact_matches,
            total_characters=int(total_characters),
            correct_characters=int(correct_characters),
        )

        return metrics, results

    def generate_evaluation_report(
        self, metrics: EvaluationMetrics, results: List[Dict], output_path: str
    ):
        """Generate detailed evaluation report"""

        report = {
            "summary": asdict(metrics),
            "detailed_results": results,
            "error_analysis": self._analyze_errors(results),
            "confidence_distribution": self._analyze_confidence(results),
        }

        # Save detailed JSON report
        json_path = Path(output_path) / "evaluation_report.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)

        # Generate human-readable summary
        summary_path = Path(output_path) / "evaluation_summary.txt"
        with open(summary_path, "w") as f:
            f.write("PINBALL OCR MODEL EVALUATION REPORT\n")
            f.write("=" * 40 + "\n\n")

            f.write(f"Total Samples: {metrics.total_samples}\n")
            f.write(f"Exact Match Accuracy: {metrics.exact_match_accuracy:.2%}\n")
            f.write(f"Character Accuracy: {metrics.character_accuracy:.2%}\n")
            f.write(f"Average Edit Distance: {metrics.edit_distance_avg:.2f}\n")
            f.write(f"Average Confidence: {metrics.confidence_avg:.2%}\n\n")

            f.write("ERROR ANALYSIS:\n")
            f.write("-" * 15 + "\n")
            error_analysis = self._analyze_errors(results)
            f.write(f"Samples with errors: {error_analysis['error_count']}\n")
            f.write(f"No text detected: {error_analysis['no_detection_count']}\n")

            f.write("\nCONFIDENCE DISTRIBUTION:\n")
            f.write("-" * 25 + "\n")
            conf_analysis = self._analyze_confidence(results)
            for range_name, count in conf_analysis.items():
                f.write(f"{range_name}: {count}\n")

        logger.info(f"Evaluation report saved to {output_path}")

    def _analyze_errors(self, results: List[Dict]) -> Dict:
        """Analyze common error patterns"""
        error_count = sum(1 for r in results if r["error"])
        no_detection_count = sum(1 for r in results if not r["predicted_text"])

        return {"error_count": error_count, "no_detection_count": no_detection_count}

    def _analyze_confidence(self, results: List[Dict]) -> Dict:
        """Analyze confidence score distribution"""
        ranges = {
            "high_confidence (>0.9)": 0,
            "medium_confidence (0.7-0.9)": 0,
            "low_confidence (<0.7)": 0,
        }

        for r in results:
            conf = r["confidence"]
            if conf > 0.9:
                ranges["high_confidence (>0.9)"] += 1
            elif conf > 0.7:
                ranges["medium_confidence (0.7-0.9)"] += 1
            else:
                ranges["low_confidence (<0.7)"] += 1

        return ranges


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PaddleOCR model on pinball score data"
    )
    parser.add_argument("test_data", help="Path to test images directory")
    parser.add_argument("metadata", help="Path to metadata JSON file with ground truth")
    parser.add_argument("--model-path", help="Path to trained model directory")
    parser.add_argument(
        "--output", default="evaluation_results", help="Output directory for results"
    )
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize evaluator
    evaluator = PinballOCREvaluator(model_path=args.model_path, use_gpu=not args.no_gpu)

    # Run evaluation
    logger.info("Starting model evaluation...")
    metrics, results = evaluator.evaluate_dataset(args.test_data, args.metadata)

    # Generate report
    evaluator.generate_evaluation_report(metrics, results, str(output_path))

    # Print summary
    print("\nEVALUATION SUMMARY:")
    print("=" * 20)
    print(f"Exact Match Accuracy: {metrics.exact_match_accuracy:.2%}")
    print(f"Character Accuracy: {metrics.character_accuracy:.2%}")
    print(f"Average Confidence: {metrics.confidence_avg:.2%}")
    print(f"Total Samples: {metrics.total_samples}")


if __name__ == "__main__":
    main()
