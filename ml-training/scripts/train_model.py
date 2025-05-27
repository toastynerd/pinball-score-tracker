#!/usr/bin/env python3
"""
PaddleOCR training script for pinball score recognition
"""

import os
import json
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PinballOCRTrainer:
    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)
        self.setup_directories()

    def load_config(self, config_path):
        """Load training configuration"""
        default_config = {
            "model": {
                "architecture": "CRNN",
                "backbone": "ResNet",
                "neck": "RNN",
                "head": "CTCHead",
            },
            "dataset": {
                "train_data_dir": "data/paddleocr_training",
                "train_list": "data/paddleocr_training/train_list.txt",
                "val_list": "data/paddleocr_training/val_list.txt",
                "character_dict": "data/paddleocr_training/dict.txt",
            },
            "training": {
                "batch_size": 8,
                "learning_rate": 0.001,
                "epochs": 100,
                "save_epoch_step": 10,
                "eval_batch_step": [0, 100],
            },
            "optimizer": {"type": "Adam", "lr": 0.001, "weight_decay": 0.0001},
            "paths": {
                "pretrained_model": None,
                "output_dir": "models/pinball_ocr",
                "log_dir": "logs",
            },
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                user_config = json.load(f)
                # Merge configs
                default_config.update(user_config)

        return default_config

    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            self.config["paths"]["output_dir"],
            self.config["paths"]["log_dir"],
            "models/checkpoints",
        ]

        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")

    def validate_dataset(self):
        """Validate that dataset files exist and are properly formatted"""
        required_files = [
            self.config["dataset"]["train_list"],
            self.config["dataset"]["val_list"],
            self.config["dataset"]["character_dict"],
        ]

        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required dataset file not found: {file_path}")

        # Validate train list format
        train_list = self.config["dataset"]["train_list"]
        with open(train_list, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if not lines:
                raise ValueError(f"Training list is empty: {train_list}")

            # Check first line format
            first_line = lines[0].strip()
            if "\t" not in first_line:
                raise ValueError(
                    f"Invalid format in training list. Expected tab-separated values."
                )

        logger.info(f"Dataset validation passed. Found {len(lines)} training samples.")
        return len(lines)

    def create_paddleocr_config(self):
        """Create PaddleOCR configuration file"""
        config = {
            "Global": {
                "debug": False,
                "use_gpu": True,
                "epoch_num": self.config["training"]["epochs"],
                "log_smooth_window": 20,
                "print_batch_step": 10,
                "save_model_dir": self.config["paths"]["output_dir"],
                "save_epoch_step": self.config["training"]["save_epoch_step"],
                "eval_batch_step": self.config["training"]["eval_batch_step"],
                "cal_metric_during_train": True,
                "checkpoints": None,
                "pretrained_model": self.config["paths"]["pretrained_model"],
                "save_inference_dir": None,
                "use_visualdl": False,
                "infer_img": None,
                "character_dict_path": self.config["dataset"]["character_dict"],
                "max_text_length": 25,
                "infer_mode": False,
                "use_space_char": False,
                "distributed": False,
            },
            "Architecture": {
                "model_type": "rec",
                "algorithm": "CRNN",
                "Transform": None,
                "Backbone": {"name": "ResNet", "layers": 34},
                "Neck": {
                    "name": "SequenceEncoder",
                    "encoder_type": "rnn",
                    "hidden_size": 256,
                },
                "Head": {"name": "CTCHead", "fc_decay": 0.0001},
            },
            "Loss": {"name": "CTCLoss"},
            "Optimizer": {
                "name": "Adam",
                "beta1": 0.9,
                "beta2": 0.999,
                "lr": {
                    "name": "Cosine",
                    "learning_rate": self.config["training"]["learning_rate"],
                    "warmup_epoch": 2,
                },
                "regularizer": {
                    "name": "L2",
                    "factor": self.config["optimizer"]["weight_decay"],
                },
            },
            "PostProcess": {"name": "CTCLabelDecode"},
            "Metric": {"name": "RecMetric", "main_indicator": "acc"},
            "Train": {
                "dataset": {
                    "name": "SimpleDataSet",
                    "data_dir": self.config["dataset"]["train_data_dir"],
                    "label_file_list": [self.config["dataset"]["train_list"]],
                    "transforms": [
                        {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                        {"CTCLabelEncode": None},
                        {"RecResizeImg": {"image_shape": [3, 32, 320]}},
                        {"KeepKeys": {"keep_keys": ["image", "label", "length"]}},
                    ],
                },
                "loader": {
                    "shuffle": True,
                    "batch_size_per_card": self.config["training"]["batch_size"],
                    "drop_last": True,
                    "num_workers": 4,
                },
            },
            "Eval": {
                "dataset": {
                    "name": "SimpleDataSet",
                    "data_dir": self.config["dataset"]["train_data_dir"],
                    "label_file_list": [self.config["dataset"]["val_list"]],
                    "transforms": [
                        {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                        {"CTCLabelEncode": None},
                        {"RecResizeImg": {"image_shape": [3, 32, 320]}},
                        {"KeepKeys": {"keep_keys": ["image", "label", "length"]}},
                    ],
                },
                "loader": {
                    "shuffle": False,
                    "drop_last": False,
                    "batch_size_per_card": self.config["training"]["batch_size"],
                    "num_workers": 4,
                },
            },
        }

        config_path = "configs/pinball_ocr_config.yml"
        os.makedirs("configs", exist_ok=True)

        # Convert to YAML format for PaddleOCR
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"Created PaddleOCR config: {config_path}")
        return config_path

    def train_model(self):
        """Train the PaddleOCR model"""
        try:
            # Validate dataset first
            num_samples = self.validate_dataset()
            logger.info(f"Starting training with {num_samples} samples")

            # Create PaddleOCR config
            config_path = self.create_paddleocr_config()

            # Import PaddleOCR training modules
            try:
                import paddle
                from paddleocr import PaddleOCR

                logger.info(f"Using PaddlePaddle version: {paddle.__version__}")
            except ImportError:
                logger.error(
                    "PaddleOCR not installed. Install with: pip install paddlepaddle paddleocr"
                )
                raise

            # Training command for PaddleOCR
            training_cmd = f"""
            python -m paddle.distributed.launch \\
                --gpus="0" \\
                tools/train.py \\
                -c {config_path}
            """

            logger.info("Training command:")
            logger.info(training_cmd)

            # For now, save the training configuration and provide instructions
            self.save_training_info(config_path, training_cmd, num_samples)

            return config_path

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def save_training_info(self, config_path, training_cmd, num_samples):
        """Save training information for manual execution"""
        info = {
            "config_path": config_path,
            "training_command": training_cmd,
            "dataset_info": {
                "num_samples": num_samples,
                "train_list": self.config["dataset"]["train_list"],
                "val_list": self.config["dataset"]["val_list"],
                "character_dict": self.config["dataset"]["character_dict"],
            },
            "model_config": self.config["model"],
            "training_config": self.config["training"],
            "next_steps": [
                "1. Install PaddleOCR: pip install paddlepaddle paddleocr",
                "2. Clone PaddleOCR repo: git clone https://github.com/PaddlePaddle/PaddleOCR.git",
                "3. Copy config file to PaddleOCR/configs/",
                "4. Run training command from PaddleOCR directory",
                "5. Monitor training logs and adjust hyperparameters as needed",
            ],
        }

        info_path = os.path.join(
            self.config["paths"]["output_dir"], "training_info.json"
        )
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

        logger.info(f"Training configuration saved to: {info_path}")
        logger.info("To start training:")
        logger.info("1. Install PaddleOCR dependencies")
        logger.info("2. Use the generated config file with PaddleOCR training tools")
        logger.info(
            f"3. Monitor training progress in: {self.config['paths']['log_dir']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Train PaddleOCR model for pinball score recognition"
    )
    parser.add_argument(
        "--config", type=str, help="Path to training configuration file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/paddleocr_training",
        help="Directory containing training data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/pinball_ocr",
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )

    args = parser.parse_args()

    # Override config with command line arguments
    config_overrides = {}
    if args.data_dir:
        config_overrides["dataset"] = {"train_data_dir": args.data_dir}
    if args.output_dir:
        config_overrides["paths"] = {"output_dir": args.output_dir}
    if args.epochs:
        config_overrides["training"] = {"epochs": args.epochs}
    if args.batch_size:
        config_overrides["training"] = config_overrides.get("training", {})
        config_overrides["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config_overrides["training"] = config_overrides.get("training", {})
        config_overrides["training"]["learning_rate"] = args.learning_rate

    # Initialize trainer
    trainer = PinballOCRTrainer(args.config)

    # Apply overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if key in trainer.config:
                trainer.config[key].update(value)
            else:
                trainer.config[key] = value

    # Start training
    logger.info("Starting PaddleOCR training for pinball score recognition")
    config_path = trainer.train_model()
    logger.info(f"Training setup complete. Configuration saved at: {config_path}")


if __name__ == "__main__":
    main()
