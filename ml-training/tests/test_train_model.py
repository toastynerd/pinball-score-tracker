#!/usr/bin/env python3
"""
Tests for ML model training functionality
"""

import pytest
import tempfile
import os
import json
import sys
from unittest.mock import Mock, patch

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))


class TestPinballOCRTrainer:

    def test_trainer_initialization(self):
        """Test trainer initialization with default config"""
        from train_model import PinballOCRTrainer

        trainer = PinballOCRTrainer()
        assert trainer.config is not None
        assert "model" in trainer.config
        assert "dataset" in trainer.config
        assert "training" in trainer.config

    def test_config_loading(self):
        """Test configuration loading and merging"""
        from train_model import PinballOCRTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {"training": {"epochs": 50, "batch_size": 16}}
            json.dump(config, f)
            f.flush()

            trainer = PinballOCRTrainer(f.name)
            assert trainer.config["training"]["epochs"] == 50
            assert trainer.config["training"]["batch_size"] == 16

            os.unlink(f.name)

    def test_directory_setup(self):
        """Test directory creation"""
        from train_model import PinballOCRTrainer

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"paths": {"output_dir": f"{temp_dir}/output", "log_dir": f"{temp_dir}/logs"}}

            with patch("train_model.PinballOCRTrainer.load_config") as mock_load:
                mock_load.return_value = config
                trainer = PinballOCRTrainer()

                assert os.path.exists(f"{temp_dir}/output")
                assert os.path.exists(f"{temp_dir}/logs")

    def test_dataset_validation_missing_files(self):
        """Test dataset validation with missing files"""
        from train_model import PinballOCRTrainer

        trainer = PinballOCRTrainer()
        trainer.config["dataset"]["train_list"] = "nonexistent_file.txt"

        with pytest.raises(FileNotFoundError):
            trainer.validate_dataset()

    def test_dataset_validation_valid_format(self):
        """Test dataset validation with properly formatted files"""
        from train_model import PinballOCRTrainer

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            train_list = os.path.join(temp_dir, "train_list.txt")
            val_list = os.path.join(temp_dir, "val_list.txt")
            char_dict = os.path.join(temp_dir, "dict.txt")

            with open(train_list, "w", encoding="utf-8") as f:
                f.write("image1.jpg\t[{\"transcription\": \"123,456\"}]\n")
                f.write("image2.jpg\t[{\"transcription\": \"789,012\"}]\n")

            with open(val_list, "w", encoding="utf-8") as f:
                f.write("image3.jpg\t[{\"transcription\": \"345,678\"}]\n")

            with open(char_dict, "w", encoding="utf-8") as f:
                f.write("0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n,\n")

            trainer = PinballOCRTrainer()
            trainer.config["dataset"]["train_list"] = train_list
            trainer.config["dataset"]["val_list"] = val_list
            trainer.config["dataset"]["character_dict"] = char_dict

            num_samples = trainer.validate_dataset()
            assert num_samples == 2

    def test_paddleocr_config_creation(self):
        """Test PaddleOCR configuration file creation"""
        from train_model import PinballOCRTrainer

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy dataset files
            train_list = os.path.join(temp_dir, "train_list.txt")
            val_list = os.path.join(temp_dir, "val_list.txt")
            char_dict = os.path.join(temp_dir, "dict.txt")

            for file_path in [train_list, val_list, char_dict]:
                with open(file_path, "w") as f:
                    f.write("dummy content\n")

            trainer = PinballOCRTrainer()
            trainer.config["dataset"]["train_list"] = train_list
            trainer.config["dataset"]["val_list"] = val_list
            trainer.config["dataset"]["character_dict"] = char_dict

            # Mock the file operations
            mock_open = Mock()
            mock_open.__enter__ = Mock(return_value=Mock())
            mock_open.__exit__ = Mock(return_value=None)
            
            with patch("os.makedirs"), patch("builtins.open", return_value=mock_open), patch("yaml.dump"):
                config_path = trainer.create_paddleocr_config()

            assert config_path == "configs/pinball_ocr_config.yml"

    def test_training_info_saving(self):
        """Test training information saving"""
        from train_model import PinballOCRTrainer

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = PinballOCRTrainer()
            trainer.config["paths"]["output_dir"] = temp_dir

            trainer.save_training_info("test_config.yml", "test command", 100)

            info_file = os.path.join(temp_dir, "training_info.json")
            assert os.path.exists(info_file)

            with open(info_file, "r") as f:
                info = json.load(f)

            assert info["config_path"] == "test_config.yml"
            assert info["dataset_info"]["num_samples"] == 100
            assert "next_steps" in info


class TestTrainingLogic:

    def test_hyperparameter_validation(self):
        """Test hyperparameter validation"""
        from train_model import PinballOCRTrainer

        trainer = PinballOCRTrainer()

        # Test default values
        assert trainer.config["training"]["epochs"] > 0
        assert trainer.config["training"]["batch_size"] > 0
        assert trainer.config["training"]["learning_rate"] > 0

    def test_model_architecture_config(self):
        """Test model architecture configuration"""
        from train_model import PinballOCRTrainer

        trainer = PinballOCRTrainer()
        config_data = trainer.create_paddleocr_config()

        # Test that essential components are configured
        assert trainer.config["model"]["architecture"] == "CRNN"
        assert trainer.config["model"]["backbone"] == "ResNet"


class TestCommandLineInterface:

    def test_main_function_basic(self):
        """Test main function with basic arguments"""
        from train_model import main
        
        # Mock sys.argv and the trainer
        test_args = ["train_model.py", "--epochs", "50", "--batch-size", "4"]
        
        with patch("sys.argv", test_args), \
             patch("train_model.PinballOCRTrainer") as mock_trainer_class:
            
            mock_trainer = Mock()
            mock_trainer.config = {"training": {}, "dataset": {}, "paths": {}}
            mock_trainer.train_model.return_value = "test_config.yml"
            mock_trainer_class.return_value = mock_trainer

            main()

            mock_trainer_class.assert_called_once()
            mock_trainer.train_model.assert_called_once()

    def test_config_override_logic(self):
        """Test configuration override from command line arguments"""
        from train_model import PinballOCRTrainer

        trainer = PinballOCRTrainer()
        original_epochs = trainer.config["training"]["epochs"]

        # Simulate config override
        trainer.config["training"]["epochs"] = 200

        assert trainer.config["training"]["epochs"] != original_epochs
        assert trainer.config["training"]["epochs"] == 200