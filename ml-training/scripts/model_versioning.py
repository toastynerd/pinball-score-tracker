#!/usr/bin/env python3
"""
Model artifact storage and versioning system for PaddleOCR models
Supports local storage, S3, and DVC-like versioning
"""

import json
import hashlib
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import tarfile
import boto3
from dataclasses import dataclass, asdict
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a trained model version"""
    model_id: str
    version: str
    name: str
    description: str
    created_at: str
    training_config: Dict
    dataset_info: Dict
    metrics: Dict
    model_size_mb: float
    model_hash: str
    files: List[str]
    tags: List[str]


class ModelVersioningSystem:
    """Manages model versions with local and S3 storage support"""
    
    def __init__(self, 
                 base_dir: str = "models",
                 s3_bucket: Optional[str] = None,
                 s3_prefix: str = "model-registry"):
        """
        Initialize model versioning system
        
        Args:
            base_dir: Local base directory for model storage
            s3_bucket: Optional S3 bucket for remote storage
            s3_prefix: S3 prefix for model artifacts
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = self.base_dir / "model_registry.json"
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        
        if s3_bucket:
            self.s3_client = boto3.client('s3')
        
        self._load_registry()
    
    def _load_registry(self):
        """Load model registry from JSON file"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                "models": {},
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
    
    def _save_registry(self):
        """Save model registry to JSON file"""
        self.registry["last_updated"] = datetime.now().isoformat()
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
        
        # Also sync to S3 if configured
        if self.s3_bucket:
            try:
                self.s3_client.upload_file(
                    str(self.registry_file),
                    self.s3_bucket,
                    f"{self.s3_prefix}/model_registry.json"
                )
            except Exception as e:
                logger.warning(f"Failed to sync registry to S3: {e}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _calculate_directory_hash(self, dir_path: Path) -> str:
        """Calculate hash of all files in a directory"""
        all_hashes = []
        for file_path in sorted(dir_path.rglob('*')):
            if file_path.is_file():
                file_hash = self._calculate_file_hash(file_path)
                rel_path = file_path.relative_to(dir_path)
                all_hashes.append(f"{rel_path}:{file_hash}")
        
        combined = "\n".join(all_hashes)
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _get_directory_size_mb(self, dir_path: Path) -> float:
        """Get total size of directory in MB"""
        total_size = 0
        for file_path in dir_path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024 * 1024)  # Convert to MB
    
    def _create_model_archive(self, model_dir: Path, archive_path: Path):
        """Create tar.gz archive of model directory"""
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(model_dir, arcname=model_dir.name)
    
    def _extract_model_archive(self, archive_path: Path, extract_dir: Path):
        """Extract model archive to directory"""
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extract_dir)
    
    def register_model(self,
                      model_dir: str,
                      name: str,
                      description: str = "",
                      training_config: Optional[Dict] = None,
                      dataset_info: Optional[Dict] = None,
                      metrics: Optional[Dict] = None,
                      tags: Optional[List[str]] = None,
                      version: Optional[str] = None) -> str:
        """
        Register a new model version
        
        Args:
            model_dir: Path to model directory
            name: Model name
            description: Model description
            training_config: Training configuration used
            dataset_info: Dataset information
            metrics: Evaluation metrics
            tags: Model tags
            version: Specific version (auto-generated if None)
            
        Returns:
            Model ID
        """
        model_path = Path(model_dir)
        if not model_path.exists():
            raise ValueError(f"Model directory does not exist: {model_dir}")
        
        # Generate version if not provided
        if version is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            version = f"v{timestamp}"
        
        # Generate model ID
        model_id = f"{name}_{version}"
        
        # Calculate metadata
        model_hash = self._calculate_directory_hash(model_path)
        model_size = self._get_directory_size_mb(model_path)
        file_list = [str(p.relative_to(model_path)) for p in model_path.rglob('*') if p.is_file()]
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            name=name,
            description=description,
            created_at=datetime.now().isoformat(),
            training_config=training_config or {},
            dataset_info=dataset_info or {},
            metrics=metrics or {},
            model_size_mb=model_size,
            model_hash=model_hash,
            files=file_list,
            tags=tags or []
        )
        
        # Check if model already exists
        if model_id in self.registry["models"]:
            existing_hash = self.registry["models"][model_id]["model_hash"]
            if existing_hash == model_hash:
                logger.info(f"Model {model_id} already exists with same hash. Skipping registration.")
                return model_id
            else:
                logger.warning(f"Model {model_id} exists but with different hash. Creating new version.")
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                version = f"v{timestamp}"
                model_id = f"{name}_{version}"
                metadata.model_id = model_id
                metadata.version = version
        
        # Store model locally
        local_model_dir = self.base_dir / model_id
        if local_model_dir.exists():
            shutil.rmtree(local_model_dir)
        
        shutil.copytree(model_path, local_model_dir)
        
        # Create archive
        archive_path = self.base_dir / f"{model_id}.tar.gz"
        self._create_model_archive(local_model_dir, archive_path)
        
        # Upload to S3 if configured
        if self.s3_bucket:
            s3_key = f"{self.s3_prefix}/{model_id}.tar.gz"
            try:
                logger.info(f"Uploading model to S3: s3://{self.s3_bucket}/{s3_key}")
                self.s3_client.upload_file(
                    str(archive_path),
                    self.s3_bucket,
                    s3_key
                )
                logger.info("Model uploaded to S3 successfully")
            except Exception as e:
                logger.error(f"Failed to upload model to S3: {e}")
        
        # Register in registry
        self.registry["models"][model_id] = asdict(metadata)
        self._save_registry()
        
        logger.info(f"Model registered successfully: {model_id}")
        logger.info(f"  Version: {version}")
        logger.info(f"  Size: {model_size:.2f} MB")
        logger.info(f"  Hash: {model_hash[:16]}...")
        
        return model_id
    
    def list_models(self, name_filter: Optional[str] = None, 
                   tag_filter: Optional[str] = None) -> List[Dict]:
        """
        List registered models
        
        Args:
            name_filter: Filter by model name
            tag_filter: Filter by tag
            
        Returns:
            List of model metadata
        """
        models = []
        for model_id, metadata in self.registry["models"].items():
            # Apply filters
            if name_filter and name_filter.lower() not in metadata["name"].lower():
                continue
            if tag_filter and tag_filter not in metadata["tags"]:
                continue
            
            models.append(metadata)
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x["created_at"], reverse=True)
        return models
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get detailed information about a model"""
        return self.registry["models"].get(model_id)
    
    def download_model(self, model_id: str, output_dir: str, 
                      force: bool = False) -> str:
        """
        Download model to local directory
        
        Args:
            model_id: Model ID to download
            output_dir: Local output directory
            force: Force download even if exists locally
            
        Returns:
            Path to downloaded model
        """
        if model_id not in self.registry["models"]:
            raise ValueError(f"Model not found: {model_id}")
        
        output_path = Path(output_dir) / model_id
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Check if already exists locally
        local_model_dir = self.base_dir / model_id
        if local_model_dir.exists() and not force:
            logger.info(f"Model exists locally, copying to {output_path}")
            if output_path.exists():
                shutil.rmtree(output_path)
            shutil.copytree(local_model_dir, output_path)
            return str(output_path)
        
        # Try to download from S3
        if self.s3_bucket:
            s3_key = f"{self.s3_prefix}/{model_id}.tar.gz"
            archive_path = Path(output_dir) / f"{model_id}.tar.gz"
            
            try:
                logger.info(f"Downloading model from S3: s3://{self.s3_bucket}/{s3_key}")
                self.s3_client.download_file(
                    self.s3_bucket,
                    s3_key,
                    str(archive_path)
                )
                
                # Extract archive
                self._extract_model_archive(archive_path, Path(output_dir))
                archive_path.unlink()  # Remove archive after extraction
                
                logger.info(f"Model downloaded successfully to {output_path}")
                return str(output_path)
                
            except Exception as e:
                logger.error(f"Failed to download from S3: {e}")
        
        # Fall back to local archive
        local_archive = self.base_dir / f"{model_id}.tar.gz"
        if local_archive.exists():
            logger.info(f"Extracting local archive to {output_path}")
            self._extract_model_archive(local_archive, Path(output_dir))
            return str(output_path)
        
        raise ValueError(f"Model {model_id} not found locally or in S3")
    
    def compare_models(self, model_id1: str, model_id2: str) -> Dict:
        """Compare two model versions"""
        model1 = self.get_model_info(model_id1)
        model2 = self.get_model_info(model_id2)
        
        if not model1 or not model2:
            raise ValueError("One or both models not found")
        
        comparison = {
            "model1": {
                "id": model_id1,
                "version": model1["version"],
                "size_mb": model1["model_size_mb"],
                "created_at": model1["created_at"]
            },
            "model2": {
                "id": model_id2,
                "version": model2["version"],
                "size_mb": model2["model_size_mb"],
                "created_at": model2["created_at"]
            },
            "differences": {
                "size_diff_mb": model2["model_size_mb"] - model1["model_size_mb"],
                "hash_differs": model1["model_hash"] != model2["model_hash"]
            }
        }
        
        # Compare metrics if available
        if model1["metrics"] and model2["metrics"]:
            metrics_diff = {}
            for metric in set(model1["metrics"].keys()) | set(model2["metrics"].keys()):
                val1 = model1["metrics"].get(metric, 0)
                val2 = model2["metrics"].get(metric, 0)
                metrics_diff[metric] = {
                    "model1": val1,
                    "model2": val2,
                    "improvement": val2 - val1
                }
            comparison["metrics_comparison"] = metrics_diff
        
        return comparison
    
    def tag_model(self, model_id: str, tags: List[str]):
        """Add tags to a model"""
        if model_id not in self.registry["models"]:
            raise ValueError(f"Model not found: {model_id}")
        
        existing_tags = set(self.registry["models"][model_id]["tags"])
        new_tags = existing_tags | set(tags)
        
        self.registry["models"][model_id]["tags"] = list(new_tags)
        self._save_registry()
        
        logger.info(f"Tags added to {model_id}: {tags}")
    
    def delete_model(self, model_id: str, confirm: bool = False):
        """Delete a model version"""
        if not confirm:
            raise ValueError("Must set confirm=True to delete model")
        
        if model_id not in self.registry["models"]:
            raise ValueError(f"Model not found: {model_id}")
        
        # Remove local files
        local_model_dir = self.base_dir / model_id
        if local_model_dir.exists():
            shutil.rmtree(local_model_dir)
        
        local_archive = self.base_dir / f"{model_id}.tar.gz"
        if local_archive.exists():
            local_archive.unlink()
        
        # Remove from S3 if configured
        if self.s3_bucket:
            s3_key = f"{self.s3_prefix}/{model_id}.tar.gz"
            try:
                self.s3_client.delete_object(Bucket=self.s3_bucket, Key=s3_key)
                logger.info(f"Deleted from S3: s3://{self.s3_bucket}/{s3_key}")
            except Exception as e:
                logger.warning(f"Failed to delete from S3: {e}")
        
        # Remove from registry
        del self.registry["models"][model_id]
        self._save_registry()
        
        logger.info(f"Model deleted: {model_id}")


def main():
    parser = argparse.ArgumentParser(description='Model versioning system')
    parser.add_argument('command', choices=['register', 'list', 'info', 'download', 'compare', 'tag', 'delete'])
    parser.add_argument('--model-dir', help='Model directory to register')
    parser.add_argument('--model-id', help='Model ID')
    parser.add_argument('--model-id2', help='Second model ID for comparison')
    parser.add_argument('--name', help='Model name')
    parser.add_argument('--description', default='', help='Model description')
    parser.add_argument('--version', help='Model version')
    parser.add_argument('--output-dir', default='downloaded_models', help='Output directory')
    parser.add_argument('--tags', nargs='*', help='Model tags')
    parser.add_argument('--s3-bucket', help='S3 bucket for remote storage')
    parser.add_argument('--base-dir', default='models', help='Local base directory')
    parser.add_argument('--force', action='store_true', help='Force operation')
    parser.add_argument('--confirm', action='store_true', help='Confirm destructive operations')
    
    args = parser.parse_args()
    
    # Initialize versioning system
    versioning = ModelVersioningSystem(
        base_dir=args.base_dir,
        s3_bucket=args.s3_bucket
    )
    
    if args.command == 'register':
        if not args.model_dir or not args.name:
            parser.error("--model-dir and --name are required for register command")
        
        model_id = versioning.register_model(
            model_dir=args.model_dir,
            name=args.name,
            description=args.description,
            version=args.version,
            tags=args.tags or []
        )
        print(f"Model registered: {model_id}")
    
    elif args.command == 'list':
        models = versioning.list_models()
        print(f"Found {len(models)} models:")
        for model in models:
            print(f"  {model['model_id']} - {model['name']} ({model['model_size_mb']:.2f} MB)")
            print(f"    Created: {model['created_at']}")
            print(f"    Tags: {', '.join(model['tags'])}")
            print()
    
    elif args.command == 'info':
        if not args.model_id:
            parser.error("--model-id is required for info command")
        
        info = versioning.get_model_info(args.model_id)
        if info:
            print(json.dumps(info, indent=2))
        else:
            print(f"Model not found: {args.model_id}")
    
    elif args.command == 'download':
        if not args.model_id:
            parser.error("--model-id is required for download command")
        
        model_path = versioning.download_model(
            model_id=args.model_id,
            output_dir=args.output_dir,
            force=args.force
        )
        print(f"Model downloaded to: {model_path}")
    
    elif args.command == 'compare':
        if not args.model_id or not args.model_id2:
            parser.error("--model-id and --model-id2 are required for compare command")
        
        comparison = versioning.compare_models(args.model_id, args.model_id2)
        print(json.dumps(comparison, indent=2))
    
    elif args.command == 'tag':
        if not args.model_id or not args.tags:
            parser.error("--model-id and --tags are required for tag command")
        
        versioning.tag_model(args.model_id, args.tags)
        print(f"Tags added to {args.model_id}: {args.tags}")
    
    elif args.command == 'delete':
        if not args.model_id:
            parser.error("--model-id is required for delete command")
        
        versioning.delete_model(args.model_id, confirm=args.confirm)
        print(f"Model deleted: {args.model_id}")


if __name__ == "__main__":
    main()