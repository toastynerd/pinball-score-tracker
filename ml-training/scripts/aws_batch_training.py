#!/usr/bin/env python3
"""
AWS Batch training orchestration script for PaddleOCR
Handles job submission, monitoring, and artifact management
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import boto3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AWSBatchTrainingOrchestrator:
    """Orchestrates PaddleOCR training on AWS Batch"""

    def __init__(self, region: str = "us-west-2"):
        """
        Initialize AWS Batch orchestrator

        Args:
            region: AWS region
        """
        self.region = region
        self.batch_client = boto3.client("batch", region_name=region)
        self.s3_client = boto3.client("s3", region_name=region)
        self.logs_client = boto3.client("logs", region_name=region)

    def upload_training_data(
        self, local_data_dir: str, s3_bucket: str, s3_prefix: str = "training-data"
    ) -> str:
        """
        Upload training data to S3

        Args:
            local_data_dir: Local directory containing training data
            s3_bucket: S3 bucket name
            s3_prefix: S3 prefix for the data

        Returns:
            S3 URI of uploaded data
        """
        local_path = Path(local_data_dir)
        if not local_path.exists():
            raise ValueError(f"Local data directory does not exist: {local_data_dir}")

        logger.info(
            f"Uploading training data from {local_data_dir} to s3://{s3_bucket}/{s3_prefix}"
        )

        uploaded_files = 0
        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_path)
                s3_key = f"{s3_prefix}/{relative_path}"

                try:
                    self.s3_client.upload_file(str(file_path), s3_bucket, s3_key)
                    uploaded_files += 1

                    if uploaded_files % 100 == 0:
                        logger.info(f"Uploaded {uploaded_files} files...")

                except Exception as e:
                    logger.error(f"Failed to upload {file_path}: {e}")

        logger.info(f"Successfully uploaded {uploaded_files} files to S3")
        return f"s3://{s3_bucket}/{s3_prefix}"

    def submit_training_job(
        self,
        job_name: str,
        job_queue: str,
        job_definition: str,
        s3_data_uri: str,
        s3_output_uri: str,
        training_config: Optional[Dict] = None,
    ) -> str:
        """
        Submit training job to AWS Batch

        Args:
            job_name: Name for the batch job
            job_queue: AWS Batch job queue name
            job_definition: AWS Batch job definition name
            s3_data_uri: S3 URI of training data
            s3_output_uri: S3 URI for output artifacts
            training_config: Optional training configuration overrides

        Returns:
            Job ID
        """
        # Prepare job parameters
        parameters = {
            "inputDataPath": s3_data_uri,
            "outputPath": s3_output_uri,
            "configPath": "configs/pinball_ocr_config.yml",
        }

        # Add training config overrides
        if training_config:
            for key, value in training_config.items():
                parameters[f"config_{key}"] = str(value)

        # Environment variables
        environment = [
            {"name": "AWS_DEFAULT_REGION", "value": self.region},
            {"name": "S3_DATA_URI", "value": s3_data_uri},
            {"name": "S3_OUTPUT_URI", "value": s3_output_uri},
        ]

        job_request = {
            "jobName": job_name,
            "jobQueue": job_queue,
            "jobDefinition": job_definition,
            "parameters": parameters,
            "containerOverrides": {
                "environment": environment,
                "command": [
                    "python",
                    "scripts/train_model.py",
                    "--config",
                    "configs/pinball_ocr_config.yml",
                    "--s3-input",
                    s3_data_uri,
                    "--s3-output",
                    s3_output_uri,
                ],
            },
        }

        logger.info(f"Submitting training job: {job_name}")
        response = self.batch_client.submit_job(**job_request)
        job_id = response["jobId"]

        logger.info(f"Job submitted successfully. Job ID: {job_id}")
        return job_id

    def monitor_job(self, job_id: str, poll_interval: int = 60) -> Dict:
        """
        Monitor training job progress

        Args:
            job_id: AWS Batch job ID
            poll_interval: Polling interval in seconds

        Returns:
            Final job status
        """
        logger.info(f"Monitoring job: {job_id}")

        while True:
            response = self.batch_client.describe_jobs(jobs=[job_id])
            job = response["jobs"][0]
            status = job["jobStatus"]

            logger.info(f"Job {job_id} status: {status}")

            if status in ["SUCCEEDED", "FAILED"]:
                if status == "SUCCEEDED":
                    logger.info(f"Job {job_id} completed successfully!")
                else:
                    logger.error(f"Job {job_id} failed!")
                    if "statusReason" in job:
                        logger.error(f"Reason: {job['statusReason']}")

                return job

            elif status == "RUNNING":
                # Try to get CloudWatch logs
                self._print_job_logs(job_id, tail=5)

            time.sleep(poll_interval)

    def _print_job_logs(self, job_id: str, tail: int = 10):
        """Print recent job logs from CloudWatch"""
        try:
            log_group = "/aws/batch/pinball-ocr-training"
            log_stream = f"{job_id}"

            response = self.logs_client.get_log_events(
                logGroupName=log_group,
                logStreamName=log_stream,
                limit=tail,
                startFromHead=False,
            )

            if response["events"]:
                logger.info("Recent logs:")
                for event in response["events"]:
                    timestamp = datetime.fromtimestamp(event["timestamp"] / 1000)
                    logger.info(f"  {timestamp}: {event['message']}")

        except Exception as e:
            logger.debug(f"Could not retrieve logs: {e}")

    def download_artifacts(self, s3_output_uri: str, local_output_dir: str):
        """
        Download training artifacts from S3

        Args:
            s3_output_uri: S3 URI of training outputs
            local_output_dir: Local directory to download artifacts
        """
        # Parse S3 URI
        s3_parts = s3_output_uri.replace("s3://", "").split("/", 1)
        bucket = s3_parts[0]
        prefix = s3_parts[1] if len(s3_parts) > 1 else ""

        local_path = Path(local_output_dir)
        local_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading artifacts from {s3_output_uri} to {local_output_dir}")

        # List objects in S3
        response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

        if "Contents" not in response:
            logger.warning("No artifacts found in S3")
            return

        downloaded_files = 0
        for obj in response["Contents"]:
            s3_key = obj["Key"]
            relative_path = s3_key[len(prefix) :].lstrip("/")
            local_file = local_path / relative_path

            # Create parent directories
            local_file.parent.mkdir(parents=True, exist_ok=True)

            try:
                self.s3_client.download_file(bucket, s3_key, str(local_file))
                downloaded_files += 1
            except Exception as e:
                logger.error(f"Failed to download {s3_key}: {e}")

        logger.info(f"Downloaded {downloaded_files} artifacts")

    def run_complete_training_pipeline(
        self,
        training_data_dir: str,
        s3_bucket: str,
        job_queue: str,
        job_definition: str,
        output_dir: str,
        job_name_prefix: str = "pinball-ocr-training",
    ) -> Dict:
        """
        Run complete training pipeline: upload data, train, download results

        Args:
            training_data_dir: Local training data directory
            s3_bucket: S3 bucket for data and artifacts
            job_queue: AWS Batch job queue
            job_definition: AWS Batch job definition
            output_dir: Local output directory for artifacts
            job_name_prefix: Prefix for job name

        Returns:
            Dictionary with training results
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"{job_name_prefix}-{timestamp}"

        # Upload training data
        s3_data_uri = self.upload_training_data(
            training_data_dir, s3_bucket, f"training-data/{timestamp}"
        )

        # Set S3 output path
        s3_output_uri = f"s3://{s3_bucket}/training-outputs/{timestamp}"

        # Submit job
        job_id = self.submit_training_job(
            job_name=job_name,
            job_queue=job_queue,
            job_definition=job_definition,
            s3_data_uri=s3_data_uri,
            s3_output_uri=s3_output_uri,
        )

        # Monitor job
        job_status = self.monitor_job(job_id)

        # Download artifacts if successful
        if job_status["jobStatus"] == "SUCCEEDED":
            self.download_artifacts(s3_output_uri, output_dir)

        return {
            "job_id": job_id,
            "job_name": job_name,
            "status": job_status["jobStatus"],
            "s3_data_uri": s3_data_uri,
            "s3_output_uri": s3_output_uri,
            "local_output_dir": output_dir,
        }


def main():
    parser = argparse.ArgumentParser(
        description="AWS Batch PaddleOCR training orchestrator"
    )
    parser.add_argument(
        "command",
        choices=["upload", "submit", "monitor", "download", "pipeline"],
        help="Command to execute",
    )
    parser.add_argument(
        "--training-data", required=True, help="Local training data directory"
    )
    parser.add_argument("--s3-bucket", required=True, help="S3 bucket name")
    parser.add_argument("--job-queue", required=True, help="AWS Batch job queue name")
    parser.add_argument(
        "--job-definition", required=True, help="AWS Batch job definition name"
    )
    parser.add_argument("--output-dir", required=True, help="Local output directory")
    parser.add_argument("--job-name", help="Job name (for submit/monitor commands)")
    parser.add_argument("--job-id", help="Job ID (for monitor command)")
    parser.add_argument("--region", default="us-west-2", help="AWS region")

    args = parser.parse_args()

    orchestrator = AWSBatchTrainingOrchestrator(region=args.region)

    if args.command == "pipeline":
        # Run complete pipeline
        result = orchestrator.run_complete_training_pipeline(
            training_data_dir=args.training_data,
            s3_bucket=args.s3_bucket,
            job_queue=args.job_queue,
            job_definition=args.job_definition,
            output_dir=args.output_dir,
        )

        print("Training pipeline completed!")
        print(f"Job ID: {result['job_id']}")
        print(f"Status: {result['status']}")
        print(f"Output directory: {result['local_output_dir']}")

    elif args.command == "upload":
        s3_uri = orchestrator.upload_training_data(args.training_data, args.s3_bucket)
        print(f"Data uploaded to: {s3_uri}")

    elif args.command == "submit":
        if not args.job_name:
            args.job_name = (
                f"pinball-ocr-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )

        s3_data_uri = f"s3://{args.s3_bucket}/training-data"
        s3_output_uri = f"s3://{args.s3_bucket}/training-outputs/{args.job_name}"

        job_id = orchestrator.submit_training_job(
            job_name=args.job_name,
            job_queue=args.job_queue,
            job_definition=args.job_definition,
            s3_data_uri=s3_data_uri,
            s3_output_uri=s3_output_uri,
        )
        print(f"Job submitted. Job ID: {job_id}")

    elif args.command == "monitor":
        if not args.job_id:
            raise ValueError("--job-id is required for monitor command")

        job_status = orchestrator.monitor_job(args.job_id)
        print(f"Final status: {job_status['jobStatus']}")

    elif args.command == "download":
        s3_output_uri = f"s3://{args.s3_bucket}/training-outputs"
        orchestrator.download_artifacts(s3_output_uri, args.output_dir)
        print(f"Artifacts downloaded to: {args.output_dir}")


if __name__ == "__main__":
    main()
