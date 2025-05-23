# AWS Batch Infrastructure for PaddleOCR Training

This directory contains AWS infrastructure configuration for running PaddleOCR training jobs on AWS Batch with GPU support.

## Prerequisites

1. **AWS CLI** installed and configured
2. **Docker** installed locally
3. **VPC with subnets** in your AWS account
4. **Appropriate IAM permissions** for:
   - CloudFormation
   - AWS Batch
   - ECR
   - S3
   - IAM role creation

## Quick Start

### 1. Set Environment Variables

```bash
export VPC_ID=vpc-xxxxxxxxx          # Your VPC ID
export SUBNET_IDS=subnet-xxx,subnet-yyy  # Comma-separated subnet IDs
export AWS_REGION=us-west-2          # Optional, defaults to us-west-2
```

### 2. Deploy Infrastructure

```bash
cd aws/
./deploy.sh
```

This will:
- Deploy CloudFormation stack with all required resources
- Build and push Docker image to ECR
- Create Batch job definitions
- Display deployment summary

### 3. Upload Training Data and Start Training

```bash
# Upload data and run complete training pipeline
python ../scripts/aws_batch_training.py pipeline \
    --training-data ../data/paddleocr_training \
    --s3-bucket pinball-ocr-training-training-data-ACCOUNT_ID \
    --job-queue pinball-ocr-training-gpu-queue \
    --job-definition pinball-ocr-training-gpu \
    --output-dir ../models/aws_trained
```

## Architecture

### Infrastructure Components

- **S3 Bucket**: Stores training data and model artifacts
- **ECR Repository**: Hosts Docker images
- **AWS Batch Compute Environments**: 
  - GPU environment (p3, g4dn instances)
  - CPU environment (c5 instances, fallback)
- **Job Queues**: Separate queues for GPU and CPU training
- **IAM Roles**: Execution roles with S3 and CloudWatch permissions

### Instance Types

**GPU Instances (Primary)**:
- p3.2xlarge (1x Tesla V100, 8 vCPUs, 61 GB RAM)
- p3.8xlarge (4x Tesla V100, 32 vCPUs, 244 GB RAM)
- g4dn.xlarge (1x Tesla T4, 4 vCPUs, 16 GB RAM)
- g4dn.2xlarge (1x Tesla T4, 8 vCPUs, 32 GB RAM)

**CPU Instances (Fallback)**:
- c5.large to c5.4xlarge

## Training Configuration

### Job Definition Parameters

```json
{
  "vcpus": 4,
  "memory": 16384,
  "resourceRequirements": [{"type": "GPU", "value": "1"}],
  "timeout": 14400,  // 4 hours
  "retryStrategy": {"attempts": 3}
}
```

### Environment Variables

- `CUDA_VISIBLE_DEVICES=0`: Use first GPU
- `PYTHONPATH=/app`: Python path
- `S3_BUCKET`: S3 bucket for data storage

## Usage Examples

### Manual Step-by-Step Process

```bash
# 1. Upload training data
python ../scripts/aws_batch_training.py upload \
    --training-data ../data/paddleocr_training \
    --s3-bucket YOUR_BUCKET_NAME

# 2. Submit training job
python ../scripts/aws_batch_training.py submit \
    --job-name my-training-job \
    --job-queue pinball-ocr-training-gpu-queue \
    --job-definition pinball-ocr-training-gpu \
    --s3-bucket YOUR_BUCKET_NAME

# 3. Monitor job progress
python ../scripts/aws_batch_training.py monitor \
    --job-id YOUR_JOB_ID

# 4. Download trained models
python ../scripts/aws_batch_training.py download \
    --s3-bucket YOUR_BUCKET_NAME \
    --output-dir ../models/aws_trained
```

### GPU vs CPU Training

```bash
# GPU training (recommended)
aws batch submit-job \
    --job-name gpu-training \
    --job-queue pinball-ocr-training-gpu-queue \
    --job-definition pinball-ocr-training-gpu

# CPU training (fallback)
aws batch submit-job \
    --job-name cpu-training \
    --job-queue pinball-ocr-training-cpu-queue \
    --job-definition pinball-ocr-training-cpu
```

## Monitoring and Logging

### CloudWatch Logs
All training logs are automatically sent to CloudWatch:
- Log Group: `/aws/batch/pinball-ocr-training`
- Log Stream: `{job-id}`

### Job Monitoring
```bash
# Check job status
aws batch describe-jobs --jobs JOB_ID

# List jobs in queue
aws batch list-jobs --job-queue QUEUE_NAME

# Cancel running job
aws batch cancel-job --job-id JOB_ID --reason "User requested cancellation"
```

## Cost Optimization

### Spot Instances
The compute environments can be configured to use Spot instances for significant cost savings:

```yaml
# In cloudformation-template.yml
BidPercentage: 50  # Bid 50% of On-Demand price
Type: SPOT
```

### Auto Scaling
- **Min vCPUs**: 0 (scales to zero when idle)
- **Desired vCPUs**: 0 (starts with no instances)
- **Max vCPUs**: 256 (configurable)

### Training Time Estimates
- **GPU (p3.2xlarge)**: ~1-2 hours for typical dataset
- **GPU (g4dn.xlarge)**: ~2-4 hours for typical dataset  
- **CPU (c5.xlarge)**: ~8-12 hours for typical dataset

## Troubleshooting

### Common Issues

1. **Job Stuck in RUNNABLE State**
   - Check compute environment capacity
   - Verify subnets have available capacity
   - Check service limits

2. **Docker Image Pull Errors**
   - Verify ECR repository exists
   - Check IAM permissions for ECR access
   - Ensure image was pushed successfully

3. **S3 Access Denied**
   - Verify IAM role has S3 permissions
   - Check bucket policy
   - Ensure bucket exists in same region

4. **Training Fails with CUDA Errors**
   - Verify GPU instances are available
   - Check CUDA compatibility
   - Try CPU queue as fallback

### Debugging Commands

```bash
# Check CloudFormation stack status
aws cloudformation describe-stacks --stack-name pinball-ocr-training-infrastructure

# List compute environments
aws batch describe-compute-environments

# Check job queue status
aws batch describe-job-queues

# View recent job logs
aws logs get-log-events \
    --log-group-name /aws/batch/pinball-ocr-training \
    --log-stream-name JOB_ID
```

## Cleanup

To avoid ongoing costs, delete the infrastructure when not in use:

```bash
# Delete CloudFormation stack (removes all resources)
aws cloudformation delete-stack --stack-name pinball-ocr-training-infrastructure

# Manually delete S3 bucket contents if needed
aws s3 rm s3://YOUR_BUCKET_NAME --recursive
```

## Security Considerations

- All resources are deployed in private subnets
- S3 bucket has public access blocked
- IAM roles follow principle of least privilege
- ECR images are scanned for vulnerabilities
- CloudWatch logs are retained for 30 days

## Customization

### Modify Instance Types
Edit `cloudformation-template.yml` to change instance types:

```yaml
InstanceTypes:
  - p3.2xlarge
  - g4dn.xlarge
  # Add or remove instance types
```

### Adjust Resource Limits
Modify job definition for different resource requirements:

```json
{
  "vcpus": 8,           // More CPUs
  "memory": 32768,      // More memory
  "resourceRequirements": [
    {"type": "GPU", "value": "2"}  // Multiple GPUs
  ]
}
```

### Custom Training Parameters
Pass custom parameters through environment variables:

```bash
python ../scripts/aws_batch_training.py pipeline \
    --training-data ../data/paddleocr_training \
    # ... other args
    --config-learning-rate 0.001 \
    --config-batch-size 32
```