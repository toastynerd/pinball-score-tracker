#!/bin/bash

# AWS Batch deployment script for PaddleOCR training
set -e

# Configuration
PROJECT_NAME="pinball-ocr-training"
AWS_REGION="${AWS_REGION:-us-west-2}"
STACK_NAME="${PROJECT_NAME}-infrastructure"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check if AWS CLI is installed and configured
check_aws_cli() {
    if ! command -v aws &> /dev/null; then
        error "AWS CLI is not installed. Please install it first."
    fi
    
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS CLI is not configured. Please run 'aws configure'."
    fi
    
    log "AWS CLI configured successfully"
}

# Get AWS Account ID
get_account_id() {
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    log "Using AWS Account ID: $ACCOUNT_ID"
}

# Check if required parameters are provided
check_parameters() {
    if [ -z "$VPC_ID" ]; then
        error "VPC_ID environment variable is required"
    fi
    
    if [ -z "$SUBNET_IDS" ]; then
        error "SUBNET_IDS environment variable is required (comma-separated)"
    fi
    
    log "Parameters validated"
}

# Deploy CloudFormation stack
deploy_infrastructure() {
    log "Deploying CloudFormation stack: $STACK_NAME"
    
    aws cloudformation deploy \
        --template-file cloudformation-template.yml \
        --stack-name "$STACK_NAME" \
        --parameter-overrides \
            ProjectName="$PROJECT_NAME" \
            VpcId="$VPC_ID" \
            SubnetIds="$SUBNET_IDS" \
            MaxvCpus="${MAX_VCPUS:-256}" \
            MinvCpus="${MIN_VCPUS:-0}" \
        --capabilities CAPABILITY_NAMED_IAM \
        --region "$AWS_REGION"
    
    log "CloudFormation stack deployed successfully"
}

# Get stack outputs
get_stack_outputs() {
    log "Retrieving stack outputs..."
    
    S3_BUCKET=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --query 'Stacks[0].Outputs[?OutputKey==`S3BucketName`].OutputValue' \
        --output text \
        --region "$AWS_REGION")
    
    ECR_REPOSITORY=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --query 'Stacks[0].Outputs[?OutputKey==`ECRRepositoryURI`].OutputValue' \
        --output text \
        --region "$AWS_REGION")
    
    GPU_JOB_QUEUE=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --query 'Stacks[0].Outputs[?OutputKey==`BatchJobQueueGPU`].OutputValue' \
        --output text \
        --region "$AWS_REGION")
    
    CPU_JOB_QUEUE=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --query 'Stacks[0].Outputs[?OutputKey==`BatchJobQueueCPU`].OutputValue' \
        --output text \
        --region "$AWS_REGION")
    
    JOB_ROLE_ARN=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --query 'Stacks[0].Outputs[?OutputKey==`BatchJobRoleArn`].OutputValue' \
        --output text \
        --region "$AWS_REGION")
    
    log "Stack outputs retrieved successfully"
}

# Build and push Docker image to ECR
build_and_push_image() {
    log "Building and pushing Docker image to ECR..."
    
    # Get ECR login token
    aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_REPOSITORY"
    
    # Build image
    docker build -t "$PROJECT_NAME" -f ../Dockerfile ..
    
    # Tag for ECR
    docker tag "$PROJECT_NAME:latest" "$ECR_REPOSITORY:latest"
    
    # Push to ECR
    docker push "$ECR_REPOSITORY:latest"
    
    log "Docker image pushed successfully"
}

# Create Batch job definition
create_job_definition() {
    log "Creating Batch job definition..."
    
    # Update job definition template with actual values
    sed "s/ACCOUNT_ID/$ACCOUNT_ID/g; s/REGION/$AWS_REGION/g" batch-job-definition.json > job-definition-temp.json
    
    # Create job definition
    aws batch register-job-definition \
        --cli-input-json file://job-definition-temp.json \
        --region "$AWS_REGION"
    
    # Clean up temp file
    rm job-definition-temp.json
    
    log "Batch job definition created successfully"
}

# Generate deployment summary
generate_summary() {
    log "Deployment completed successfully!"
    
    cat << EOF

=== DEPLOYMENT SUMMARY ===
Project Name: $PROJECT_NAME
AWS Region: $AWS_REGION
Account ID: $ACCOUNT_ID

Resources Created:
- S3 Bucket: $S3_BUCKET
- ECR Repository: $ECR_REPOSITORY
- GPU Job Queue: $GPU_JOB_QUEUE
- CPU Job Queue: $CPU_JOB_QUEUE
- Job Role ARN: $JOB_ROLE_ARN

Next Steps:
1. Upload training data to S3: aws s3 sync ../data/ s3://$S3_BUCKET/data/
2. Submit a training job: aws batch submit-job --job-name training-job-1 --job-queue $GPU_JOB_QUEUE --job-definition pinball-ocr-training-gpu

EOF
}

# Main execution
main() {
    log "Starting AWS Batch deployment for $PROJECT_NAME"
    
    check_aws_cli
    get_account_id
    check_parameters
    deploy_infrastructure
    get_stack_outputs
    build_and_push_image
    create_job_definition
    generate_summary
}

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy AWS Batch infrastructure for PaddleOCR training.

Required Environment Variables:
  VPC_ID          - ID of the VPC to deploy resources in
  SUBNET_IDS      - Comma-separated list of subnet IDs

Optional Environment Variables:
  AWS_REGION      - AWS region (default: us-west-2)
  MAX_VCPUS       - Maximum vCPUs (default: 256)
  MIN_VCPUS       - Minimum vCPUs (default: 0)

Example:
  export VPC_ID=vpc-1234567890abcdef0
  export SUBNET_IDS=subnet-12345678,subnet-87654321
  ./deploy.sh

Options:
  -h, --help      Show this help message

EOF
}

# Parse command line arguments
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac