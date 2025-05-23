# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pinball Score Tracker is an AI-powered application for tracking pinball scores. The application aims to:
- Automatically extract scores using AI (no manual entry required)
- Track user location for comparing progress with other players locally and globally
- Provide score tracking and comparison features

## Current Status

This is a new project with minimal initial setup. The codebase structure and technology stack are yet to be determined.

## ML Model Development

**Technology Stack:**
- PyTorch for ML model training and inference
- Hugging Face models as base/pretrained models
- AWS for containerized training infrastructure

**Model Architecture:**
- PaddleOCR for end-to-end text detection and recognition from pinball score pictures
- Handles multiple numbers in single image (detection + recognition pipeline)
- Fine-tuning on pinball-specific score images
- Training pipeline designed for AWS Batch execution

## Development Notes

When setting up this project, consider:
- Technology stack selection (web app, mobile app, or both)
- Location services integration
- Database design for scores, users, and locations
- Authentication system for user accounts