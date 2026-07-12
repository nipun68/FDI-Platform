provider "aws" {
  region = "us-east-1"
}

# S3 Bucket for Data Lake & Versioning (DVC)
resource "aws_s3_bucket" "fdi_data_lake" {
  bucket = "fdi-platform-data-lake-unique-id"
  tags = {
    Name        = "FDI Data Lake"
    Environment = "Prod"
  }
}

# ECR Repository for Docker Images
resource "aws_ecr_repository" "fdi_api_repo" {
  name                 = "fdi-platform-api"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
}