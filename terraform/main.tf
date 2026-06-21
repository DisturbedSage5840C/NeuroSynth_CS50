terraform {
  required_version = ">= 1.7.0"

  # Workspace-separated state is expected: dev, staging, prod.
  # Example: terraform workspace select prod
  cloud {
    organization = "neurosynth"
    workspaces {
      tags = ["neurosynth"]
    }
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.45"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 5.35"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.30"
    }
  }
}

locals {
  # If var.environment is unset, workspace name becomes the environment contract.
  environment = var.environment != "" ? var.environment : terraform.workspace
  common_tags = {
    project             = "neurosynth"
    environment         = local.environment
    data_classification = "phi"
    managed_by          = "terraform"
  }
}

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = local.common_tags
  }
}

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

module "networking" {
  source = "./modules/networking"

  environment          = local.environment
  cloud_provider       = var.cloud_provider
  cidr_block           = var.vpc_cidr
  private_subnet_cidrs = var.private_subnet_cidrs
  public_subnet_cidrs  = var.public_subnet_cidrs

  # NAT gateway is mandatory for private egress in production workloads.
  enable_nat_gateway = true
  tags               = local.common_tags
}

module "kubernetes_cluster" {
  source = "./modules/kubernetes-cluster"

  environment    = local.environment
  cloud_provider = var.cloud_provider
  cluster_name   = "neurosynth-${local.environment}"

  # EKS path (managed node groups).
  aws_vpc_id             = module.networking.vpc_id
  aws_private_subnet_ids = module.networking.private_subnet_ids
  eks_node_instance_type = var.eks_node_instance_type
  eks_min_nodes          = var.eks_min_nodes
  eks_max_nodes          = var.eks_max_nodes

  # GKE path (Autopilot).
  gcp_project_id     = var.gcp_project_id
  gcp_region         = var.gcp_region
  gke_network        = module.networking.gcp_network_name
  gke_subnetwork     = module.networking.gcp_subnetwork_name
  gke_autopilot_mode = true

  tags = local.common_tags
}

module "postgres" {
  source = "./modules/postgres"

  environment    = local.environment
  cloud_provider = var.cloud_provider

  # AWS: RDS PostgreSQL / GCP: Cloud SQL PostgreSQL.
  vpc_id             = module.networking.vpc_id
  private_subnet_ids = module.networking.private_subnet_ids
  instance_class     = var.postgres_instance_class
  allocated_storage  = var.postgres_allocated_storage
  db_name            = "neurosynth"
  username           = var.postgres_username
  password           = var.postgres_password

  # Prefer private connectivity to reduce PHI exposure risk.
  enable_private_service_connect = true
  tags                           = local.common_tags
}

module "redis" {
  source = "./modules/redis"

  environment    = local.environment
  cloud_provider = var.cloud_provider
  vpc_id         = module.networking.vpc_id
  subnet_ids     = module.networking.private_subnet_ids
  node_type      = var.redis_node_type
  tags           = local.common_tags
}

module "lakehouse" {
  source = "./modules/lakehouse"

  environment    = local.environment
  cloud_provider = var.cloud_provider

  # Iceberg object storage: S3 on AWS, GCS on GCP.
  bucket_name            = "neurosynth-iceberg-${local.environment}"
  versioning_enabled     = true
  force_destroy_non_prod = local.environment != "prod"
  tags                   = local.common_tags
}

module "local_dev_minio" {
  source = "./modules/minio"
  count  = local.environment == "dev" ? 1 : 0

  enabled       = true
  minio_console = true
}

# ── GPU Node Group (Priority 9) ────────────────────────────
module "gpu_nodes" {
  source = "./modules/gpu-nodes"
  count  = var.cloud_provider == "aws" ? 1 : 0

  environment        = local.environment
  cluster_name       = module.kubernetes_cluster.cluster_name
  private_subnet_ids = module.networking.private_subnet_ids
  gpu_instance_type  = var.gpu_instance_type
  gpu_min_nodes      = var.gpu_min_nodes
  gpu_max_nodes      = var.gpu_max_nodes
  gpu_desired_nodes  = var.gpu_desired_nodes
  tags               = local.common_tags
}

# ── MSK Kafka Cluster (Priority 9) ─────────────────────────
module "kafka" {
  source = "./modules/kafka"
  count  = var.cloud_provider == "aws" ? 1 : 0

  environment    = local.environment
  cluster_name   = "neurosynth-kafka-${local.environment}"
  vpc_id         = module.networking.vpc_id
  subnet_ids     = module.networking.private_subnet_ids
  instance_type  = var.kafka_instance_type
  broker_count   = var.kafka_broker_count
  ebs_volume_size = var.kafka_ebs_volume_size
  tags           = local.common_tags
}

output "kubernetes_cluster_name" {
  value       = module.kubernetes_cluster.cluster_name
  description = "Target Kubernetes cluster name (EKS or GKE)."
}

output "postgres_endpoint" {
  value       = module.postgres.endpoint
  description = "Private PostgreSQL endpoint (Cloud SQL or RDS)."
}

output "redis_endpoint" {
  value       = module.redis.endpoint
  description = "Private Redis endpoint (ElastiCache or Memorystore)."
}

output "lakehouse_bucket" {
  value       = module.lakehouse.bucket_name
  description = "Object storage bucket for Iceberg lakehouse artifacts."
}

output "gpu_node_group" {
  value       = var.cloud_provider == "aws" ? module.gpu_nodes[0].gpu_node_group_name : "n/a"
  description = "GPU node group name (EKS only)."
}

output "kafka_bootstrap_servers" {
  value       = var.cloud_provider == "aws" ? module.kafka[0].bootstrap_servers : "n/a"
  description = "MSK Kafka bootstrap servers."
}

