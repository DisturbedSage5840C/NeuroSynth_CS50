variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "eks_cluster_name" {
  type    = string
  default = "neurosynth-eks"
}

variable "eks_cluster_role_arn" {
  type = string
}

variable "eks_node_role_arn" {
  type = string
}

variable "private_subnet_ids" {
  type = list(string)
}

variable "timescale_kms_key_arn" {
  type = string
}

variable "eks_oidc_provider_arn" {
  type = string
}

variable "audit_bucket_name" {
  type = string
}

variable "healthlake_kms_key_arn" {
  type = string
}

# ── GPU Node Group (Priority 9) ─────────────────────
variable "gpu_instance_type" {
  type        = string
  default     = "g4dn.xlarge"
  description = "EC2 instance type for GPU inference nodes"
}

variable "gpu_min_nodes" {
  type    = number
  default = 0
}

variable "gpu_max_nodes" {
  type    = number
  default = 4
}

variable "gpu_desired_nodes" {
  type    = number
  default = 1
}

# ── MSK Kafka (Priority 9) ──────────────────────────
variable "kafka_instance_type" {
  type        = string
  default     = "kafka.m5.large"
  description = "MSK broker instance type"
}

variable "kafka_broker_count" {
  type    = number
  default = 3
}

variable "kafka_ebs_volume_size" {
  type    = number
  default = 100
}
