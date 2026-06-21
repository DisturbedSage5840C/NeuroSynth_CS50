resource "aws_db_instance" "timescale" {
  identifier        = "neurosynth-timescaledb"
  engine            = "postgres"
  engine_version    = "16.2"
  instance_class    = "db.r6i.2xlarge"
  allocated_storage = 500
  storage_encrypted = true
  backup_retention_period = 7
  username          = "postgres"
  manage_master_user_password = true
  master_user_secret_kms_key_id = var.timescale_kms_key_arn
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id = "neurosynth-redis"
  engine               = "redis"
  node_type            = "cache.r7g.large"
  num_cache_clusters   = 2
  at_rest_encryption_enabled = true
}

resource "aws_neptune_cluster" "graph" {
  cluster_identifier = "neurosynth-neptune"
  backup_retention_period = 7
  storage_encrypted = true
}

variable "timescale_kms_key_arn" {
  type = string
}
