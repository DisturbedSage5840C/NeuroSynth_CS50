# MSK (Managed Streaming for Apache Kafka) module
# For streaming biomarker ingestion and real-time prediction events

variable "environment" {
  type = string
}

variable "cluster_name" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "subnet_ids" {
  type = list(string)
}

variable "instance_type" {
  type    = string
  default = "kafka.m5.large"
}

variable "broker_count" {
  type    = number
  default = 3
}

variable "ebs_volume_size" {
  type    = number
  default = 100
}

variable "tags" {
  type    = map(string)
  default = {}
}

# Security group for MSK
resource "aws_security_group" "msk" {
  name_prefix = "neurosynth-msk-${var.environment}-"
  vpc_id      = var.vpc_id

  ingress {
    description = "Kafka plaintext"
    from_port   = 9092
    to_port     = 9092
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
  }

  ingress {
    description = "Kafka TLS"
    from_port   = 9094
    to_port     = 9094
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
  }

  ingress {
    description = "Zookeeper"
    from_port   = 2181
    to_port     = 2181
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, {
    Name = "neurosynth-msk-${var.environment}"
  })
}

# MSK Configuration
resource "aws_msk_configuration" "neurosynth" {
  name              = "neurosynth-${var.environment}"
  kafka_versions    = ["3.6.0"]
  server_properties = <<PROPERTIES
auto.create.topics.enable=true
default.replication.factor=3
min.insync.replicas=2
num.partitions=6
log.retention.hours=168
log.retention.bytes=1073741824
message.max.bytes=10485760
PROPERTIES
}

# MSK Cluster
resource "aws_msk_cluster" "neurosynth" {
  cluster_name           = var.cluster_name
  kafka_version          = "3.6.0"
  number_of_broker_nodes = var.broker_count

  broker_node_group_info {
    instance_type  = var.instance_type
    client_subnets = var.subnet_ids
    security_groups = [aws_security_group.msk.id]

    storage_info {
      ebs_storage_info {
        volume_size = var.ebs_volume_size
      }
    }
  }

  configuration_info {
    arn      = aws_msk_configuration.neurosynth.arn
    revision = aws_msk_configuration.neurosynth.latest_revision
  }

  encryption_info {
    encryption_in_transit {
      client_broker = "TLS_PLAINTEXT"
      in_cluster    = true
    }
  }

  open_monitoring {
    prometheus {
      jmx_exporter {
        enabled_in_broker = true
      }
      node_exporter {
        enabled_in_broker = true
      }
    }
  }

  logging_info {
    broker_logs {
      cloudwatch_logs {
        enabled   = true
        log_group = "/msk/neurosynth-${var.environment}"
      }
    }
  }

  tags = merge(var.tags, {
    Name = var.cluster_name
  })
}

output "bootstrap_servers" {
  value       = aws_msk_cluster.neurosynth.bootstrap_brokers
  description = "MSK Kafka bootstrap servers (plaintext)."
}

output "bootstrap_servers_tls" {
  value       = aws_msk_cluster.neurosynth.bootstrap_brokers_tls
  description = "MSK Kafka bootstrap servers (TLS)."
}

output "zookeeper_connect" {
  value       = aws_msk_cluster.neurosynth.zookeeper_connect_string
  description = "Zookeeper connection string."
}
