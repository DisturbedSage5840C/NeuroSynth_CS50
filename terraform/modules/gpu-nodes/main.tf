# GPU Node Group for EKS — g4dn.xlarge instances for model inference
# Deploy alongside the main kubernetes-cluster module

variable "environment" {
  type = string
}

variable "cluster_name" {
  type = string
}

variable "private_subnet_ids" {
  type = list(string)
}

variable "gpu_instance_type" {
  type    = string
  default = "g4dn.xlarge"
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

variable "tags" {
  type    = map(string)
  default = {}
}

# IAM role for GPU nodes
resource "aws_iam_role" "gpu_node" {
  name = "neurosynth-${var.environment}-gpu-node-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "gpu_worker" {
  for_each = toset([
    "arn:aws:iam::policy/AmazonEKSWorkerNodePolicy",
    "arn:aws:iam::policy/AmazonEKS_CNI_Policy",
    "arn:aws:iam::policy/AmazonEC2ContainerRegistryReadOnly",
  ])

  role       = aws_iam_role.gpu_node.name
  policy_arn = each.value
}

# GPU node group with NVIDIA drivers
resource "aws_eks_node_group" "gpu" {
  cluster_name    = var.cluster_name
  node_group_name = "neurosynth-gpu-${var.environment}"
  node_role_arn   = aws_iam_role.gpu_node.arn
  subnet_ids      = var.private_subnet_ids

  instance_types = [var.gpu_instance_type]
  ami_type       = "AL2_x86_64_GPU"  # Amazon Linux 2 with NVIDIA drivers

  scaling_config {
    min_size     = var.gpu_min_nodes
    max_size     = var.gpu_max_nodes
    desired_size = var.gpu_desired_nodes
  }

  # GPU nodes get tainted so only GPU workloads schedule on them
  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  labels = {
    "workload-type"           = "gpu-inference"
    "nvidia.com/gpu.present"  = "true"
    "neurosynth/node-purpose" = "model-serving"
  }

  tags = merge(var.tags, {
    Name = "neurosynth-gpu-${var.environment}"
  })

  depends_on = [aws_iam_role_policy_attachment.gpu_worker]
}

# Kubernetes NVIDIA device plugin DaemonSet
resource "kubernetes_daemon_set_v1" "nvidia_device_plugin" {
  metadata {
    name      = "nvidia-device-plugin"
    namespace = "kube-system"
  }

  spec {
    selector {
      match_labels = {
        name = "nvidia-device-plugin"
      }
    }

    template {
      metadata {
        labels = {
          name = "nvidia-device-plugin"
        }
      }

      spec {
        toleration {
          key      = "nvidia.com/gpu"
          operator = "Exists"
          effect   = "NoSchedule"
        }

        container {
          name  = "nvidia-device-plugin"
          image = "nvcr.io/nvidia/k8s-device-plugin:v0.15.0"

          security_context {
            privileged = true
          }

          volume_mount {
            name       = "device-plugin"
            mount_path = "/var/lib/kubelet/device-plugins"
          }
        }

        volume {
          name = "device-plugin"
          host_path {
            path = "/var/lib/kubelet/device-plugins"
          }
        }
      }
    }
  }
}

output "gpu_node_group_name" {
  value       = aws_eks_node_group.gpu.node_group_name
  description = "GPU node group name"
}

output "gpu_node_group_status" {
  value       = aws_eks_node_group.gpu.status
  description = "GPU node group status"
}
