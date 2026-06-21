resource "aws_iam_role" "irsa_api" {
  name = "neurosynth-irsa-api"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{ Action = "sts:AssumeRoleWithWebIdentity", Effect = "Allow", Principal = { Federated = var.eks_oidc_provider_arn } }]
  })
}

resource "aws_iam_policy" "healthlake_access" {
  name = "neurosynth-healthlake-access"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{ Effect = "Allow", Action = ["healthlake:*", "s3:*", "secretsmanager:GetSecretValue"], Resource = "*" }]
  })
}

resource "aws_iam_role_policy_attachment" "attach" {
  role       = aws_iam_role.irsa_api.name
  policy_arn = aws_iam_policy.healthlake_access.arn
}

variable "eks_oidc_provider_arn" {
  type = string
}
