resource "aws_cloudwatch_log_group" "app" {
  name              = "/neurosynth/app"
  retention_in_days = 90
}

resource "aws_cloudtrail" "audit" {
  name                          = "neurosynth-cloudtrail"
  s3_bucket_name                = var.audit_bucket_name
  include_global_service_events = true
  is_multi_region_trail         = true
}

resource "aws_guardduty_detector" "main" {
  enable = true
}

resource "aws_healthlake_fhir_datastore" "main" {
  datastore_name = "neurosynth-fhir"
  datastore_type_version = "R4"
  sse_configuration {
    kms_encryption_config {
      cmk_type = "CUSTOMER_MANAGED_KMS_KEY"
      kms_key_id = var.healthlake_kms_key_arn
    }
  }
}

variable "audit_bucket_name" {
  type = string
}

variable "healthlake_kms_key_arn" {
  type = string
}
