from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(model_dir: Path) -> dict:
    files = []
    for p in sorted(model_dir.rglob("*")):
        if p.is_file():
            files.append(
                {
                    "path": str(p.relative_to(model_dir)),
                    "size_bytes": p.stat().st_size,
                    "sha256": sha256(p),
                }
            )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir),
        "files": files,
    }


def upload_to_r2(model_dir: Path) -> int:
    """Upload model artifacts to Cloudflare R2 (matches api.py's _pull_models_from_r2).

    Requires R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY in the env.
    Keys are stored under ``models/...`` so the API downloads them into ./models.
    """
    import os

    import boto3
    from botocore.config import Config

    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )
    bucket = os.environ.get("R2_BUCKET", "neurosynth-models")
    uploaded = 0
    for path in sorted(model_dir.rglob("*")):
        if path.is_file() and path.suffix in {".pkl", ".npy", ".pt", ".json"}:
            key = str(path)  # e.g. models/rf_model.pkl
            print(f"  uploading {key} ...", end=" ", flush=True)
            s3.upload_file(str(path), bucket, key)
            print("done")
            uploaded += 1
    print(f"\n{uploaded} artifacts uploaded to r2://{bucket}/")
    return uploaded


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish model artifact manifest / upload to R2")
    parser.add_argument("--model-dir", default=Path("models"), type=Path)
    parser.add_argument("--out", default=Path("artifacts/model_artifacts_manifest.json"), type=Path)
    parser.add_argument("--upload-r2", action="store_true", help="Upload artifacts to Cloudflare R2")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = collect(args.model_dir)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(args.out), "n_files": len(manifest["files"])}, indent=2))

    if args.upload_r2:
        upload_to_r2(args.model_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
