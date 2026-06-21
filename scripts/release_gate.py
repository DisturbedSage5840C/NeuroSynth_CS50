# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_PATTERNS = [
    r"123456789012",
    r"REPLACE_IN_SECRET_MANAGER",
    r"example\.org",
    r"\bstub\b",
    r"\bplaceholder\b",
]

FORBIDDEN_SRC_MARKERS = [
    "TODO",
    "PLACEHOLDER",
]

# The real production env contract enforced by backend/core/config.py (env prefix
# NEUROSYNTH_). Previously these used a stale NEURO_ prefix that matched nothing in
# the config, so the gate passed vacuously and gave no coverage.
REQUIRED_PROD_ENV_KEYS = [
    "NEUROSYNTH_APP_ENV",
    "NEUROSYNTH_JWT_SECRET",
    "NEUROSYNTH_PATIENT_HASH_SECRET",
    "NEUROSYNTH_POSTGRES_DSN",
    "NEUROSYNTH_REDIS_URL",
    "NEUROSYNTH_ALLOWED_ORIGINS",
    "AWS_REGION",
    "AWS_ACCOUNT_ID",
    "ECR_REPOSITORY",
]

SCAN_GLOBS = ["src/**/*.py", "terraform/**/*.tf", "helm/**/*.yaml", "helm/**/*.yml", ".github/workflows/*.yml"]


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return proc.returncode, (proc.stdout + "\n" + proc.stderr).strip()


def scan_placeholders() -> list[str]:
    issues: list[str] = []
    regexes = [re.compile(p, re.IGNORECASE) for p in FORBIDDEN_PATTERNS]
    for glob in SCAN_GLOBS:
        for path in ROOT.glob(glob):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for rx in regexes:
                if rx.search(text):
                    issues.append(f"{path.relative_to(ROOT)} matched {rx.pattern}")
    return sorted(set(issues))


def scan_src_markers() -> list[str]:
    issues: list[str] = []
    for path in ROOT.glob("src/**/*.py"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in FORBIDDEN_SRC_MARKERS:
            if marker in text:
                issues.append(f"{path.relative_to(ROOT)} contains marker '{marker}'")
    return sorted(set(issues))


def parse_env_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key:
            keys.add(key)
    return keys


def main() -> int:
    report: dict[str, object] = {"checks": {}}

    code, out = run([sys.executable, "-m", "compileall", "src"])
    report["checks"]["compile_src"] = {"ok": code == 0, "output": out[-2000:]}

    placeholders = scan_placeholders()
    report["checks"]["placeholder_scan"] = {"ok": len(placeholders) == 0, "issues": placeholders}

    src_markers = scan_src_markers()
    report["checks"]["src_todo_placeholder_scan"] = {"ok": len(src_markers) == 0, "issues": src_markers}

    required_env = [
        "NEUROSYNTH_APP_ENV",
        "NEUROSYNTH_JWT_SECRET",
        "NEUROSYNTH_PATIENT_HASH_SECRET",
        "NEUROSYNTH_POSTGRES_DSN",
        "NEUROSYNTH_REDIS_URL",
        "NEUROSYNTH_ALLOWED_ORIGINS",
    ]
    missing = [v for v in required_env if not os.getenv(v)]
    report["checks"]["required_env"] = {"ok": len(missing) == 0, "missing": missing}

    env_prod = ROOT / ".env.prod.example"
    if env_prod.exists():
        declared = parse_env_keys(env_prod)
        missing_keys = [k for k in REQUIRED_PROD_ENV_KEYS if k not in declared]
        report["checks"]["env_prod_contract"] = {"ok": len(missing_keys) == 0, "missing_keys": missing_keys}
    else:
        report["checks"]["env_prod_contract"] = {"ok": False, "missing_keys": REQUIRED_PROD_ENV_KEYS}

    ok = all(v.get("ok", False) for v in report["checks"].values())
    report["release_ready"] = ok

    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
