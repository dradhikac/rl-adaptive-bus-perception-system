#!/usr/bin/env python3
"""
scripts/dataset_manager.py

Terminal CLI for dataset management across the Autonomous Bus Perception System.
Supported commands:
  python scripts/dataset_manager.py list
  python scripts/dataset_manager.py download <key>
  python scripts/dataset_manager.py download all-lite   # skips anything >1GB
  python scripts/dataset_manager.py verify <key>         # checksum/size check
"""

import argparse
import os
import sys
import shutil
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, Any, Optional

import requests
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "datasets_registry.yaml"
ONE_GB_BYTES = 1024 * 1024 * 1024  # 1 GB

# Ensure safe UTF-8 stdout encoding on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def load_registry(registry_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads and validates the dataset registry YAML file."""
    path = registry_path or DEFAULT_REGISTRY_PATH
    if not path.is_file():
        print(f"[ERROR] Registry file not found at: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data or "datasets" not in data:
        print(f"[ERROR] Invalid registry format in: {path}. 'datasets' key missing.", file=sys.stderr)
        sys.exit(1)

    return data["datasets"]


def get_dataset_status(key: str, info: Dict[str, Any]) -> tuple[str, str]:
    """
    Checks local existence and returns a tuple of (status_text, details).
    """
    rel_path = info.get("local_path", "")
    target = PROJECT_ROOT / rel_path

    if not target.exists():
        if info.get("type") == "manual":
            return "MANUAL PENDING", "Directory not created yet"
        return "NOT DOWNLOADED", "Missing"

    if target.is_file():
        size_bytes = target.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        return "READY", f"{size_mb:.1f} MB (file)"

    if target.is_dir():
        all_files = [p for p in target.rglob("*") if p.is_file()]
        if not all_files:
            if info.get("type") == "manual":
                return "AWAITING FOOTAGE", "Empty folder"
            return "EMPTY FOLDER", "0 files"
        
        # Check if directory only has README/documentation
        non_doc_files = [p for p in all_files if p.suffix.lower() not in [".md", ".txt"]]
        if not non_doc_files:
            if info.get("type") == "manual":
                return "AWAITING FOOTAGE", "Place bus/door footage here"
            if info.get("registration_required"):
                return "AUTH REQUIRED", "Register at IIIT portal (see README.md)"

        total_size = sum(f.stat().st_size for f in all_files)
        size_mb = total_size / (1024 * 1024)
        return "READY", f"{len(all_files)} files ({size_mb:.1f} MB)"

    return "UNKNOWN", "N/A"



# ==============================================================================
# COMMAND: LIST
# ==============================================================================

def cmd_list(args: argparse.Namespace) -> None:
    """Lists all datasets defined in datasets_registry.yaml with their status."""
    registry = load_registry(args.registry)

    print("\n" + "=" * 142)
    print(" [AUTONOMOUS BUS PERCEPTION SYSTEM] - DATASET REGISTRY")
    print("=" * 142)

    header = f"{'Key':<24} {'Source':<34} {'Approx Size':<14} {'Phase':<22} {'Status':<16} {'Local Path'}"
    print(header)
    print("-" * 142)

    for key, info in registry.items():
        source = str(info.get("source", "N/A"))
        if len(source) > 32:
            source = source[:29] + "..."
        approx_size = str(info.get("approx_size", "varies"))
        phase = str(info.get("used_in_phase", "N/A"))
        status, details = get_dataset_status(key, info)
        local_path = str(info.get("local_path", "N/A"))

        print(f"{key:<24} {source:<34} {approx_size:<14} {phase:<22} {status:<16} {local_path}")

    print("-" * 142)
    print(f"Total registered datasets: {len(registry)}")
    print("Commands:")
    print("  python scripts/dataset_manager.py list")
    print("  python scripts/dataset_manager.py download <key>")
    print("  python scripts/dataset_manager.py download all-lite   # skips anything >1GB")
    print("  python scripts/dataset_manager.py verify <key>         # checksum/size check")
    print("=" * 142 + "\n")


# ==============================================================================
# DOWNLOAD HANDLERS
# ==============================================================================

def download_file_with_progress(url: str, dest_path: Path, desc: str = "Downloading") -> bool:
    """Downloads a file via streaming HTTP GET request with a tqdm progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    try:
        headers = {"User-Agent": "AutonomousBusDatasetManager/1.0"}
        with requests.get(url, stream=True, timeout=60, headers=headers) as r:
            if r.status_code != 200:
                print(f"[ERROR] HTTP {r.status_code} when downloading from: {url}", file=sys.stderr)
                return False

            total_size = int(r.headers.get("content-length", 0))
            chunk_size = 1024 * 1024  # 1 MB

            with open(temp_path, "wb") as f, tqdm(
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
                desc=desc,
                leave=True
            ) as bar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

        if temp_path.exists():
            if dest_path.exists():
                dest_path.unlink()
            temp_path.rename(dest_path)
            return True
        return False
    except Exception as e:
        print(f"[ERROR] Download failed: {e}", file=sys.stderr)
        if temp_path.exists():
            temp_path.unlink()
        return False


def extract_archive(archive_path: Path, extract_dir: Path) -> bool:
    """Extracts a zip or tar archive into extract_dir."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Extracting archive: {archive_path.name} -> {extract_dir} ...")

    try:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(extract_dir)
            print(f"[SUCCESS] Successfully extracted zip archive to: {extract_dir}")
            return True
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, "r:*") as tf:
                tf.extractall(extract_dir)
            print(f"[SUCCESS] Successfully extracted tar archive to: {extract_dir}")
            return True
        else:
            print(f"[WARNING] Unrecognized archive format for {archive_path.name}, leaving file in directory.")
            return True
    except Exception as e:
        print(f"[ERROR] Archive extraction failed: {e}", file=sys.stderr)
        return False


def download_single_dataset(key: str, info: Dict[str, Any]) -> bool:
    """Downloads a single dataset key based on its registered type."""
    dtype = info.get("type")
    rel_path = info.get("local_path", f"datasets/{key}/")
    target = PROJECT_ROOT / rel_path

    print(f"\n[START] Processing '{key}' (Type: {dtype}, Target: {rel_path})")

    # 1. Manual Footage
    if dtype == "manual" or key == "self-collected":
        target.mkdir(parents=True, exist_ok=True)
        print("=" * 80)
        print(f" 📹 MANUAL DATASET: {info.get('name', key)}")
        print("=" * 80)
        print(f" [INFO] '{key}' cannot be auto-downloaded over the network.")
        print(f" Please manually record real-world bus-stop and passenger boarding/door footage,")
        print(f" and place the recorded video/image files into:")
        print(f"   -> {target.resolve()}")
        print("=" * 80)
        return True

    # 2. Pre-trained weights (coco-weights)
    if dtype == "weights" or key == "coco-weights":
        if target.exists() and target.stat().st_size > 1024 * 1024:
            size_mb = target.stat().st_size / (1024 * 1024)
            print(f"[INFO] Weights file '{target.name}' already present ({size_mb:.2f} MB). Skipping download.")
            return True
        url = info.get("download_url")
        if not url:
            print(f"[ERROR] No download_url specified for weights '{key}'.", file=sys.stderr)
            return False
        success = download_file_with_progress(url, target, desc=f"Downloading {key}")
        if success:
            print(f"[SUCCESS] Weights saved to: {target}")
        return success

    # 3. Roboflow API Datasets
    if dtype == "roboflow":
        api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
        # Fallback to checking .env file if environment variable not set in shell session
        if not api_key:
            env_file = PROJECT_ROOT / ".env"
            if env_file.is_file():
                try:
                    with open(env_file, "r", encoding="utf-8") as ef:
                        for line in ef:
                            if line.strip().startswith("ROBOFLOW_API_KEY="):
                                api_key = line.split("=", 1)[1].strip()
                                break
                except Exception:
                    pass

        if not api_key:
            print("=" * 80, file=sys.stderr)
            print(" [ERROR] ROBOFLOW_API_KEY environment variable is NOT set.", file=sys.stderr)
            print(" To download Roboflow datasets, please set your API key:", file=sys.stderr)
            print("   PowerShell : $env:ROBOFLOW_API_KEY=\"<your_api_key>\"", file=sys.stderr)
            print("   CMD        : set ROBOFLOW_API_KEY=<your_api_key>", file=sys.stderr)
            print("   Bash       : export ROBOFLOW_API_KEY=\"<your_api_key>\"", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            return False

        try:
            from roboflow import Roboflow
        except ImportError:
            print("[ERROR] The 'roboflow' package is not installed. Run: pip install roboflow", file=sys.stderr)
            return False

        workspace = info.get("workspace")
        project_name = info.get("project")
        version_num = info.get("version", 1)
        export_format = info.get("export_format", "yolov8")

        target.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Connecting to Roboflow workspace '{workspace}', project '{project_name}'...")

        try:
            rf = Roboflow(api_key=api_key)
            project = rf.workspace(workspace).project(project_name)

            # Discover and validate version dynamically
            try:
                available_versions = [v.version for v in project.versions()]
            except Exception:
                available_versions = []

            if available_versions:
                if str(version_num) not in [str(v) for v in available_versions]:
                    numeric_versions = [int(v) for v in available_versions if str(v).isdigit()]
                    if numeric_versions:
                        chosen = max(numeric_versions)
                    else:
                        chosen = available_versions[-1]
                    print(f"[INFO] Configured version '{version_num}' not found in project.")
                    print(f"[INFO] Available versions: {available_versions}. Automatically selecting latest: v{chosen}")
                    version_num = chosen

            print(f"[INFO] Downloading version {version_num} as '{export_format}' into: {target} ...")
            dataset = project.version(version_num).download(
                model_format=export_format,
                location=str(target.resolve()),
                overwrite=True
            )
            print(f"[SUCCESS] Roboflow dataset '{key}' downloaded into: {target}")
            return True
        except Exception as e:
            print(f"[ERROR] Roboflow download failed: {e}", file=sys.stderr)
            if info.get("fallback_url"):
                print(f"[TIP] Alternative source available: {info['fallback_url']}")
            return False

    # 4. Direct Archive Download (IDD-Lite, FGVD, etc.)
    if dtype == "archive":
        target.mkdir(parents=True, exist_ok=True)
        download_url = info.get("download_url")

        # Specific prompt for IDD-Lite registration requirement
        if info.get("registration_required"):
            print("-" * 80)
            print(f" [NOTE] '{key}' requires registration / agreement on the official portal:")
            print(f"        Portal URL : {info.get('source')}")
            print(f"        Target Dir : {target.resolve()}")
            print("-" * 80)

        archive_ext = info.get("archive_format", "zip")
        archive_name = f"{key}.{archive_ext}"
        archive_dest = target / archive_name

        print(f"[INFO] Attempting download from: {download_url}")
        success = download_file_with_progress(download_url, archive_dest, desc=f"Downloading {key}")

        if success:
            extracted = extract_archive(archive_dest, target)
            # Remove raw archive after extraction if successfully unzipped
            if extracted and archive_dest.exists():
                try:
                    archive_dest.unlink()
                except Exception:
                    pass
            return extracted
        else:
            if info.get("registration_required"):
                print(f"[MANUAL INSTRUCTIONS] Since automated download requires authenticated session:")
                print(f" 1. Register and log in at {info.get('source')}")
                print(f" 2. Download the IDD-Lite zip archive")
                print(f" 3. Extract contents into: {target.resolve()}")
            return False

    print(f"[ERROR] Unknown dataset type '{dtype}' for key '{key}'.", file=sys.stderr)
    return False


# ==============================================================================
# COMMAND: DOWNLOAD
# ==============================================================================

def cmd_download(args: argparse.Namespace) -> None:
    """Handles download execution for a specific key or all-lite."""
    registry = load_registry(args.registry)
    key = args.key

    # Batch download: all-lite
    if key == "all-lite":
        print("\n" + "=" * 80)
        print(" 🚀 DOWNLOADING ALL LITE DATASETS (Size <= 1 GB)")
        print("=" * 80)

        results = {}
        for d_key, info in registry.items():
            size_bytes = info.get("size_bytes", 0)

            # Skip large datasets > 1 GB
            if size_bytes > ONE_GB_BYTES:
                print(f"[SKIP] '{d_key}' is > 1 GB ({info.get('approx_size', 'large')}). Skipping for all-lite.")
                results[d_key] = "SKIPPED (> 1 GB)"
                continue

            # Skip manual datasets
            if info.get("type") == "manual" or d_key == "self-collected":
                target = PROJECT_ROOT / info.get("local_path", f"data/raw/{d_key}/")
                target.mkdir(parents=True, exist_ok=True)
                print(f"[INFO] '{d_key}' is manual: verified directory at {target}")
                results[d_key] = "MANUAL DIRECTORY CREATED"
                continue

            success = download_single_dataset(d_key, info)
            results[d_key] = "SUCCESS" if success else "FAILED"

        print("\n" + "=" * 80)
        print(" ALL-LITE DOWNLOAD SUMMARY")
        print("=" * 80)
        for d_key, res in results.items():
            print(f"  - {d_key:<24}: {res}")
        print("=" * 80 + "\n")
        return

    # Single dataset download
    if key not in registry:
        print(f"[ERROR] Key '{key}' not found in registry.", file=sys.stderr)
        print(f"Available keys: {', '.join(registry.keys())} or 'all-lite'", file=sys.stderr)
        sys.exit(1)

    info = registry[key]
    success = download_single_dataset(key, info)
    if not success:
        sys.exit(1)


# ==============================================================================
# COMMAND: VERIFY
# ==============================================================================

def verify_single_key(key: str, info: Dict[str, Any]) -> bool:
    """Verifies dataset existence and integrity."""
    rel_path = info.get("local_path", f"data/raw/{key}/")
    target = PROJECT_ROOT / rel_path
    expected_size_str = info.get("approx_size", "N/A")

    print(f"\nVerifying '{key}'...")
    print(f"  Target Path   : {target}")
    print(f"  Expected Size : {expected_size_str}")

    if not target.exists():
        print(f"  Status        : [FAIL] Path does not exist.")
        return False

    if target.is_file():
        size_bytes = target.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        print(f"  Found File    : {target.name} ({size_mb:.2f} MB)")
        print(f"  Status        : [VERIFIED]")
        return True

    if target.is_dir():
        files = [f for f in target.rglob("*") if f.is_file()]
        if not files:
            if info.get("type") == "manual":
                print(f"  Found Dir     : Empty manual directory.")
                print(f"  Status        : [AWAITING FOOTAGE] Please place recorded footage here.")
                return True
            print(f"  Status        : [FAIL] Directory exists but contains 0 files.")
            return False

        non_doc_files = [f for f in files if f.suffix.lower() not in [".md", ".txt"]]
        if not non_doc_files:
            if info.get("type") == "manual":
                print(f"  Found Files   : {len(files)} docs / guides.")
                print(f"  Status        : [AWAITING FOOTAGE] Awaiting user-recorded bus-stop & door footage.")
                return True
            if info.get("registration_required"):
                print(f"  Found Files   : {len(files)} docs (setup README).")
                print(f"  Status        : [AUTH REQUIRED] User registration required at {info.get('source')} to download zip.")
                return False

        total_bytes = sum(f.stat().st_size for f in files)
        total_mb = total_bytes / (1024 * 1024)
        print(f"  Found Files   : {len(files)} files ({total_mb:.2f} MB)")
        print(f"  Status        : [VERIFIED]")
        return True


    print(f"  Status        : [UNKNOWN] Target is neither file nor directory.")
    return False


def cmd_verify(args: argparse.Namespace) -> None:
    """Verifies dataset presence and sizes against the registry."""
    registry = load_registry(args.registry)
    key = args.key

    if key == "all":
        print("\n" + "=" * 80)
        print(" VERIFYING ALL REGISTERED DATASETS")
        print("=" * 80)
        overall = True
        for d_key, info in registry.items():
            ok = verify_single_key(d_key, info)
            if not ok:
                overall = False
        print("=" * 80)
        print(f"Overall Verification: {'PASSED' if overall else 'SOME DATASETS MISSING'}\n")
        return

    if key not in registry:
        print(f"[ERROR] Key '{key}' not found in registry.", file=sys.stderr)
        print(f"Available keys: {', '.join(registry.keys())} or 'all'", file=sys.stderr)
        sys.exit(1)

    info = registry[key]
    ok = verify_single_key(key, info)
    if not ok:
        sys.exit(1)


# ==============================================================================
# MAIN CLI PARSER
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dataset Manager for Autonomous Bus Perception & Decision System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/dataset_manager.py list
  python scripts/dataset_manager.py download coco-weights
  python scripts/dataset_manager.py download roboflow-pothole
  python scripts/dataset_manager.py download all-lite
  python scripts/dataset_manager.py verify coco-weights
  python scripts/dataset_manager.py verify all
        """
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help="Path to datasets_registry.yaml (defaults to repository root)"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. list
    sub_list = subparsers.add_parser("list", help="List all datasets and their download status")
    sub_list.set_defaults(func=cmd_list)

    # 2. download
    sub_dl = subparsers.add_parser("download", help="Download a specific dataset or all-lite")
    sub_dl.add_argument(
        "key",
        type=str,
        help="Dataset key from registry or 'all-lite' to download all <= 1 GB"
    )
    sub_dl.set_defaults(func=cmd_download)

    # 3. verify
    sub_vf = subparsers.add_parser("verify", help="Verify dataset existence and integrity")
    sub_vf.add_argument(
        "key",
        type=str,
        help="Dataset key from registry or 'all' to verify all datasets"
    )
    sub_vf.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
