#!/usr/bin/env python3
"""
scripts/prepare_dataset.py
===============================================================================
Autonomous Bus Perception & Decision System - Dataset Preparation Pipeline
===============================================================================
Prepares unified YOLOv8 training data by:
1. Ingesting FGVD (Pascal VOC XML) and IDD-Lite (Cityscapes masks) from data/raw/
   plus available complementary sources (roboflow-pothole, roboflow-traffic-sign,
   roboflow-bus-passenger, self-collected).
2. Mapping all source annotations onto a unified 10-class ontology defined in
   data/processed/classes.yaml:
   [person, bicycle, motorcycle, autorickshaw, car, bus, truck,
    bus_stop_sign, traffic_light, pothole]
3. Partitioning images by SOURCE DRIVING SESSION / SEQUENCE (80/10/10 train/val/test)
   to eliminate near-duplicate frame leakage between splits.
4. Exporting data.yaml formatted for Ultralytics YOLOv8 training.
5. Printing a class-balance report (image counts, instance counts, proportions,
   and representation alerts).

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --raw-dir data/raw --processed-dir data/processed
    python scripts/prepare_dataset.py --clean
===============================================================================
"""

import argparse
import hashlib
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image
import yaml
from tqdm import tqdm

# Force UTF-8 stdout encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==============================================================================
# UNIFIED CLASS ONTOLOGY
# ==============================================================================

UNIFIED_CLASSES = [
    "person",          # 0: Pedestrians, commuters, passengers, riders
    "bicycle",         # 1: Bicycles, cyclists
    "motorcycle",      # 2: Motorcycles, scooters, mopeds
    "autorickshaw",    # 3: Three-wheel auto-rickshaws (Bajaj, Piaggio, Atul, etc.)
    "car",             # 4: Sedans, hatchbacks, SUVs, vans, taxis
    "bus",             # 5: Transit buses, city buses, mini-buses
    "truck",           # 6: Heavy goods vehicles, lorries, tippers, container trucks
    "bus_stop_sign",   # 7: Bus stop markers, road signs, regulatory/warning signs
    "traffic_light",   # 8: Traffic signals (red/yellow/green)
    "pothole"          # 9: Road surface potholes, drain hazards
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(UNIFIED_CLASSES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(UNIFIED_CLASSES)}


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class BoundingBox:
    class_id: int
    class_name: str
    xc: float
    yc: float
    w: float
    h: float

    def to_yolo_line(self) -> str:
        return f"{self.class_id} {self.xc:.6f} {self.yc:.6f} {self.w:.6f} {self.h:.6f}\n"


@dataclass
class ImageRecord:
    image_id: str
    dataset_source: str
    sequence_id: str
    image_path: Path
    bboxes: List[BoundingBox] = field(default_factory=list)


# ==============================================================================
# CLASS MAPPING LOGIC
# ==============================================================================

def map_fgvd_class(raw_name: str) -> Optional[str]:
    """Maps FGVD vehicle object name to unified class."""
    if not raw_name:
        return None
    name = raw_name.strip().lower()

    if name.startswith("motorcycle") or name.startswith("scooter") or "moped" in name:
        return "motorcycle"
    if name.startswith("autorickshaw") or name.startswith("auto_") or "rickshaw" in name:
        return "autorickshaw"
    if name.startswith("car") or name.startswith("sedan") or name.startswith("suv") or name.startswith("van"):
        return "car"
    if name == "bus" or name.startswith("bus_") or name.startswith("mini-bus") or "minibus" in name:
        return "bus"
    if name.startswith("truck") or name.startswith("lorry") or "tipper" in name:
        return "truck"
    if "bicycle" in name or "cycle" in name:
        return "bicycle"
    if "person" in name or "pedestrian" in name:
        return "person"
    return None


def map_roboflow_traffic_sign(raw_name: str) -> Optional[str]:
    """Maps Indian traffic signs / signals to unified class."""
    name = raw_name.strip().upper()
    if name == "TRAFFIC_SIGNAL" or "TRAFFIC_LIGHT" in name:
        return "traffic_light"
    # All other regulatory, warning, and informational signs act as transit/road signs
    return "bus_stop_sign"


def map_roboflow_pothole(raw_name: str) -> Optional[str]:
    """Maps pothole dataset labels to unified class."""
    name = raw_name.strip().lower()
    if "pothole" in name or "drain" in name or "hole" in name or "sewer" in name:
        return "pothole"
    return None


# ==============================================================================
# DATASET INGESTION PARSERS
# ==============================================================================

def ingest_fgvd(raw_root: Path) -> List[ImageRecord]:
    """
    Parses Fine-Grained Vehicle Detection (FGVD) dataset.
    Reads Pascal VOC XML annotations from test, train, and val folders.
    Extracts sequence ID from XML <filename> tag.
    """
    records: List[ImageRecord] = []
    fgvd_candidates = [
        raw_root / "IDD_FGVD",
        raw_root / "fgvd",
        PROJECT_ROOT / "datasets" / "IDD_FGVD",
        PROJECT_ROOT / "datasets" / "fgvd"
    ]
    fgvd_dir = next((p for p in fgvd_candidates if p.is_dir()), None)
    if not fgvd_dir:
        print("[WARN] FGVD directory not found in raw data paths. Skipping FGVD.")
        return records

    print(f"[INFO] Ingesting FGVD from: {fgvd_dir}")
    anno_files: List[Path] = []
    for split in ["train", "val", "test"]:
        split_annos = fgvd_dir / split / "annos"
        if split_annos.is_dir():
            anno_files.extend(list(split_annos.glob("*.xml")))

    for xml_path in tqdm(anno_files, desc="Parsing FGVD XMLs", unit="xml"):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Locate image file
            split_dir = xml_path.parent.parent
            img_stem = xml_path.stem
            img_candidates = [
                split_dir / "images" / f"{img_stem}.jpg",
                split_dir / "images" / f"{img_stem}.png",
                split_dir / "images" / f"{img_stem}.jpeg"
            ]
            img_path = next((p for p in img_candidates if p.is_file()), None)
            if not img_path:
                continue

            # Extract sequence / driving session
            fn_elem = root.find("filename")
            fn_text = fn_elem.text if fn_elem is not None and fn_elem.text else ""
            fn_parts = fn_text.replace("\\", "/").split("/")
            if len(fn_parts) >= 3:
                sequence_id = f"fgvd_{fn_parts[1]}"
            elif len(fn_parts) >= 2:
                sequence_id = f"fgvd_{fn_parts[0]}"
            else:
                # Group by image numeric block
                num = "".join(c for c in img_stem if c.isdigit())
                blk = int(num) // 100 if num else 0
                sequence_id = f"fgvd_seq_block_{blk}"

            # Extract dimensions
            size_elem = root.find("size")
            if size_elem is not None:
                width = float(size_elem.find("width").text or 0)
                height = float(size_elem.find("height").text or 0)
            else:
                width, height = 0, 0

            if width <= 0 or height <= 0:
                with Image.open(img_path) as im:
                    width, height = float(im.width), float(im.height)

            record = ImageRecord(
                image_id=f"fgvd_{split_dir.name}_{img_stem}",
                dataset_source="IDD_FGVD",
                sequence_id=sequence_id,
                image_path=img_path
            )

            for obj in root.findall("object"):
                name_elem = obj.find("name")
                if name_elem is None or not name_elem.text:
                    continue
                unified_cls = map_fgvd_class(name_elem.text)
                if not unified_cls:
                    continue

                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)

                bw = xmax - xmin
                bh = ymax - ymin
                if bw <= 0 or bh <= 0:
                    continue

                xc = ((xmin + xmax) / 2.0) / width
                yc = ((ymin + ymax) / 2.0) / height
                nw = bw / width
                nh = bh / height

                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))

                record.bboxes.append(BoundingBox(
                    class_id=CLASS_TO_IDX[unified_cls],
                    class_name=unified_cls,
                    xc=xc, yc=yc, w=nw, h=nh
                ))

            records.append(record)

        except Exception:
            continue

    print(f"[INFO] Ingested {len(records)} images from FGVD.")
    return records


def ingest_idd_lite(raw_root: Path) -> List[ImageRecord]:
    """
    Parses IDD-Lite (India Driving Dataset - Lite).
    Extracts person/living-thing bounding boxes from gtFine segmentation masks
    using OpenCV connected components.
    Organized by sequence directory (e.g., train/0/, train/1/).
    """
    records: List[ImageRecord] = []
    idd_candidates = [
        raw_root / "idd20k_lite",
        raw_root / "idd-lite",
        PROJECT_ROOT / "datasets" / "idd20k_lite",
        PROJECT_ROOT / "datasets" / "idd-lite"
    ]
    idd_dir = next((p for p in idd_candidates if p.is_dir()), None)
    if not idd_dir:
        print("[WARN] IDD-Lite directory not found in raw data paths. Skipping IDD-Lite.")
        return records

    print(f"[INFO] Ingesting IDD-Lite from: {idd_dir}")

    for split in ["train", "val"]:
        img_split_dir = idd_dir / "leftImg8bit" / split
        gt_split_dir = idd_dir / "gtFine" / split
        if not img_split_dir.is_dir() or not gt_split_dir.is_dir():
            continue

        seq_dirs = [d for d in img_split_dir.iterdir() if d.is_dir()]

        for sdir in tqdm(seq_dirs, desc=f"Parsing IDD-Lite {split} sequences", unit="seq"):
            seq_id = f"iddlite_seq_{sdir.name}"
            for img_file in sdir.glob("*"):
                if not img_file.is_file() or img_file.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
                    continue

                base_stem = img_file.stem.replace("_image", "").replace("_leftImg8bit", "")
                label_candidates = [
                    gt_split_dir / sdir.name / f"{base_stem}_label.png",
                    gt_split_dir / sdir.name / f"{base_stem}_label.jpg",
                    gt_split_dir / sdir.name / f"{img_file.stem}_label.png"
                ]
                label_path = next((p for p in label_candidates if p.is_file()), None)
                if not label_path:
                    continue

                try:
                    mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        continue
                    h, w = mask.shape[:2]

                    record = ImageRecord(
                        image_id=f"iddlite_{split}_{sdir.name}_{base_stem}",
                        dataset_source="idd20k_lite",
                        sequence_id=seq_id,
                        image_path=img_file
                    )

                    # Extract class 2 (Living-thing: pedestrians / persons)
                    person_mask = (mask == 2).astype(np.uint8)
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(person_mask)

                    for i in range(1, num_labels):
                        area = stats[i, cv2.CC_STAT_AREA]
                        if area < 20:  # filter noise
                            continue
                        bx = stats[i, cv2.CC_STAT_LEFT]
                        by = stats[i, cv2.CC_STAT_TOP]
                        bw = stats[i, cv2.CC_STAT_WIDTH]
                        bh = stats[i, cv2.CC_STAT_HEIGHT]

                        xc = (bx + bw / 2.0) / w
                        yc = (by + bh / 2.0) / h
                        nw = bw / w
                        nh = bh / h

                        xc = max(0.0, min(1.0, xc))
                        yc = max(0.0, min(1.0, yc))
                        nw = max(0.0, min(1.0, nw))
                        nh = max(0.0, min(1.0, nh))

                        record.bboxes.append(BoundingBox(
                            class_id=CLASS_TO_IDX["person"],
                            class_name="person",
                            xc=xc, yc=yc, w=nw, h=nh
                        ))

                    records.append(record)

                except Exception:
                    continue

    print(f"[INFO] Ingested {len(records)} images from IDD-Lite.")
    return records


def ingest_roboflow_yolo(raw_root: Path, folder_name: str, mapping_func) -> List[ImageRecord]:
    """Ingests Roboflow datasets formatted with existing YOLO annotations."""
    records: List[ImageRecord] = []
    candidates = [
        raw_root / folder_name,
        PROJECT_ROOT / "datasets" / folder_name
    ]
    rf_dir = next((p for p in candidates if p.is_dir()), None)
    if not rf_dir:
        return records

    # Read class names from data.yaml
    data_yaml_path = rf_dir / "data.yaml"
    raw_names = []
    if data_yaml_path.is_file():
        try:
            with open(data_yaml_path, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f)
                names = y.get("names", [])
                if isinstance(names, dict):
                    raw_names = [names[k] for k in sorted(names.keys())]
                elif isinstance(names, list):
                    raw_names = names
        except Exception:
            pass

    for split in ["train", "valid", "val", "test"]:
        img_dir = rf_dir / split / "images"
        lbl_dir = rf_dir / split / "labels"
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            continue

        for img_file in img_dir.glob("*"):
            if not img_file.is_file() or img_file.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
                continue

            lbl_file = lbl_dir / f"{img_file.stem}.txt"
            if not lbl_file.is_file():
                continue

            # Group sequence by common prefix or hash block
            seq_prefix = img_file.stem.split("_")[0] if "_" in img_file.stem else img_file.stem[:6]
            sequence_id = f"{folder_name}_{seq_prefix}"

            record = ImageRecord(
                image_id=f"{folder_name}_{img_file.stem}",
                dataset_source=folder_name,
                sequence_id=sequence_id,
                image_path=img_file
            )

            try:
                with open(lbl_file, "r", encoding="utf-8") as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_idx = int(parts[0])
                            xc, yc, bw, bh = map(float, parts[1:5])
                            raw_cls_name = raw_names[cls_idx] if cls_idx < len(raw_names) else str(cls_idx)
                            unified_cls = mapping_func(raw_cls_name)
                            if unified_cls:
                                record.bboxes.append(BoundingBox(
                                    class_id=CLASS_TO_IDX[unified_cls],
                                    class_name=unified_cls,
                                    xc=xc, yc=yc, w=bw, h=bh
                                ))
                records.append(record)
            except Exception:
                continue

    return records


# ==============================================================================
# SEQUENCE-BASED PARTITIONING (80 / 10 / 10)
# ==============================================================================

def split_by_sequence(
    records: List[ImageRecord],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, List[ImageRecord]]:
    """
    Partitions records into train, val, and test strictly by sequence_id.
    Ensures ZERO image leakage from the same sequence across splits.
    """
    dataset_sequences: Dict[str, Set[str]] = defaultdict(set)
    for r in records:
        dataset_sequences[r.dataset_source].add(r.sequence_id)

    sequence_to_split: Dict[str, str] = {}
    rng = random.Random(seed)

    for d_source, seqs in dataset_sequences.items():
        sorted_seqs = sorted(list(seqs))
        rng.shuffle(sorted_seqs)

        n = len(sorted_seqs)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # Ensure at least 1 in val and test if dataset has multiple sequences
        if n >= 3:
            n_train = max(1, min(n - 2, n_train))
            n_val = max(1, min(n - n_train - 1, n_val))

        for seq in sorted_seqs[:n_train]:
            sequence_to_split[seq] = "train"
        for seq in sorted_seqs[n_train:n_train + n_val]:
            sequence_to_split[seq] = "val"
        for seq in sorted_seqs[n_train + n_val:]:
            sequence_to_split[seq] = "test"

    splits: Dict[str, List[ImageRecord]] = {"train": [], "val": [], "test": []}
    for r in records:
        target_split = sequence_to_split.get(r.sequence_id, "train")
        splits[target_split].append(r)

    return splits


# ==============================================================================
# EXPORT PROCESSOR
# ==============================================================================

def export_prepared_dataset(
    splits: Dict[str, List[ImageRecord]],
    processed_dir: Path
) -> None:
    """
    Links images and writes YOLOv8 normalized label txt files into data/processed/.
    Uses NTFS hard links (os.link) to consume 0 additional disk space.
    """
    for split in ["train", "val", "test"]:
        (processed_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (processed_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Exporting prepared dataset to: {processed_dir}")

    for split, record_list in splits.items():
        img_out_dir = processed_dir / "images" / split
        lbl_out_dir = processed_dir / "labels" / split

        for rec in tqdm(record_list, desc=f"Exporting {split} split", unit="img"):
            ext = rec.image_path.suffix.lower()
            dest_img = img_out_dir / f"{rec.image_id}{ext}"
            dest_lbl = lbl_out_dir / f"{rec.image_id}.txt"

            # Create image link
            if not dest_img.exists():
                try:
                    os.link(rec.image_path, dest_img)
                except Exception:
                    try:
                        shutil.copy2(rec.image_path, dest_img)
                    except Exception as e:
                        print(f"[ERROR] Failed to link/copy image {rec.image_path}: {e}")
                        continue

            # Write YOLOv8 label text file
            with open(dest_lbl, "w", encoding="utf-8") as lf:
                for b in rec.bboxes:
                    lf.write(b.to_yolo_line())


# ==============================================================================
# CONFIG GENERATORS: classes.yaml & data.yaml
# ==============================================================================

def write_classes_yaml(processed_dir: Path) -> Path:
    """Writes the unified classes definition and ontology mappings."""
    classes_file = processed_dir / "classes.yaml"

    content = {
        "unified_classes": UNIFIED_CLASSES,
        "class_indices": CLASS_TO_IDX,
        "source_mappings": {
            "IDD_FGVD": {
                "car_* / sedan_* / hatchback_* / suv_* / van_*": "car (4)",
                "motorcycle_* / scooter_* / moped_*": "motorcycle (2)",
                "autorickshaw_* / auto_* / rickshaw_*": "autorickshaw (3)",
                "bus / mini-bus_* / minibus_*": "bus (5)",
                "truck_* / lorry_* / tipper_*": "truck (6)",
                "bicycle_* / cycle_*": "bicycle (1)",
                "person_* / pedestrian_*": "person (0)"
            },
            "idd20k_lite": {
                "gtFine mask class 2 (Living-thing)": "person (0)"
            },
            "roboflow-pothole": {
                "Pothole / Drain Hole / Sewer Cover": "pothole (9)"
            },
            "roboflow-traffic-sign": {
                "TRAFFIC_SIGNAL": "traffic_light (8)",
                "BUS_STOP / PEDESTRIAN_CROSSING / warning / mandatory signs": "bus_stop_sign (7)"
            },
            "roboflow-bus-passenger": {
                "passenger / person / 5": "person (0)"
            },
            "self-collected": {
                "bus stop commuters / door passengers": "person (0)"
            }
        }
    }

    with open(classes_file, "w", encoding="utf-8") as f:
        yaml.dump(content, f, sort_keys=False, indent=2)

    print(f"[SUCCESS] Wrote unified classes mapping to: {classes_file}")
    return classes_file


def write_data_yaml(processed_dir: Path) -> Path:
    """Writes standard Ultralytics YOLOv8 data.yaml."""
    data_file = processed_dir / "data.yaml"

    # Ultralytics path formatted with forward slashes for universal compatibility
    abs_path_str = str(processed_dir.resolve()).replace("\\", "/")

    content = {
        "path": abs_path_str,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(UNIFIED_CLASSES),
        "names": {idx: name for idx, name in enumerate(UNIFIED_CLASSES)}
    }

    with open(data_file, "w", encoding="utf-8") as f:
        yaml.dump(content, f, sort_keys=False, indent=2)

    print(f"[SUCCESS] Wrote YOLOv8 training config to: {data_file}")
    return data_file


# ==============================================================================
# REPORTING: MAPPING TABLE & CLASS BALANCE REPORT
# ==============================================================================

def log_mapping_table() -> None:
    """Logs the ontology mapping table showing source class mapping into unified class."""
    print("\n" + "=" * 110)
    print(" UNIFIED CLASS ONTOLOGY MAPPING TABLE")
    print("=" * 110)
    header = f"{'ID':<4} {'Unified Class':<18} {'Source Dataset':<24} {'Source Category Pattern'}"
    print(header)
    print("-" * 110)

    mappings = [
        (0, "person", "idd20k_lite", "Living-thing (gtFine mask label == 2)"),
        (0, "person", "IDD_FGVD", "person_*, pedestrian_*"),
        (0, "person", "roboflow-bus-passenger", "passenger, person, 5"),
        (0, "person", "self-collected", "bus stop commuters, boarding passengers"),
        (1, "bicycle", "IDD_FGVD", "bicycle_*, cycle_*"),
        (2, "motorcycle", "IDD_FGVD", "motorcycle_*, scooter_*, moped_* (Hero, Bajaj, Honda, etc.)"),
        (3, "autorickshaw", "IDD_FGVD", "autorickshaw_*, auto_*, rickshaw_* (Bajaj, Piaggio, Atul)"),
        (4, "car", "IDD_FGVD", "car_*, sedan_*, hatchback_*, suv_*, van_* (Swift, Alto, etc.)"),
        (5, "bus", "IDD_FGVD", "bus, mini-bus_*, minibus_* (Tata, Ashok Leyland, etc.)"),
        (6, "truck", "IDD_FGVD", "truck_*, lorry_*, tipper_* (Tata, Eicher, BharatBenz)"),
        (7, "bus_stop_sign", "roboflow-traffic-sign", "BUS_STOP, PEDESTRIAN_CROSSING, STOP, signs"),
        (8, "traffic_light", "roboflow-traffic-sign", "TRAFFIC_SIGNAL, traffic lights"),
        (9, "pothole", "roboflow-pothole", "Pothole, Drain Hole, Sewer Cover"),
    ]

    for uid, ucls, sdata, spattern in mappings:
        print(f"{uid:<4} {ucls:<18} {sdata:<24} {spattern}")

    print("=" * 110 + "\n")


def print_class_balance_report(splits: Dict[str, List[ImageRecord]]) -> None:
    """
    Computes and prints a detailed class balance report (image count, instance count,
    train/val/test breakdown, proportion %, and representation status).
    """
    split_inst_counts: Dict[str, Counter] = {s: Counter() for s in splits}
    class_img_counts: Counter = Counter()
    total_inst_counts: Counter = Counter()

    total_images = sum(len(r_list) for r_list in splits.values())

    for split_name, records in splits.items():
        for r in records:
            seen_in_img = set()
            for b in r.bboxes:
                split_inst_counts[split_name][b.class_name] += 1
                total_inst_counts[b.class_name] += 1
                seen_in_img.add(b.class_name)
            for cname in seen_in_img:
                class_img_counts[cname] += 1

    total_instances = sum(total_inst_counts.values())

    print("\n" + "=" * 120)
    print(" CLASS BALANCE REPORT (DATASET READINESS & SKEW ANALYSIS)")
    print("=" * 120)
    header = (
        f"{'ID':<4} {'Class Name':<16} {'Train Inst':<12} {'Val Inst':<10} "
        f"{'Test Inst':<10} {'Total Inst':<12} {'Share %':<10} {'Images':<10} {'Status'}"
    )
    print(header)
    print("-" * 120)

    for idx, cname in enumerate(UNIFIED_CLASSES):
        tr_c = split_inst_counts["train"][cname]
        va_c = split_inst_counts["val"][cname]
        te_c = split_inst_counts["test"][cname]
        tot = total_inst_counts[cname]
        imgs = class_img_counts[cname]
        pct = (tot / total_instances * 100.0) if total_instances > 0 else 0.0

        if tot == 0:
            status = "[CRITICAL: ZERO INSTANCES]"
        elif pct < 1.0:
            status = "[UNDER-REPRESENTED (<1%)]"
        elif pct < 5.0:
            status = "[ADEQUATE]"
        else:
            status = "[BALANCED]"

        print(
            f"{idx:<4} {cname:<16} {tr_c:<12} {va_c:<10} {te_c:<10} "
            f"{tot:<12} {pct:>6.2f}%    {imgs:<10} {status}"
        )

    print("-" * 120)
    print(f"Total Dataset Images   : {total_images:,}")
    print(f"  - Train Images       : {len(splits['train']):,} ({len(splits['train'])/total_images*100:.1f}%)")
    print(f"  - Val Images         : {len(splits['val']):,} ({len(splits['val'])/total_images*100:.1f}%)")
    print(f"  - Test Images        : {len(splits['test']):,} ({len(splits['test'])/total_images*100:.1f}%)")
    print(f"Total Object Instances : {total_instances:,}")
    print(f"  - Train Instances    : {sum(split_inst_counts['train'].values()):,}")
    print(f"  - Val Instances      : {sum(split_inst_counts['val'].values()):,}")
    print(f"  - Test Instances     : {sum(split_inst_counts['test'].values()):,}")
    print("=" * 120 + "\n")


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Bus Perception System - Unified Dataset Preparation Pipeline"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Path to raw datasets directory (default: data/raw)"
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Path to processed output directory (default: data/processed)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
        help="Train split ratio (default: 0.80)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Validation split ratio (default: 0.10)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.10,
        help="Test split ratio (default: 0.10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sequence assignment (default: 42)"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean existing data/processed directory before preparation"
    )
    args = parser.parse_args()

    # Fallback to datasets/ if raw-dir doesn't exist
    raw_root = args.raw_dir
    if not raw_root.exists():
        fallback_root = PROJECT_ROOT / "datasets"
        if fallback_root.exists():
            print(f"[INFO] '{raw_root}' not found, falling back to '{fallback_root}'")
            raw_root = fallback_root
        else:
            print(f"[ERROR] Neither '{raw_root}' nor '{fallback_root}' exists.", file=sys.stderr)
            sys.exit(1)

    processed_dir = args.processed_dir

    if args.clean and processed_dir.exists():
        print(f"[INFO] Cleaning existing output directory: {processed_dir}")
        shutil.rmtree(processed_dir)

    processed_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(" 🚀 STARTING DATASET PREPARATION PIPELINE")
    print(f" Source Directory    : {raw_root}")
    print(f" Target Directory    : {processed_dir}")
    print(f" Target Classes      : {len(UNIFIED_CLASSES)} classes")
    print(f" Split Strategy      : By Driving Session/Sequence ({args.train_ratio*100:.0f}/{args.val_ratio*100:.0f}/{args.test_ratio*100:.0f})")
    print("=" * 80)

    # 1. Log Mapping Table
    log_mapping_table()

    # 2. Ingest Primary Datasets: FGVD and IDD-Lite
    all_records: List[ImageRecord] = []
    all_records.extend(ingest_fgvd(raw_root))
    all_records.extend(ingest_idd_lite(raw_root))

    # 3. Ingest Complementary Datasets for full class coverage
    pothole_records = ingest_roboflow_yolo(raw_root, "roboflow-pothole", map_roboflow_pothole)
    if pothole_records:
        print(f"[INFO] Ingested {len(pothole_records)} images from roboflow-pothole.")
        all_records.extend(pothole_records)

    sign_records = ingest_roboflow_yolo(raw_root, "roboflow-traffic-sign", map_roboflow_traffic_sign)
    if sign_records:
        print(f"[INFO] Ingested {len(sign_records)} images from roboflow-traffic-sign.")
        all_records.extend(sign_records)

    passenger_records = ingest_roboflow_yolo(raw_root, "roboflow-bus-passenger", lambda _: "person")
    if passenger_records:
        print(f"[INFO] Ingested {len(passenger_records)} images from roboflow-bus-passenger.")
        all_records.extend(passenger_records)

    if not all_records:
        print("[ERROR] No records were ingested. Please verify raw datasets.", file=sys.stderr)
        sys.exit(1)

    print(f"\n[INFO] Total ingested images across all sources: {len(all_records):,}")

    # 4. Partition by Source Session / Sequence (80/10/10)
    splits = split_by_sequence(
        all_records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    # 5. Export dataset files (images via zero-cost hardlinks + YOLOv8 txt labels)
    export_prepared_dataset(splits, processed_dir)

    # 6. Write classes.yaml & data.yaml
    write_classes_yaml(processed_dir)
    write_data_yaml(processed_dir)

    # 7. Print Class Balance Report
    print_class_balance_report(splits)

    print("[DONE] Dataset preparation complete.")


if __name__ == "__main__":
    main()
