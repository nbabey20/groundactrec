#!/usr/bin/env python3
import csv
from pathlib import Path

# Corrected UCF-19-Y-OCC mapping
UCF19_CLASS_INDICES = {
    4: "BandMarching",
    6: "BenchPress",
    10: "Biking",
    11: "BabyCrawling",
    27: "Drumming",
    37: "HandstandPushups",
    49: "Kayaking",
    55: "MoppingFloor",
    56: "Nunchucks",
    58: "PizzaTossing",
    59: "PlayingCello",
    62: "PlayingFlute",
    72: "PushUps",
    80: "SkateBoarding",
    81: "Skiing",
    84: "SoccerJuggling",
    85: "SoccerPenalty",
    88: "Surfing",
    98: "WalkingWithDog",
}

# Paths
ROOT = Path("/workspace")
UCF_VJEPA = ROOT / "results/ucf_vjepa"
UCF_COMOTION = ROOT / "results/comotion_ucf"
TRAIN_LIST = ROOT / "lists/trainlist01.txt"
TEST_LIST = ROOT / "lists/testlist01.txt"
OUT_DIR = ROOT / "datasets/ucf_indices"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def make_index_file(list_file: Path, out_csv: Path):
    rows = []
    with open(list_file, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    for ln in lines:
        # UCF-101 format: ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi
        action = ln.split("/")[0]
        stem = Path(ln).stem

        path_vjepa = UCF_VJEPA / "UCF-101" / action / f"{stem}.pt"
        path_skel = UCF_COMOTION / "ucf101/UCF-101" / action / f"{stem}.pt"

        rows.append({
            "id": stem,
            "label": action,
            "path_vjepa": str(path_vjepa),
            "path_skel": str(path_skel),
        })

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "path_vjepa", "path_skel"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"✔ wrote {out_csv} with {len(rows)} entries")


if __name__ == "__main__":
    make_index_file(TRAIN_LIST, OUT_DIR / "ucf_train.csv")
    make_index_file(TEST_LIST, OUT_DIR / "ucf_val.csv")

    # Extra validation split for UCF-19-Y-OCC
    occ_rows = []
    occ_root_vjepa = UCF_VJEPA / "occlusion_extracted"
    occ_root_skel = UCF_COMOTION / "occlusion_extracted"

    for idx, action in UCF19_CLASS_INDICES.items():
        folder = str(idx)
        for pt_file in sorted((occ_root_vjepa / folder).glob("*.pt")):
            stem = pt_file.stem
            path_vjepa = pt_file
            path_skel = occ_root_skel / folder / f"{stem}.pt"
            occ_rows.append({
                "id": stem,
                "label": action,
                "path_vjepa": str(path_vjepa),
                "path_skel": str(path_skel),
            })

    out_occ_csv = OUT_DIR / "ucf19_occ_val.csv"
    with open(out_occ_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "path_vjepa", "path_skel"])
        writer.writeheader()
        writer.writerows(occ_rows)
    print(f"✔ wrote {out_occ_csv} with {len(occ_rows)} entries")
