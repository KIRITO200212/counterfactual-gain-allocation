#!/usr/bin/env python3
"""Generate frozen bridge X images from separately acquired source images."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from PIL import Image
from regcfpo.operators.pixel_ops import pixel_pair_slot_swap

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,default=ROOT/"data/splits/bridge_train21_schedule80.jsonl")
    a=p.parse_args()
    done={}
    for row in map(json.loads,a.manifest.read_text().splitlines()):
        if row["pairaug_branch"]!="swap":continue
        dst=ROOT/row["image"]
        if dst in done:continue
        source=ROOT/row["pairaug_source_image"]
        with Image.open(source) as im:
            result=pixel_pair_slot_swap(im.convert("RGB"),row["gt_box_a"],row["gt_box_b"])
        if not result.accepted:
            raise RuntimeError(f'{row["sample_id"]}: {result.reject_reason}')
        dst.parent.mkdir(parents=True,exist_ok=True)
        result.image.save(dst)
        done[dst]=row["sample_id"]
    print(json.dumps({"generated_images":len(done)}))
if __name__=="__main__":
    main()
