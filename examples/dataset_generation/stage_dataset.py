"""Stage the dataset directory the way the HuggingFace layout expects it: one
folder per layer type holding ``shard_NNNN`` directories of hard links to the
combined shards written by ``combine_shards.py`` (or symlinks with ``--link symlink``), plus the dataset's README.md, DATASHEET.md and
representative_sample.ipynb from ``dataset_card/``.

    python stage_dataset.py --src $SCRATCH/resmill_dataset_win64 --dst $SCRATCH/SiliciclasticReservoirs

``build_splits.py --root <dst>`` then indexes ``<dst>/<layer type>/shard_NNNN``.
Existing links are replaced.
"""
import argparse
import os
import shutil
from pathlib import Path

# HF layer-type directory -> combined preset directory
MAP = {
    "lobe": "lobes_combined",
    "channel_pv_shoestring": "pv_shoestring_combined",
    "channel_cb_labyrinth": "cb_labyrinth_combined",
    "channel_cb_jigsaw": "cb_jigsaw_combined",
    "channel_sh_distal": "sh_distal_combined",
    "channel_sh_proximal": "sh_proximal_combined",
    "channel_meander_oxbow": "meander_oxbow_combined",
    "delta": "delta_combined",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--link", choices=("symlink", "hardlink"), default="hardlink",
                    help="shard_NNNN as a symlink to the combined shard dir, or a real dir of hard-linked files (default; every uploader sees plain files, no extra space)")
    ap.add_argument("--cards", default=str(Path(__file__).resolve().parent / "dataset_card"),
                    help="directory with the dataset's README.md, DATASHEET.md and notebook, copied to --dst (default: dataset_card/ next to this script)")
    args = ap.parse_args()
    src = Path(os.path.expandvars(os.path.expanduser(args.src))).resolve()
    dst = Path(os.path.expandvars(os.path.expanduser(args.dst)))
    total = 0
    for hf, combined in MAP.items():
        shards = sorted((src / combined).glob("combined_shard_*"))
        if not shards:
            print(f"  skip {hf}: no shards in {src / combined}")
            continue
        (dst / hf).mkdir(parents=True, exist_ok=True)
        for shard in shards:
            link = dst / hf / ("shard_" + shard.name.split("combined_shard_")[1])
            if link.is_symlink():
                link.unlink()
            elif link.is_dir():
                shutil.rmtree(link)
            if args.link == "symlink":
                link.symlink_to(shard)
            else:
                link.mkdir()
                for f in sorted(shard.iterdir()):
                    os.link(f, link / f.name)
        total += len(shards)
        print(f"  {hf:24s} {len(shards)} shards -> {dst / hf}")
    cards = Path(os.path.expandvars(os.path.expanduser(args.cards)))
    for name in ("README.md", "DATASHEET.md", "representative_sample.ipynb"):
        if (cards / name).is_file():
            shutil.copyfile(cards / name, dst / name)
            print(f"  copied {name}")
    print(f"  staged {total} shards under {dst}")


if __name__ == "__main__":
    main()
