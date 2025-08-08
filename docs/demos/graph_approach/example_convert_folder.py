import argparse, json
from schemas import ConvertConfig
from dataset import convert_folder_to_pt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        raw = json.load(f)
    cfg = ConvertConfig(**raw)
    out = convert_folder_to_pt(cfg)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
