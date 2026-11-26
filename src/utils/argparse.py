import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg",
        type=str,
        default="config.yaml",
        help="Config yaml path"
    )

    return parser.parse_args()
