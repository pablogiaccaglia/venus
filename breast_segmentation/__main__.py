"""Main entry point for the breast segmentation package."""

import sys
from .train import main, parse_arguments

if __name__ == "__main__":
    args = parse_arguments()
    sys.exit(main(args))
