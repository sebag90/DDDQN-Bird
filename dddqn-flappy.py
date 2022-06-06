import importlib

from utils.cli import get_arguments


def main():
    args = get_arguments()
    subparser = args.subparser
    module = importlib.import_module(f"bin.{subparser}")
    module.main(args)


if __name__ == "__main__":
    main()
