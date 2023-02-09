import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser")

    # PLAY
    play = subparsers.add_parser(
        "play",
        help="play flappy bird"
    )
    play.add_argument(
        "--fps", type=int, default=30, help="FPS for the game"
    )

    # TRAIN
    train = subparsers.add_parser(
        "train",
        help="train a model to play flappy bird"
    )

    train.add_argument(
        "--episodes", type=int, default=100000, help="Default: %(default)s"
    )

    train.add_argument(
        "--init-epsilon",
        type=float,
        default=0.9,
        help=(
            "Starting epsilon. "
            "Tradeoff between exploration (high) and exploitation (low). "
            "Default: %(default)s"
        ),
    )
    train.add_argument(
        "--final-epsilon",
        type=float,
        default=0.5,
        help=(
            "Final epsilon. "
            "Tradeoff between exploration (high) and exploitation (low). "
            "Default: %(default)s"
        ),
    )
    train.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help=(
            "Discount factor of future reward "
            "(1: the agent values future reward as much as immediate reward). "
            "Default: %(default)s"
        ),
    )
    train.add_argument(
        "--lr",
        type=float,
        default=0.00001,
        help="learning rate for the neural network",
    )
    train.add_argument(
        "--checkpoint",
        action="store",
        type=int,
        default=5000,
        help="save a checkpoint every X episodes",
    )
    train.add_argument(
        "--init",
        action="store",
        type=float,
        default=0.01,
        help="value for uniform initialization",
    )
    train.add_argument(
        "--fps", type=int, default=30, help="FPS for the game"
    )
    train.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="print update every X steps",
    )
    train.add_argument(
        "--update-network",
        type=int,
        default=1000,
        help="update value network every X steps",
    )
    train.add_argument(
        "--resume", action="store", help="path to the level file"
    )
    train.add_argument(
        "--buffer-size", action="store", type=int, default=10000,
        help="Size of the replay buffer"
    )
    train.add_argument(
        "--batch-size", action="store", type=int, default=64,
        help="Batch size for learning"
    )
    train.add_argument("--headless", action="store_true")

    # EVALUATE
    evaluate = subparsers.add_parser(
        "evaluate",
        help="evaluate a trained model"
    )
    evaluate.add_argument(
        "file", action="store", help="path to the level file"
    )
    evaluate.add_argument(
        "--fps", type=int, default=30, help="FPS for the game"
    )

    args = parser.parse_args()
    if args.subparser not in subparsers.choices.keys():
        parser.print_help()
        return
    return args
