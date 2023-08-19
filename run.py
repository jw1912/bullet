import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="NNUE Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--data-path',
        type=str,
        help="Path to data file.",
        required=True
    )

    parser.add_argument(
        '--threads',
        type=int,
        help="Number of threads to run on.",
        default=1
    )

    parser.add_argument(
        '--lr',
        type=float,
        help="The starting learning rate.",
        default=0.001
    )

    parser.add_argument(
        '--wdl',
        type=float,
        help="The weighting of WDL versus score, 1.0 is fully WDL.",
        default=0.5
    )

    parser.add_argument(
        '--max-epochs',
        type=int,
        help="Number of epochs for which the trainer will run.",
        default=65
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        help="Size of each batch.",
        default=16384
    )

    parser.add_argument(
        '--save-rate',
        type=int,
        help="The number of epochs between saving the network.",
        default=10
    )

    parser.add_argument(
        '--test-id',
        type=str,
        help="Test ID, will be the name of the produced network.",
        default="net"
    )

    args = parser.parse_args()

    if args.data_path is None:
        print("No path to data provided!")
        return

    commands = [
        "cargo",
        "run",
        "--release",
        "--bin",
        "trainer",
        args.data_path,
        str(args.threads),
        str(args.lr),
        str(args.wdl),
        str(args.max_epochs),
        str(args.batch_size),
        str(args.save_rate),
        args.test_id,
    ]

    subprocess.run(commands)


if __name__ == "__main__":
    main()
