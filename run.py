import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Neural Network Trainer",
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
        help="The starting learning rate (LR).",
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

    parser.add_argument(
        '--skip-prop',
        type=float,
        help="Proportion of fens skipped each epoch.",
        default=0.0
    )

    parser.add_argument(
        '--lr-end',
        type=float,
        help="Ending value of lR for exponential LR decay.",
        default=0.0
    )

    parser.add_argument(
        '--lr-step',
        type=int,
        help="Drop LR every given epochs.",
        default=0
    )

    parser.add_argument(
        '--lr-drop',
        type=int,
        help="Drop LR once, at given epoch.",
        default=0
    )

    parser.add_argument(
        '--lr-gamma',
        type=float,
        help="Factor to drop LR by for `drop` and `step` LR scheduling.",
        default=0.1
    )

    parser.add_argument(
        '--scale',
        type=int,
        help="Eval scale.",
        default=400
    )

    parser.add_argument(
        '--cbcs',
        help="Alternate colour scheme.",
        action="store_true",
    )

    args = parser.parse_args()

    if args.data_path is None:
        print("No path to data provided!")
        return

    try:
        os.mkdir("nets")
    except FileExistsError:
        pass
    except OSError as error:
        print(error)
        return

    commands = [
        "cargo",
        "rustc",
        "--release",
        "--bin",
        "trainer",
        "--",
        "-C",
        "target-cpu=native",
    ]

    subprocess.run(commands)

    exe_path = "target/release/trainer"
    if os.name == 'nt':
        exe_path += ".exe"

    commands = [
        exe_path,
        args.data_path,
        args.test_id,
        str(args.threads),
        str(args.lr),
        str(args.wdl),
        str(args.max_epochs),
        str(args.batch_size),
        str(args.save_rate),
        str(args.skip_prop),
        str(args.lr_end),
        str(args.lr_step),
        str(args.lr_drop),
        str(args.lr_gamma),
        str(args.scale),
        str(args.cbcs).lower(),
    ]

    subprocess.run(commands)


if __name__ == "__main__":
    main()
