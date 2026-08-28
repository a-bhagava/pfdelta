import os
import glob
import json
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
    description="Plot error values for a specific run and error key."
)
parser.add_argument(
    "--run_name",
    type=str,
    required=True,
    help="Name of the run to plot errors for."
)
parser.add_argument(
    "--error",
    type=str,
    default="_first_one",
    help="Error key being plotted, from train.json (and val.json too, "
    "unless --val_error is given separately)."
)
parser.add_argument(
    "--val_error",
    type=str,
    default=None,
    help="Error key to plot from val.json, if different from --error -- "
    "train_loss and val_loss don't always share the same names for the "
    "same underlying quantity (e.g. a train_loss entry wrapped in "
    "combined_loss gets a concatenated name, while the matching val_loss "
    "entry may be reported standalone). '_first_one' resolves to val.json's "
    "own first key, same convention as --error's default. Defaults to "
    "whatever --error resolves to, i.e. the old single-key behavior."
)
parser.add_argument(
    "--log",
    action="store_true",
    default=False,
    help="Change scale to log."
)
parser.add_argument(
    "--curve",
    type=str,
    nargs="+",
    default=["all"],
    help="Specifies which learning curve to plot."
)
args = parser.parse_args()


def find_run_folder(run_name):
    """Searches for the folder in 'runs' using glob and ensures the name is
    unique."""
    # Assuming the root folder is "runs"
    run_pattern = os.path.join("runs", "**", run_name)
    matching_folders = glob.glob(run_pattern, recursive=True)

    if len(matching_folders) == 0:
        print(f"Error: No folder named '{run_name}' found in 'runs'.")
        return None
    elif len(matching_folders) > 1:
        print(
            f"Error: Multiple folders with the name '{run_name}' found."
            + "Please ensure the name is unique."
        )
        return None
    else:
        return matching_folders[0]  # Return the unique matching folder


def plot_errors(run_folder, error_key, val_error_key=None, curve=["all"]):
    """Loads the train.json file and plots the errors for the given run and
    error key. `val_error_key` (defaulting to `error_key`, i.e. the old
    single-key behavior) lets val.json be plotted under a different key --
    train_loss/val_loss don't always name the same underlying quantity the
    same way (e.g. a train_loss entry wrapped in combined_loss gets a
    concatenated name -- see CombinedLoss.loss_name -- while the matching
    val_loss entry may be reported standalone)."""
    max_ticks = 15

    # Build the path to the train.json file
    train_path = os.path.join(run_folder, "train.json")
    with open(train_path, "r") as f:
        train_data = json.load(f)

    val_path = os.path.join(run_folder, "val.json")
    with open(val_path, "r") as f:
        val_data = json.load(f)

    summary_path = os.path.join(run_folder, "summary.json")
    with open(summary_path, "r") as f:
        summary = json.load(f)

    if error_key == "_first_one":
        first_epoch = list(train_data.keys())[0]
        error_key = list(train_data[first_epoch].keys())[0]

    if val_error_key is None:
        val_error_key = error_key
    elif val_error_key == "_first_one":
        first_val_epoch = sorted(val_data.keys(), key=int)[0]
        val_error_key = list(val_data[first_val_epoch][0].keys())[0]

    # Set up figure
    plt.figure(figsize=(10, 6))
    plt.xlabel("Training point")
    ylabel = error_key if val_error_key == error_key else f"{error_key} (train) / {val_error_key} (val)"
    plt.ylabel(ylabel)
    plt.title(f"{run_name} - {ylabel}")

    # Highlight the best epoch
    best_point = summary["best_point"]
    plt.axvline(
        x=int(best_point),
        color='red',
        linestyle='--',
        linewidth=1,
        label="Best point"
    )

    ## Train values
    # Extract errors for each epoch
    epochs = sorted(train_data.keys(), key=int)  # Sort epochs numerically
    highest_epoch = int(epochs[-1]) + 1
    if len(epochs) > max_ticks:
        right_gaps = len(epochs) // max_ticks
    else:
        right_gaps = 1
    epochs = epochs[::right_gaps] + [epochs[-1]]
    # Plot train errors
    errors = [train_data[epoch].get(error_key, None) for epoch in epochs]
    print(epochs)
    print_epochs = list(map(int, epochs))
    if "all" in curve or "train" in curve:
        plt.plot(
            print_epochs,
            errors,
            marker="o",
            linestyle="-",
            color="b",
            label="Train"
        )

    ## Val values
    epochs = sorted(val_data.keys(), key=int)  # Sort epochs numerically
    highest_epoch = int(epochs[-1]) + 1
    if len(epochs) > max_ticks:
        right_gaps = len(epochs) // max_ticks
    else:
        right_gaps = 1
    epochs = epochs[::right_gaps] + [epochs[-1]]
    # Plot val errors
    num_vals = len(val_data[epochs[0]])
    for i in range(num_vals):
        errors = [val_data[epoch][i].get(val_error_key, None) for epoch in epochs]
        print_epochs = list(map(int, epochs))
        if "all" in curve or f"val_{i}" in curve:
            plt.plot(
                print_epochs,
                errors,
                marker="o",
                linestyle="-",
                label=f"Val {i}"
            )

    # Change to logscale if needed
    if args.log:
        plt.yscale("log")

    # Print summary
    print(json.dumps(summary, indent=3))

    plt.legend()
    plt.show()
    plt.savefig("plot_results.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    # Gather arguments
    run_name = args.run_name
    error_name = args.error
    val_error_name = args.val_error
    curve = args.curve

    # Find path to run
    run_path = find_run_folder(run_name)
    assert run_path is not None, "Run not found!"
    print(f"Run name found in path: {run_path}")

    # Plot the errors for the given run and error key
    plot_errors(run_path, error_name, val_error_name, curve)
