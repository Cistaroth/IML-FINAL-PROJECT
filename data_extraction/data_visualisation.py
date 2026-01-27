from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from read_data import read_raw_data_files
from rich.console import Console
from rich.markdown import Markdown

console = Console()

# Source - https://stackoverflow.com/a
# Posted by Joe Kington, modified by community.
# See post 'Timeline' for change history
# Retrieved 2025-12-25, License - CC BY-SA 3.0


def is_outlier(points: np.ndarray, thresh: float = 3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Args:
        points(np.ndarray): An numobservations by
            numdimensions array of observations
        thresh(float): The modified z-score to use as a threshold.
            Observations with a modified z-score
            (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
        mask(np.ndarray): A numobservations-length boolean array.

    References:
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """

    if len(points.shape) == 1:
        points = points[:, None]

    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def histogram_plot(
        data: pd.Series,
        title: str,
        xlabel: str,
        y_label: str,
        save_path: str | Path,
        n_bins: int = 30) -> None:

    """
    Plot a histogram of a pandas Series.

    Args:
        data (pd.Series): The data to plot
        title (str): The title of the plot
        xlabel (str): The label for the x-axis
        y_label (str): The label for the y-axis
        save_path (str | Path): The path to save the plot
        n_bins (int, optional): The number of bins to use. Defaults to 30.

    Returns:
        None
    """
    plt.clf()
    plt.figure(figsize=(10, 6))

    hist_range = (max(min(data), 0), max(max(data), 5))
    counts, bin_edges = np.histogram(data, bins=n_bins,
                                     range=hist_range)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = (bin_edges[1] - bin_edges[0]) * 0.8

    bars = plt.bar(bin_centres, counts, width=width, color="#3c7ee7",
                   alpha=0.8)

    padding = max(counts) * 0.02
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + padding,
                "",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#555555"
            )

    plt.title(title, fontsize=16, loc="center", pad=15)
    plt.xlabel(xlabel=xlabel, fontsize=12)
    plt.ylabel(ylabel=y_label, fontsize=12)

    plt.gca().spines[["right", "top"]].set_visible(False)
    plt.grid(axis="y", alpha=0.5, linestyle=":")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def ratings_plot(
        data: pd.Series,
        title: str,
        save_path: str | Path) -> None:

    """
    Plot a horizontal bar chart of a pandas Series.

    Args:
        data (pd.Series): The data to plot
        title (str): The title of the plot
        save_path (str | Path): The path to save the plot

    Returns:
        None
    """

    plt.clf()
    plt.figure(figsize=(9, 5))

    counts = data.value_counts().sort_index()
    y_pos = counts.index
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]

    bars = plt.barh(y_pos, counts, color=colors, alpha=0.7)
    plt.title(title, fontsize=16, loc="center", pad=15)
    plt.yticks(y_pos, [str(i) + "/5 rating" for i in y_pos])
    plt.xticks([])

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    for bar in bars:
        width = bar.get_width()
        height = bar.get_height()
        plt.text(
            width + 100,
            bar.get_y() + height / 2,
            f"{width}",
            va="center",
            fontsize=11,
            color="#555555"
        )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def dataset_visualisation(
        data_files: list[Path],
        save_path: Path) -> pd.DataFrame:
    """
    Visualise the overall dataset, including the user rating distribution,
    show rating distribution, user rating count distribution, show rating count
    distribution, overall rating distribution, and dataset statistics.

    Args:
        data_files (list[Path]): The paths to the files
        save_path (Path): The path to save the plots

    Returns:
        pd.DataFrame: The dataframe of the data
    """
    console.print("\n[bold white]Visualising Overall Dataset...[/bold white]")
    data = read_raw_data_files(data_files)

    # User rating distribution
    histogram_plot(
        data=data.groupby("user_id")["rating"].mean(),
        title="User Rating Distribution over All Users",
        xlabel="Average User Rating",
        y_label="Number of Users",
        save_path=save_path / "users_rating_distribution.png"
    )

    # Show rating distribution
    histogram_plot(
        data=data.groupby("show_id")["rating"].mean(),
        title="Show Rating Distribution over All Shows",
        xlabel="Average Show Rating",
        y_label="Number of Shows",
        save_path=save_path / "shows_rating_distribution.png"
    )

    # User rating count distribution
    user_rating_count = data.groupby("user_id").size().to_numpy()
    user_rating_count = user_rating_count[~is_outlier(user_rating_count)]
    histogram_plot(
        data=pd.Series(user_rating_count),
        title="User Rating Count Distribution over All Users " +
        "(Excluding Outliers)",
        xlabel="User Rating Count",
        y_label="Number of Users",
        save_path=save_path / "users_rating_count_distribution.png"
    )

    # Show rating count distribution
    show_rating_count = data.groupby("show_id").size().to_numpy()
    show_rating_count = show_rating_count[~is_outlier(show_rating_count)]
    histogram_plot(
        data=pd.Series(show_rating_count),
        title="Show Rating Count Distribution over All Shows " +
        "(Excluding Outliers)",
        xlabel="Show Rating Count",
        y_label="Number of Shows",
        save_path=save_path / "shows_rating_count_distribution.png"
    )

    # Overall rating distribution
    ratings_plot(
        data=data["rating"],
        title="Rating Distribution Breakdown for All Users",
        save_path=save_path / "dataset_rating_distribution.png"
    )

    # Collect dataset statistics
    user_count = data["user_id"].nunique()
    show_count = data["show_id"].nunique()

    rating_mean = data["rating"].mean()
    rating_std = data["rating"].std()
    rating_count = data["rating"].count()

    # Print dataset statistics
    console.print()
    console.print(f"  [bold cyan]{'Unique Users':<15}:[/bold cyan] \
                   [white]{user_count}[/white]")
    console.print(f"  [bold magenta]{'Unique Shows':<15}:[/bold magenta] \
                   [white]{show_count}[/white]")

    console.print(f"  [bold cyan]{'Mean Rating':<15}:[/bold cyan] \
                   [white]{rating_mean:.2f}[/white]")
    console.print(f"  [bold green]{'Std Rating':<15}:[/bold green] \
                   [white]{rating_std:.2f}[/white]")
    console.print(f"  [bold yellow]{'Total Ratings':<15}:[/bold yellow] \
                   [white]{rating_count}[/white]")
    console.print()

    return data


def user_visualisation(data_file_path: Path, save_path: Path) -> pd.DataFrame:
    """
    Visualises the user data for a selected user.

    Args:
        data_file_path (Path): The path to the file containing the user data
        save_path (Path): The path to the directory where the plots are saved

    Returns:
        pd.DataFrame: The dataframe of the user data
    """
    console.print("\n[bold white]Visualising User Data...[/bold white]")

    data = pd.read_csv(
        data_file_path,
        quotechar='"',
        escapechar='\\',
        doublequote=True
    )

    # Rating distribution
    ratings_plot(
        data=data["rating"],
        title="Rating Distribution Breakdown for Selected User",
        save_path=save_path / "selected_user_rating_distribution.png"
    )

    # Collect user statistics
    target_user_mean = data["rating"].mean()
    target_user_std = data["rating"].std()
    target_user_count = data["rating"].count()

    # Print user statistics
    console.print(f"  [bold cyan]{'Mean Rating':<15}:[/bold cyan] \
                   [white]{target_user_mean:.2f}[/white]")
    console.print(f"  [bold green]{'Std Rating':<15}:[/bold green] \
                   [white]{target_user_std:.2f}[/white]")
    console.print(f"  [bold yellow]{'Total Ratings':<15}:[/bold yellow] \
                   [white]{target_user_count}[/white]")
    console.print()

    return data


def user_selection_data_visualisation(
        dataset: pd.DataFrame,
        user_data: pd.DataFrame,
        target=3,
        alpha=0.25) -> None:
    """
    Visualises the user selection data.

    Args:
        dataset (pd.DataFrame): The dataframe of the dataset
        user_data (pd.DataFrame): The dataframe of the user data
        target (int): The target rating
        alpha (float): The alpha value

    Returns:
        None
    """
    console.print("\n[bold white]Visualising User Selection...[/bold white]")

    # All possible candidates
    user_statistics = dataset.groupby("user_id")["rating"].agg(
        ["mean", "count", "std"])

    lower_bound, upper_bound = target - alpha, target + alpha
    mask = (user_statistics["mean"] >= lower_bound) & \
        (user_statistics["mean"] <= upper_bound)

    filtered_users = user_statistics[mask]

    if filtered_users.empty:
        raise Exception("No users found with the given target and alpha.")

    console.print(f"  [bold cyan]{'Possible Users':<15}:[/bold cyan] \
                   [white]{filtered_users.shape[0]}[/white]")

    # Specific candidate
    target_user_id = filtered_users["count"].idxmax()
    target_user_mean = filtered_users["mean"].loc[target_user_id]
    target_user_std = filtered_users["std"].loc[target_user_id]
    target_user_count = filtered_users["count"].loc[target_user_id]

    console.print(f"  [bold cyan]{'User ID':<15}:[/bold cyan] \
                   [white]{target_user_id}[/white]")
    console.print(f"  [bold green]{'Mean Rating':<15}:[/bold green] \
                   [white]{target_user_mean:.2f}[/white]")
    console.print(f"  [bold yellow]{'Std Rating':<15}:[/bold yellow] \
                   [white]{target_user_std:.2f}[/white]")
    console.print(f"  [bold cyan]{'Total Ratings':<15}:[/bold cyan] \
                   [white]{target_user_count}[/white]")
    console.print()

    # Calculate data loss due to loss metadata
    data_loss = target_user_count - user_data.shape[0]
    data_loss_ratio = data_loss / target_user_count

    console.print(f"  [bold cyan]{'Data Loss':<15}:[/bold cyan] \
                   [white]{data_loss}[/white]")
    console.print(f"  [bold green]{'Data Loss Ratio':<15}:[/bold green] \
                   [white]{data_loss_ratio:.2f}[/white]")
    console.print()


def main(
        raw_data_files_paths: list[Path],
        user_data_file_path: Path,
        save_path: Path) -> None:
    """
    Main function

    Args:
        raw_data_files_paths (list[Path]): The paths to the raw data files
        user_data_file_path (Path): The path to the user data file
        save_path (Path): The path to the directory where the plots are saved

    Returns:
        None
    """
    console.print(Markdown("# Extracting User Data"))

    dataset = dataset_visualisation(
        data_files=raw_data_files_paths,
        save_path=save_path
    )

    user_data = user_visualisation(data_file_path=user_data_file_path,
                                   save_path=save_path)

    user_selection_data_visualisation(
        dataset=dataset,
        user_data=user_data
    )


if __name__ == "__main__":
    RAW_DATA_FILES = [
        Path(__file__).parent.parent / "raw_data" / "combined_data_1.txt",
        Path(__file__).parent.parent / "raw_data" / "combined_data_2.txt",
        Path(__file__).parent.parent / "raw_data" / "combined_data_3.txt",
        Path(__file__).parent.parent / "raw_data" / "combined_data_4.txt",
    ]

    USER_DATA_PATH = Path(__file__).parent.parent / "processed_data" / \
        "userdata_metadata.csv"

    SAVE_PATH = Path(__file__).parent.parent / "graphs"

    main(raw_data_files_paths=RAW_DATA_FILES,
         user_data_file_path=USER_DATA_PATH,
         save_path=SAVE_PATH)
