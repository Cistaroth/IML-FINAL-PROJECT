from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.markdown import Markdown

from read_data import read_raw_data_files, read_show_mapping

console = Console()


def user_selection(df: pd.DataFrame, target=3, alpha=0.25) -> int:
    """
    Select the user with the highest number of ratings with a mean rating
    within range of target +/- alpha

    Args:
        df (pd.DataFrame): The dataframe of the data
        target (int, optional): The target rating. Defaults to 3.
        alpha (float, optional): The alpha value. Defaults to 0.25.

    Returns:
        int: The user id
    """

    console.print("\n[bold white]Selecting User...[/bold white]")

    user_statistics = df.groupby("user_id")["rating"].agg(["mean", "count"])

    lower_bound, upper_bound = target - alpha, target + alpha
    mask = (user_statistics["mean"] >= lower_bound) & \
        (user_statistics["mean"] <= upper_bound)

    filtered_users = user_statistics[mask]

    if filtered_users.empty:
        raise Exception("No users found with the given target and alpha.")

    target_user_id = filtered_users["count"].idxmax()
    target_user_mean = filtered_users["mean"].loc[target_user_id]
    target_user_count = filtered_users["count"].loc[target_user_id]

    console.print(f"  [bold cyan]{'User ID':<15}:[/bold cyan] \
                   [white]{target_user_id}[/white]")
    console.print(f"  [bold green]{'Mean Rating':<15}:[/bold green] \
                   [white]{target_user_mean:.2f}[/white]")
    console.print(f"  [bold yellow]{'Total Ratings':<15}:[/bold yellow] \
                   [white]{target_user_count}[/white]")
    console.print()

    return int(target_user_id)


def main(data_files: list[Path],
         show_mapping_file_path: Path, save_path: Path) -> None:
    """
    Main function

    Args:
        data_files (list[Path]): The paths to the files
        show_mapping_file_path (Path): The path to the file
    """

    console.print(Markdown("# Extracting User Data"))

    data = read_raw_data_files(file_paths=data_files)
    show_mapping = read_show_mapping(file_path=show_mapping_file_path)

    user_id = user_selection(df=data, target=3, alpha=0.25)
    user_data = data[data["user_id"] == user_id].copy()

    user_data.rename({"show_id": "title"}, axis=1, inplace=True)

    user_data["title"] = user_data["title"].map(show_mapping)

    user_data.drop("user_id", axis=1, inplace=True)

    user_data.to_csv(save_path, index=False, quotechar='"', quoting=1,
                     doublequote=True, escapechar='\\')

    console.print(f"[bold green]✔[/bold green] Raw userdata saved to \
                  {save_path.name}")


if __name__ == "__main__":
    DATA_FILES = [
        Path(__file__).parent.parent / "raw_data" / "combined_data_1.txt",
        Path(__file__).parent.parent / "raw_data" / "combined_data_2.txt",
        Path(__file__).parent.parent / "raw_data" / "combined_data_3.txt",
        Path(__file__).parent.parent / "raw_data" / "combined_data_4.txt",
    ]
    SHOW_MAPPING_FILE_PATH = Path(__file__).parent.parent / "raw_data" / \
        "movie_titles.csv"

    SAVE_PATH = Path(__file__).parent.parent / "processed_data" / \
        "userdata_raw.csv"

    main(data_files=DATA_FILES, show_mapping_file_path=SHOW_MAPPING_FILE_PATH,
         save_path=SAVE_PATH)
