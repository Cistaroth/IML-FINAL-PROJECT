import numpy as np
import pandas as pd
from pathlib import Path
from rich.progress import track
from rich.console import Console
from rich.markdown import Markdown

console = Console()


def count_lines_file(file_path: Path) -> int:
    with open(file_path, "r", errors="ignore") as file:
        lines = len(file.readlines())

    return lines


def read_raw_data_files(file_paths: list[Path]) -> pd.DataFrame:
    """
    Read the raw data from the files

    Args:
        file_paths (list[Path]): The paths to the files

    Returns:
        pd.DataFrame: The dataframe of the data
    """
    console.print("\n[bold white]Reading Data...[/bold white]")

    data = []
    for file_path in file_paths:
        show_ids, user_ids, ratings = [], [], []

        with open(file_path, "r") as file:
            current_show = None

            for line in track(
                        file,
                        total=count_lines_file(file_path),
                        description=f"Reading {file_path.name}"):
                line = line.strip()

                if line.endswith(":"):
                    current_show = int(line[:-1])
                else:
                    user_id, rating, _ = line.split(",")

                    show_ids.append(current_show)
                    user_ids.append(int(user_id))
                    ratings.append(int(rating))

        data_chunk = pd.DataFrame({
            "show_id": np.array(show_ids, dtype=np.int32),
            "user_id": np.array(user_ids, dtype=np.int32),
            "rating": np.array(ratings, dtype=np.int8)
        })

        data.append(data_chunk)

    return pd.concat(data, ignore_index=True)


def read_show_mapping(file_path: Path) -> dict:
    """
    Read the show mapping from the file

    Args:
        file (Path): The path to the file

    Returns:
        dict: The dictionary of the data
    """

    console.print("\n[bold white]Reading Show Mapping..[/bold white]")
    mapping = {}

    with open(file_path, "r", encoding="latin-1") as file:
        for line in track(file, total=count_lines_file(file_path),
                          description=f"Reading {file_path.name}"):
            line = line.strip()

            show_id, _, show_title = line.split(",", maxsplit=2)
            show_id = int(show_id.rstrip())
            show_title = show_title.lstrip()

            mapping[show_id] = show_title

    return mapping


def main(data_files: list[Path],
         show_mapping_file_path: Path) -> tuple[pd.DataFrame, dict]:
    """
    Main function

    Args:
        data_files (list[Path]): The paths to the files
        show_mapping_file_path (Path): The path to the file

    Returns:
        tuple[pd.DataFrame, dict]: The dataframe of the data and
            the dictionary of the data
    """
    console.print(Markdown("# Reading Data"))
    data = read_raw_data_files(file_paths=data_files)
    show_mapping = read_show_mapping(file_path=show_mapping_file_path)

    return data, show_mapping


if __name__ == "__main__":
    DATA_FILES = [
        Path(__file__).parent.parent / "raw_data/combined_data_1.txt",
        Path(__file__).parent.parent / "raw_data/combined_data_2.txt",
        Path(__file__).parent.parent / "raw_data/combined_data_3.txt",
        Path(__file__).parent.parent / "raw_data/combined_data_4.txt",
    ]
    SHOW_MAPPING_FILE_PATH = Path(__file__).parent.parent / "raw_data" / \
        "movie_titles.csv"

    main(data_files=DATA_FILES, show_mapping_file_path=SHOW_MAPPING_FILE_PATH)
