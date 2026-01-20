import pandas as pd
from pathlib import Path
from typing import Literal
import requests
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import track
from dotenv import load_dotenv
import os

TMDB_REQUEST_URL_ID = "https://api.themoviedb.org/3/{media_type}/{show_id}?" +\
    "append_to_response=credits"

TMDB_REQUEST_URL_TITLE = "https://api.themoviedb.org/3/search/multi?query=\
    {title}"

TMDB_REQUEST_URL_GENRE_MAP = "https://api.themoviedb.org/3/genre/{media_type}/\
    list"

console = Console()


def construct_genre_map(api_key: str) -> dict[str, dict[int, str]]:
    """
    Construct a dictionary mapping genre IDs to names for movies and TV shows

    Args:
        api_key (str): API key for The Movie Database

    Returns:
        dict[str, dict[int, str]]: Dictionary mapping genre IDs to names for
            movies and TV shows
    """
    genre_map = {}

    for media_type in ["movie", "tv"]:
        request_url = TMDB_REQUEST_URL_GENRE_MAP.format(media_type=media_type)
        response = requests.get(request_url, params={"api_key": api_key})

        response = {genre["id"]: genre["name"] for genre in response.json()
                    ["genres"]}

        genre_map[media_type] = response

    return genre_map


def TMDB_info_by_ID(
    show_id: int,
    media_type: Literal["tv", "movie"],
    api_key: str,
    cast_limit: int = 5,
) -> dict[str, str | float | int | list | None] | None:
    """
    Fetch information about a movie or TV show from The Movie Database by ID

    Args:
        show_id (int): ID of the movie or TV show
        media_type (Literal["tv", "movie"]): Type of the movie or TV show
        api_key (str): API key for The Movie Database
        cast_limit (int, optional): Maximum number of people to include in the
            credits. Defaults to 5.

    Returns:
        dict[str, str | float | int | list | None] | None: Information about
            the movie or TV show, or None if the request fails
    """

    request_url = TMDB_REQUEST_URL_ID.format(media_type=media_type,
                                             show_id=show_id)
    response = requests.get(request_url, params={"api_key": api_key})

    if response.status_code == 200 and response.json():
        item = response.json()

        title = item.get("title", None) or item.get("name", None)
        overview = item.get("overview", None)
        date = item.get("first_air_date", None) or item.get("release_date",
                                                            None)
        year = int(date[:4]) if date else None
        language = item.get("original_language", None)
        vote_average = item.get("vote_average", None)

        production_countries = [
            production_country["name"]
            for production_country in item.get("production_countries", [])
        ]
        genres = [genre["name"] for genre in item.get("genres", [])]

        credits_dict = item.get("credits", {"cast": [], "crew": []})

        credits_cast = credits_dict["cast"][: cast_limit + 1]
        credits_crew = credits_dict["crew"][: cast_limit + 1]

        credits = set(person["name"] for person in credits_cast + credits_crew)
        credits = list(credits)

        popularity = item.get("popularity", None)

        try:
            runtime = int(item.get("runtime", 0)) or \
                item.get("episode_run_time", [])[0]
        except Exception:
            runtime = None

        return {
            "title": title,
            "overview": overview,
            "year": year,
            "language": language,
            "media_type": media_type,
            "vote_average": vote_average,
            "production_countries": production_countries,
            "genres": genres,
            "credits": credits,
            "runtime": runtime,
            "popularity": popularity,
        }


def TMDB_info_by_title(
    title: str, api_key: str, cast_limit: int = 5
) -> dict[str, str | float | int | list | None] | None:
    """
    Fetch information about a movie or TV show from The Movie Database by title

    Args:
        title (str): Title of the movie or TV show
        api_key (str): API key for The Movie Database
        cast_limit (int, optional): Maximum number of people
            to include in the credits. Defaults to 5.

    Returns:
        dict[str, str | float | int | list | None] | None: Information about
            the movie or TV show, or None if the request fails
    """
    media_type, show_id = None, None

    request_url = TMDB_REQUEST_URL_TITLE.format(title=title.replace(" ", "+"))
    response = requests.get(request_url, params={"api_key": api_key})

    if response.status_code == 200 and response.json()["results"]:
        results = response.json()["results"]

        for item in results:
            item_media_type = item.get("media_type", None)
            if (
                (item_media_type == "tv" or
                 item_media_type == "movie")
                and item.get("overview", None)
            ):
                media_type = item_media_type
                show_id = item.get("id", None)
                break

    if media_type is None or show_id is None:
        return None

    return TMDB_info_by_ID(
        show_id=show_id,
        media_type=media_type,
        api_key=api_key,
        cast_limit=cast_limit
    )


def main(file_path: Path, api_key: str, save_path: Path) -> None:
    """
    Main function

    Args:
        file_path (Path): The path to the file
        save_path (Path): The path to the file
    """

    console.print(Markdown("# Adding Show Metadata"))

    data = pd.read_csv(
        file_path,
        quotechar='"',
        escapechar='\\',
        doublequote=True
    )

    show_information = []
    for idx, row in track(data.iterrows(),
                          total=data.shape[0],
                          description="Adding metadata using TMDB"):
        current_record = row.to_dict()

        metadata = TMDB_info_by_title(title=current_record["title"],
                                      api_key=api_key)

        if metadata is None:
            continue

        metadata.update({"rating": current_record["rating"]})
        show_information.append(metadata)

    show_information = pd.DataFrame(show_information)

    show_information.to_csv(save_path, index=False, quotechar='"', quoting=1,
                            doublequote=True, escapechar='\\')

    console.print(f"[bold green]✔[/bold green] Metadata of userdata saved to \
                  {save_path.name}")


if __name__ == "__main__":
    DATA_FILE_PATH = Path(__file__).parent.parent / "processed_data" / \
        "userdata_raw.csv"

    SAVE_PATH = Path(__file__).parent.parent / "processed_data" / \
        "userdata_metadata.csv"

    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

    API_KEY = os.getenv("TMDB_API_KEY")
    if not API_KEY:
        raise Exception("TMDB_API_KEY not found in .env file")

    main(file_path=DATA_FILE_PATH, api_key=API_KEY, save_path=SAVE_PATH)
