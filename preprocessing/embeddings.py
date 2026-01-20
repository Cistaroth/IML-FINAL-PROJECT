import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rich.console import Console

console = Console()


def generate_embeddings(
        df: pd.DataFrame,
        columns: list[str],
        embedding_model: SentenceTransformer) -> dict[str, pd.DataFrame]:
    """
    Generate embeddings for the given columns

    Args:
        df (pd.DataFrame): The dataframe of the data
        columns (list[str]): The columns to generate embeddings for
        embedding_model (SentenceTransformer): The embedding model

    Returns:
        dict[str, pd.DataFrame]: The embeddings, keys are the column names
            and the values are the embeddings dataframes
    """
    embeddings = {}

    for column in columns:
        column_to_encode = df[column].to_list()

        embedded_column = embedding_model.encode(
            column_to_encode, show_progress_bar=True
        )
        embedded_column = pd.DataFrame(embedded_column)

        embedded_column.columns = [
            f"{column}_emb_dim_{i}" for i in range(embedded_column.shape[1])
        ]

        embeddings[column] = embedded_column

    return embeddings


def main(
        file_path: Path,
        save_path: Path,
        model_name: str,
        columns: list[str]) -> None:

    data = pd.read_csv(
        file_path,
        quotechar='"',
        escapechar='\\',
        doublequote=True
    )

    embedding_model = SentenceTransformer(model_name)

    embeddings = generate_embeddings(df=data, columns=["title", "overview"],
                                     embedding_model=embedding_model)
    embeddings = pd.concat(list(embeddings.values()), axis=1)

    data = pd.concat([embeddings, data.drop(columns=columns)], axis=1)

    data.to_csv(save_path, index=False, quotechar='"', quoting=1,
                doublequote=True, escapechar='\\')

    console.print(f"[bold green]✔[/bold green] Embeddings saved to \
                  {save_path.name}")


if __name__ == "__main__":
    DATA_FILE_PATH = Path(__file__).parent.parent / "processed_data" / \
        "userdata_metadata.csv"

    SAVE_PATH = Path(__file__).parent.parent / "processed_data" / \
        "userdata_embeddings.csv"

    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    COLUMNS_TO_EMBED = ["title", "overview"]

    main(
        file_path=DATA_FILE_PATH,
        model_name=EMBEDDING_MODEL_NAME,
        save_path=SAVE_PATH,
        columns=COLUMNS_TO_EMBED
    )
