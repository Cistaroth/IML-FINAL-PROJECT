# Netflix Recommendation System

<div align="center">

![Static Badge](https://img.shields.io/badge/github-repo-blue?logo=github)
![Static Badge](https://img.shields.io/badge/version-1.0.0-green)
![Static Badge](https://github.com/Cistaroth/IML-FINAL-PROJECT/actions/workflows/run.yml/badge.svg)


![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

</div>

<br>
<p align= "center">
We attempt to classify whether a user will watch a show using a KNN architecture. We train on the Netflix Prize Dataset.
</p>

## Workflow

```mermaid
graph LR;
    data_pull[Pull Data]
    data_visualisation[Data Visualisation]
    select_user[Selection of User]
    collect_metadata[Collect Movie Meta Data]
    data_cleaning[Cleaning up Data]

    generate_embeddings[Generate Embeddings]
    train_test_split[Train-Validation-Test Split]
    one_hot_encode_data[One Hot Encode Data]
    normalize_data[Normalize Data]
    dimensionality_reduction[Dimensionality Reduction]

    train_model[Train KNN model]
    k_fold_cross_validation[K-fold Cross Validation]
    select_optimal_parameters[Select Optimal Parameters]
    train_final_model[Train Final Model]
    model_evaluation[Evaluate Model]

    subgraph obtain_data["<h1> Obtaining Data </h1>"]
    direction TB;
    s1[ ]---data_pull;
    data_pull-->data_visualisation;
    data_visualisation-->select_user;
    select_user-->collect_metadata;
    collect_metadata-->data_cleaning
    end

    subgraph preprocessing["<h1> Preprocessing </h1>"]
    direction TB;
    s2[ ]---generate_embeddings;
    generate_embeddings-->train_test_split;
    train_test_split-->one_hot_encode_data;
    one_hot_encode_data-->normalize_data;
    normalize_data-->dimensionality_reduction
    end

    subgraph model["<h1> <br/> Model </h1>"]
    direction TB;
    s3[ ]---train_model;
    train_model-->k_fold_cross_validation;
    k_fold_cross_validation-->select_optimal_parameters;
    select_optimal_parameters-->train_final_model;
    train_final_model-->model_evaluation;
    end

    obtain_data-->preprocessing;

    preprocessing-->model;
    
    style s1 fill:none,stroke:none
    style s2 fill:none,stroke:none
    style s3 fill:none,stroke:none
    linkStyle 0,5,10 stroke:none



```

<!--

1. Pulling Data from [Dataset](#dataset)
2. Data Visualisation
3. Extract Rated Movies and Ratings from One User
4. Collect Metadata of Rated Movies
5. Convert Metadata to Embeddings where Necessary
6. Train-Validation-Test Split
7. One Hot Encode and Normalize Data
8. Train KNN Model
9. Perform K-fold Cross-Validation to Find Optimal Parameters
10. Evaluate KNN Model on Test-Dataset
-->

## Dataset

<h3> Netflix Price Dataset  </h3>

The dataset can be found [here.](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)

<h3> Context: </h3>
Netflix held the Netflix Prize open competition for the best algorithm to predict user ratings for films. The grand prize was $1,000,000 and was won by BellKor's Pragmatic Chaos team. This is the dataset that was used in that competition.

<br>
<h3> Dataset format: </h3>

The file "training_set.tar" is a tar of a directory containing 17770 files, one
per movie. The first line of each file contains the movie id followed by a
colon. Each subsequent line in the file corresponds to a rating from a customer
and its date in the following format:

<p align="center">
CustomerID,Rating,Date
</p>

MovieIDs range from 1 to 17770 sequentially.
CustomerIDs range from 1 to 2649429, with gaps. There are 480189 users.
Ratings are on a five star (integral) scale from 1 to 5.
Dates have the format YYYY-MM-DD.

## Usage
We require installation of the UV Python package and project manager. Instructions for installation can be found [here](https://docs.astral.sh/uv/getting-started/installation/). We also require git to be installed.

```
> git clone https://github.com/Cistaroth/IML-FINAL-PROJECT.git
> cd "IML-FINAL-PROJECT"
> uv sync
> uv run main.py
```

Raw data must be placed in the [raw_data](/raw_data/) folder. Please refer to the README.md of the aforementioned folder. To access processed data, the zip within the [processed_data](/processed_data/) folder must be unpacked.

## Report
The report can be found [here](https://www.overleaf.com/8783437716pkfsdmznyffg#48f67d).


## Credits
Developed as part of an Introduction to Machine Learning Course in 2025/2026

[@Cistaroth](https://github.com/Cistaroth) [@SyntaxSculptor1](https://github.com/SyntaxSculptor1) [@ctenderini-web](https://github.com/ctenderini-web)
