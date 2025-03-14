import numpy as np
import pandas as pd
from calibration.GreedyCalibration import GreedyCalibration
import csv
import time
import os
import pickle

reco_matrix = np.load("./model_reco/vaecf/reco_matrix_vaecf_ml100k_100.npy")
with open("./model_reco/vaecf/score_dicts_vaecf_ml100k.pkl", "rb") as file:
    scores = pickle.load(file)
movies = pd.read_csv(
    "./data/ml-100k/i_id_mapping_genre.csv",
    sep="\t",
    names=["item_id", "Name", "genres", "itemID"],
    header=0,
)
movies = movies.drop(columns=["item_id"])
movies = movies.sort_values(by="itemID")

unique_genres = [
    "Action",
    "Thriller",
    "Romance",
    "Western",
    "Children's",
    "Mystery",
    "Fantasy",
    "Film-Noir",
    "Documentary",
    "Comedy",
    "Adventure",
    "Sci-Fi",
    "Horror",
    "Crime",
    "Musical",
    "War",
    "Animation",
    "Drama",
]
for genre in unique_genres:
    movies[genre] = 0

for index, row in movies.iterrows():
    genres = row["genres"].split("|")
    
    for genre in genres:
        movies.at[index, genre] = 1

users = pd.read_csv("./data/ml-100k/u_id_mapping_demographic_.csv", sep="\t")

users = users.sort_values(by="userID")
gender_map = {"M": 0, "F": 1}
users["Gender"] = users["Gender"].map(gender_map)
user_features_numpy = users.to_numpy()
from metrics.utils import map_age
users["Age_Code"] = users["Age"].apply(map_age)
item_ids = movies["itemID"].to_numpy()

top_k = 20
sensitive_attr = "Gender"
beta = 0.6
config = {"user_genre_dist_file": "./data/ml-100k/pgui.csv"}
calibration = GreedyCalibration(
    config, movies, top_k, unique_genres, users, sensitive_attr, beta
)
reranked_reco = calibration.get_improved_reco(reco_matrix, list(item_ids), scores)

# save the reranked reco as csv for npy and you are done!


