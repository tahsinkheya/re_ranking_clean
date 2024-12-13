import numpy as np
import pandas as pd
from calibration.Calibration import Calibration 
from mpi4py import MPI
import csv
import time 
import os
# load rankings
# print(pwd)
print(os.getcwd())
start_time = time.time()
print("Loading data...")
data_load_start = time.time()
reco_matrix = np.load("../test/reco_matrix.npy")[0]
# scores = np.load("reco_matrix_mapped_scores.npy")[0]
print(reco_matrix.shape)
# print(scores.shape)
import pickle

with open("../test/score_dicts.pkl", "rb") as file:
    scores = pickle.load(file)
print(f"{os.getcwd()} yo yo ")

# print("Loaded List of OrderedDicts:", scores)
df_m = pd.read_csv(
    "../data/ml-100k/u.item",
    sep="|",
    names=[
        "movieID",
        "Name",
        "Date",
        "Video_Date",
        "IMDB_URL",
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ],
    header=None,
    encoding="latin-1",
)
print(df_m.shape)
df_m = df_m[
    [
        "movieID",
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]
]

df_movies_mapped = pd.read_csv(
    "../data/ml-100k/i_id_mapping.csv",
    sep="\t",
    names=["movieID", "itemID"],
    header=None,
    encoding="latin-1",
)
movies = pd.merge(df_m, df_movies_mapped, how="inner", on="movieID")
movies = movies.drop(columns=["movieID"])
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
genre = movies[unique_genres]
item_features_numpy = genre.to_numpy()

users = pd.read_csv("../data/ml-100k/u_id_mapping.csv", sep="\t")

users = users.sort_values(by="userID")

users = users.drop(columns=users.columns[0])
gender_map = {"M": 0, "F": 1}
users["Gender"] = users["Gender"].map(gender_map)
user_features_numpy = users.to_numpy()
def create_genre_column(r):
    all_genres = [g for g in unique_genres if r[g] == 1]
    return "|".join(all_genres)


movies["genres"] = movies.apply(create_genre_column, axis=1)
movies
item_ids = movies["itemID"].to_numpy()
item_ids
top_k=10

data_load_end = time.time()
print(f"Data loaded in {data_load_end - data_load_start:.2f} seconds.")

calibration_start = time.time()

config = {"user_genre_dist_file": "../data/ml-100k/pgui.csv"}
calibration = Calibration(config, movies, top_k, unique_genres, users)
reranked_reco=calibration.get_improved_reco(reco_matrix, list(item_ids), scores)

calibration_end = time.time()



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(f"Calibration process took {calibration_end - calibration_start:.2f} seconds for Rank {rank}.")


# print(":::::::::::")
# print(f"right now working on {rank*30} - {rank*30+30}")
print(">>"*10)
print(reranked_reco)
print(">>"*10)

node_data = reranked_reco
all_data = comm.gather(node_data, root=0)
def write_to_file(all_data, top_k):
    with open('reranked_recommendations.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['user_id'] + [f'top_{i+1}' for i in range(top_k)])
        print(":"*10)
        print(all_data)
        print(all_data.__len__())
        print(":"*10)
        i=0
        for reco in all_data:
            r = [str(i)] + [str(w) for w in reco]
            writer.writerow(r)
            i=i+1
if rank==0:
    write_start = time.time()
    all_data = [item for sublist in all_data for item in sublist]
    write_to_file(all_data, top_k)
    write_end = time.time()
    print(f"Data writing took {write_end - write_start:.2f} seconds.")
    
end_time = time.time()
print(f"Total script execution time: {end_time - start_time:.2f} seconds.")

