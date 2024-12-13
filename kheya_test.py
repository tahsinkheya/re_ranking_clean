import numpy as np
import pandas as pd
from calibration.Calibration import Calibration 
from mpi4py import MPI
# load rankings
# print(pwd)
reco_matrix = np.load("../test/reco_matrix.npy")[0]
# scores = np.load("reco_matrix_mapped_scores.npy")[0]
print(reco_matrix.shape)
# print(scores.shape)
import pickle

with open("../test/score_dicts.pkl", "rb") as file:
    scores = pickle.load(file)

# print("Loaded List of OrderedDicts:", scores)
df_m = pd.read_csv(
    "./data/ml-100K/u.item",
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
    "./data/ml-100K/i_id_mapping.csv",
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

users = pd.read_csv("./data/ml-100k/u_id_mapping.csv", sep="\t")

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

config = {"user_genre_dist_file": "./data/ml-100k/pgui.csv"}
calibration = Calibration(config, movies, 10, unique_genres, users)
reranked_reco=calibration.get_improved_reco(reco_matrix, list(item_ids), scores)



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# print(":::::::::::")
# print(f"right now working on {rank*30} - {rank*30+30}")
# print(reranked_reco)
node_data = reranked_reco
all_data = comm.gather(node_data, root=0)
if rank==0:
    all_data = [item for sublist in all_data for item in sublist]
    write_to_file(all_data, top_k)
def write_to_file(all_data, top_k):
    with open('reranked_recommendations.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['user_id'] + [f'top_{i+1}' for i in range(top_k)])
        for uid, reco in all_data:
            r = [str(uid)+"," + ','.join(str(w) for w in reco)]
            writer.writerow(r)