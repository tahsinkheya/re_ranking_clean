import numpy as np
import pandas as pd
from calibration.GreedyCalibration import GreedyCalibration
from mpi4py import MPI
import csv
import time
import os

start_time = time.time()
data_load_start = time.time()
reco_matrix = np.load("../model_reco/yelp/wmf/reco_matrix_wmf_yelp100k_100.npy")[0]
print(reco_matrix.shape)
# print(scores.shape)
import pickle
# tahsin/reranking_fairnes/model_reco/neumf/score_dicts_neumf100k.pkl
with open("../model_reco/yelp/wmf/score_dicts_wmf_yelp100k.pkl", "rb") as file:
    # tahsin/reranking_fairnes/model_reco/mf/score_dicts_mf100k.pkl
    scores = pickle.load(file)
print(f"{os.getcwd()} yo yo ")


restaurants = pd.read_csv("../data/yelp-100K/i_id_mapping.csv",sep="\t",
    header=0,
    names=[ "item_id","Category","itemID"])
restaurants=restaurants.sort_values(by="itemID")

unique_categories = [
    "Active Life & Fitness",
    "Arts & Entertainment",
    "Automotive",
    "Bars & Nightlife",
    "Coffee,Tea & Desserts",
    "Drinks & Spirits",
    "Education & Learning",
    "Event Services",
    "Family & Kids",
    "Food & Restaurants",
    "Health & Beauty",
    "Home & Garden",
    "Miscellaneous",
    "Outdoor Activities",
    "Public Services & Community",
    "Shopping & Fashion",
    "Specialty Food & Groceries",
    "Sports & Recreation",
    "Technology & Electronics",
    "Travel & Transportation",
    "Asian",
]
for c in unique_categories:
    restaurants[c] = 0
for index, row in restaurants.iterrows():
    cats = row["Category"].split("|")
    for cat in cats:
        restaurants.at[index, cat] = 1

cat = restaurants[unique_categories]
# cat[:1]
item_features_numpy = cat.to_numpy()
item_features = {
    str(item_id): {"category_" + str(idx): value for idx, value in enumerate(row)}
    for item_id, row in enumerate(item_features_numpy)
}
# ids = list(range(0, 3416))
restaurants

users = pd.read_csv("../data/yelp-100K/u_id_mapping.csv", sep="\t",header=0,
    names=[ "user_id","Gender","userID"])
gender_map = {"M": 0, "F": 1}
users["Gender"] = users["Gender"].map(gender_map)
users = users.sort_values(by="userID")
users = users[["Gender", "userID"]]
user_features_numpy = users.to_numpy()
users
item_ids = restaurants["itemID"].to_numpy()



# users = pd.read_csv("../data/ml-100k/u_id_mapping_demographic_.csv", sep="\t")

# users = users.sort_values(by="userID")


# # users = users.drop(columns=users.columns[0])
# gender_map = {"M": 0, "F": 1}
# users["Gender"] = users["Gender"].map(gender_map)
# user_features_numpy = users.to_numpy()

# item_ids = movies["itemID"].to_numpy()
# # item_ids
# from metrics.utils import map_age
# users["Age_Code"] = users["Age"].apply(map_age)
top_k = 20

data_load_end = time.time()
print(f"Data loaded in {data_load_end - data_load_start:.2f} seconds.")

calibration_start = time.time()
sensitive_attr = "Gender"
beta = 0.1
config = {"user_genre_dist_file": "../data/yelp-100K/pgui.csv"}
calibration = GreedyCalibration(
    config, restaurants, top_k, unique_categories, users, sensitive_attr, beta
)
reranked_reco = calibration.get_improved_reco(reco_matrix, list(item_ids), scores)

calibration_end = time.time()


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(
    f"Calibration process took {calibration_end - calibration_start:.2f} seconds for Rank {rank}."
)


# print(":::::::::::")
# print(f"right now working on {rank*30} - {rank*30+30}")
print(">>" * 10)
print(reranked_reco)
print(">>" * 10)

node_data = reranked_reco
all_data = comm.gather(node_data, root=0)


def write_to_file(all_data, top_k):
    with open("Gender_1_yelp.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["user_id"] + [f"top_{i+1}" for i in range(top_k)])
        print(":" * 10)
        print(all_data)
        print(all_data.__len__())
        print(":" * 10)
        i = 0
        for reco in all_data:
            r = [str(i)] + [str(w) for w in reco]
            writer.writerow(r)
            i = i + 1


if rank == 0:
    write_start = time.time()
    all_data = [item for sublist in all_data for item in sublist]
    write_to_file(all_data, top_k)
    write_end = time.time()
    print(f"Data writing took {write_end - write_start:.2f} seconds.")

end_time = time.time()
print(f"Total script execution time: {end_time - start_time:.2f} seconds.")
