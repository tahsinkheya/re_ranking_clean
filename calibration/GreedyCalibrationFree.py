# tahsin kheya
# last modified 07/01/2025
import pandas as pd
import os
import numpy as np

# import torch
# import time
from logging import getLogger
import random
import numpy as np
from scipy.stats import entropy
# from mpi4py import MPI

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()


class GreedyCalibrationFree(object):
    def __init__(
        self, config, movies, top_k, unique_genres, users, sensitive_attr, beta
    ):
        self.top_k = top_k
        self.beta = beta
        self.users_df = users
        self.actual_genre_dist = pd.read_csv(
            os.path.join(config["user_genre_dist_file"]),
            sep="\t",
        )
        self.actual_genre_dist = self.actual_genre_dist.sort_values(by="userID")
        self.unique_genres = unique_genres
        self.item_df = movies
        self.sensitive_attr = sensitive_attr
        self.sensitive_compare_dist = []
        self.actual_distribution_sensitive = []

    def get_recom_distribution(self, reco, uid, compare_dist, alpha):
        reco = np.array(reco)
        df_reco = pd.DataFrame(
            {
                "userID": np.repeat(uid, len(reco)),
                "itemID": reco.flatten(),
                "rank": np.tile(np.arange(1, len(reco) + 1), 1),
            }
        )
        df_reco["weight_factor"] = 1 / (df_reco["rank"]) ** 0.1
        merged_df = pd.merge(df_reco, self.item_df, on="itemID", how="inner")

        merged_df[self.unique_genres] = merged_df[self.unique_genres].div(
            merged_df[self.unique_genres].sum(axis=1), axis=0
        )
        for i, genre in enumerate(self.unique_genres):
            merged_df[genre] = (1 - alpha) * merged_df[genre] + alpha * compare_dist[i]

        merged_df[self.unique_genres] = (
            merged_df["weight_factor"].values[:, None] * merged_df[self.unique_genres]
        )
        # for i, genre in enumerate(self.unique_genres):
        #     merged_df[genre] = (1 - alpha) * merged_df[genre] + alpha * compare_dist[i]

        summed_genre = (
            merged_df.groupby("userID")[self.unique_genres].sum().reset_index()
        )

        return summed_genre[self.unique_genres].to_numpy()

    def compute_diversity_score(self, reco_items, uid, scores, b):
        alpha = 0.01
        sum_score = 0
        current_user_sensitive_attr = self.users_df.loc[
            self.users_df["userID"] == uid, self.sensitive_attr
        ].item()
        compare_dist = self.sensitive_compare_dist[current_user_sensitive_attr]
        reco_dist = self.get_recom_distribution(reco_items, uid, compare_dist, alpha)[
            0
        ]  # sum wr(i)q˜(д|i),

        reco_dist = np.log(reco_dist)  # log sum wr(i)q˜(д|i),
        faireness_term = np.sum(compare_dist * reco_dist)

        for r in range(len(reco_items)):
            sum_score += scores[reco_items[r]]

        return (sum_score, faireness_term)

    def get_improved_reco(self, top_items, items, scores):
        self.get_sensitive_genre_dist()

        return self.get_new_recommendations(
            reco=top_items, scores=scores, all_items=items
        )

    def get_sensitive_genre_dist(self):
        actual_dist_sensitive = pd.merge(
            self.actual_genre_dist, self.users_df, on="userID"
        )
        sensitive_genre_weights_a = actual_dist_sensitive.groupby(self.sensitive_attr)[
            self.unique_genres
        ].mean()

        self.actual_distribution_sensitive = sensitive_genre_weights_a.sort_index()

        sensitive_compare_dist = self.actual_distribution_sensitive.sum()

        self.sensitive_compare_dist = self.actual_distribution_sensitive.apply(
            lambda r: sensitive_compare_dist - r, axis=1
        )
        self.sensitive_compare_dist = (
            self.sensitive_compare_dist.sort_index().to_numpy()
        )

    def normalize_scores(self, diversity_scores, b):
        filtered_scores = [
            x for x in diversity_scores if isinstance(x, tuple) and len(x) == 2
        ]

        min_score = min(filtered_scores, key=lambda x: x[0])[0]
        max_score = max(filtered_scores, key=lambda x: x[0])[0]

        min_fairness = min(filtered_scores, key=lambda x: x[1])[1]
        max_fairness = max(filtered_scores, key=lambda x: x[1])[1]

        all_scores = []
        for i in range(diversity_scores.__len__()):
            if diversity_scores[i] != 0:
                score = (diversity_scores[i][0] - min_score) / (max_score - min_score)
                fairness_term = (diversity_scores[i][1] - min_fairness) / (
                    max_fairness - min_fairness
                )
                # print(
                #     f"score {score} ft {fairness_term} total {(1 - b) * score + b * fairness_term}"
                # )
                all_scores.append((1 - b) * score + b * fairness_term)
            else:
                # print(f"index {i}")
                all_scores.append(-9999)

        return all_scores

    def get_new_recommendations(self, reco, scores, all_items):
        """reco is 6040x50 and scores is 6040x3416"""
        b = self.beta  # beta for the fairness term
        all_users = []
        top_k = self.top_k
        num_users = len(scores)
        # upper_bound = min(num_users, rank * 25 + 25)
        for u in range(0, num_users):
            # remaining_items = list(range(20))
            remaining_items = reco[u]
            # print(remaining_items)
            u_calibrated = []
            for k in range(top_k):
                diversity_scores = [
                    self.compute_diversity_score(u_calibrated + [i], u, scores[u], b)
                    for i in remaining_items
                ]
                # print(f"before {remaining_items} len {remaining_items.__len__()}")

                norm_diversity_scores = self.normalize_scores(diversity_scores, b)
                max_index = np.argmax(norm_diversity_scores)
                best_item = remaining_items[max_index]
                # print(f"best item {best_item}")
                # popped_element = arr[index_to_pop     ]
                # new_arr = np.delete(arr, index_to_pop)

                u_calibrated.append(best_item)
                remaining_items = np.delete(remaining_items, max_index)
                # print(f"after {remaining_items} len {remaining_items.__len__()}")
                # print(u_calibrated)

                # remaining_items.pop(max_index)
                # print(u_calibrated)
            print(f"user {u} u_calibrate {u_calibrated}")

            all_users.append(u_calibrated)

        return np.array(all_users)
