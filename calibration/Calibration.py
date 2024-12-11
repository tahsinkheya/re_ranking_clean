# tahsin kheya
# last modified 11/12/2024
import pandas as pd
import os
import numpy as np
import torch
import time
from logging import getLogger
import random
import numpy as np
from scipy.stats import entropy


class Calibration(object):
    def __init__(self, config, movies, top_k, unique_genres, users):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device)
        self.reco_distribution = []
        self.kl = []
        self.top_k = top_k
        self.gkl = []
        self.gender_df = users
        self.actual_genre_dist = pd.read_csv(
            os.path.join(config["user_genre_dist_file"]),
            sep="\t",
        )
        self.actual_genre_dist = self.actual_genre_dist.sort_values(by="userID")
        self.unique_genres = unique_genres
        self.config = config
        self.item_df = movies
        self.actual_distribution_without_id = self.actual_genre_dist.drop(
            columns=["userID"]
        )
        self.actual_distribution_without_id = self.actual_distribution_without_id.drop(
            columns=["user_timestamp_sum"]
        ).to_numpy()
        self.actual_distribution_gender = []

    def get_all_user_recommended_genre_dist(self, topk_reco):
        df_reco = pd.DataFrame(
            {
                "userID": np.repeat(np.arange(topk_reco.shape[0]), self.top_k),
                "itemID": topk_reco.flatten(),
                "rank": np.tile(np.arange(1, self.top_k + 1), topk_reco.shape[0]),
            }
        )
        df_reco["weight_factor"] = 1 / (df_reco["rank"]) ** 0.1
        merged_df = pd.merge(df_reco, self.item_df, on="itemID", how="inner")
        merged_df[self.unique_genres] = merged_df[self.unique_genres].div(
            merged_df[self.unique_genres].sum(axis=1), axis=0
        )
        merged_df[self.unique_genres] = (
            merged_df["weight_factor"].values[:, None] * merged_df[self.unique_genres]
        )
        summed_genre = (
            merged_df.groupby("userID")[self.unique_genres].sum().reset_index()
        )
        weight_total = merged_df.groupby("userID")["weight_factor"].sum()
        summed_genre["user_weight_factor"] = summed_genre["userID"].map(weight_total)
        summed_genre = (
            merged_df.groupby("userID")[self.unique_genres].sum().reset_index()
        )
        summed_genre = summed_genre[self.unique_genres].div(
            summed_genre["user_weight_factor"], axis=0
        )

        return summed_genre

    def get_recom_distribution(self, reco, uid):
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
        merged_df[self.unique_genres] = (
            merged_df["weight_factor"].values[:, None] * merged_df[self.unique_genres]
        )
        summed_genre = (
            merged_df.groupby("userID")[self.unique_genres].sum().reset_index()
        )

        weight_total = merged_df.groupby("userID")["weight_factor"].sum()
        summed_genre["user_weight_factor"] = summed_genre["userID"].map(weight_total)

        summed_genre = summed_genre[self.unique_genres].div(
            summed_genre["user_weight_factor"], axis=0
        )

        return summed_genre[self.unique_genres].to_numpy()

    def get_kl_div(self, q_dist, p_dist, a, uid):
        qg_u = (1 - a) * q_dist + a * p_dist
        kl_div = entropy(p_dist, qg_u, base=10)

        return kl_div

    def compute_diversity_score(self, reco_items, uid, l, scores, b):
        alpha = 0.01
        sum_score = 0

        # -----------------the calibration term----------------------------------

        reco_dist = self.get_recom_distribution(reco_items, uid)[0]
        kl = self.get_kl_div(
            reco_dist, self.actual_distribution_without_id[uid], alpha, uid
        )

        # -----------------the calibration term----------------------------------

        # -----------------the fairness term----------------------------------
        # recommended dist mean for each gender
        male_user_ids = self.gender_df[self.gender_df["Gender"] == 0]["userID"]
        male_user_ids = male_user_ids.to_list()
        gender_genre_dist = self.actual_distribution_gender
        if uid in male_user_ids:
            compare_dist = gender_genre_dist[self.unique_genres].to_numpy()[
                1
            ]  # this is the female avg genre pref
        else:
            compare_dist = gender_genre_dist[self.unique_genres].to_numpy()[0]

        gender_kl = self.get_kl_div(reco_dist, compare_dist, alpha, uid)

        # -----------------the fairness term----------------------------------

        for r in range(len(reco_items)):
            sum_score += scores[reco_items[r]]

        return (1 - l - b) * sum_score - l * kl - b * gender_kl

    def get_improved_reco(self, top_items, items, scores):
        return self.get_new_recommendations(
            reco=top_items, scores=scores, all_items=items
        )

    def get_gender_genre_dist(self):
        actual_dist_gender = pd.merge(
            self.actual_genre_dist, self.gender_df, on="userID"
        )
        gender_genre_weights_a = actual_dist_gender.groupby("Gender")[
            self.unique_genres
        ].mean()
        self.actual_distribution_gender = gender_genre_weights_a.sort_index()

    def get_new_recommendations(self, reco, scores, all_items):
        """reco is 6040x50 and scores is 6040x3416"""
        self.get_gender_genre_dist()
        b = 0.69  # beta for the fairness term
        l = 0.29  # lambda for the calibration term
        all_users = []
        top_k = self.top_k
        num_users = len(scores)
        for u in range(num_users):
            remaining_items = all_items
            u_calibrated = []
            for k in range(top_k):
                diversity_scores = [
                    self.compute_diversity_score(u_calibrated + [i], u, l, scores[u], b)
                    for i in remaining_items
                ]
                max_index = np.argmax(diversity_scores)

                best_item = remaining_items[max_index]
                u_calibrated.append(best_item)
                remaining_items.pop(max_index)
                print(u_calibrated)

            all_users.append(u_calibrated)

        return np.array(all_users)
