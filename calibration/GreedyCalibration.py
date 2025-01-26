# tahsin kheya
# last modified 26/01/2025
import pandas as pd
import os
import numpy as np

# import torch
# import time
from logging import getLogger
import random
import numpy as np
from scipy.stats import entropy
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class GreedyCalibration(object):
    def __init__(self, config, movies, top_k, unique_genres, users, sensitive_attr, beta):
        self.top_k = top_k
        self.users_df = users
        self.users_np = users.to_numpy()
        self.beta = beta
        self.actual_genre_dist = pd.read_csv(
            os.path.join(config["user_genre_dist_file"]),
            sep="\t",
        )
        self.actual_genre_dist = self.actual_genre_dist.sort_values(by="userID")
        self.unique_genres = unique_genres
        self.i =movies
        self.i[unique_genres] = self.i[self.unique_genres].div(
            self.i[self.unique_genres].sum(axis=1), axis=0
        )
        self.sensitive_attr = sensitive_attr
        self.sensitive_compare_dist = []
        self.actual_distribution_sensitive = []

    def get_recom_distribution(self, reco, uid, compare_dist, alpha):
        reco = np.array(reco)
        selected_items = self.i[self.i['itemID'].isin(reco)]
        selected_items = selected_items.set_index('itemID').loc[reco].reset_index()
        selected_items = selected_items[self.unique_genres].to_numpy()
        indices = np.arange(len(reco))
        weights = 1 / (indices + 1) ** 0.1
        updated_genre_proportions = (1 - alpha) * selected_items + alpha * compare_dist
        updated_genre_proportions = weights[:, None] * updated_genre_proportions
        updated_genre_proportions = updated_genre_proportions.sum(axis=0)
        return updated_genre_proportions

    def compute_diversity_score(self, reco_items, uid, scores, b):
        alpha = 0.01
        userid = self.users_df.columns.get_loc("userID")
        sens_atr = self.users_df.columns.get_loc(self.sensitive_attr)
        filtered_rows = self.users_np[self.users_np[:, userid] == uid][0]
        current_user_sensitive_attr= filtered_rows[sens_atr]
        compare_dist = self.sensitive_compare_dist[current_user_sensitive_attr]
        # print(f"user id {uid} with gender {current_user_sensitive_attr} compare_dist {compare_dist} ")
        reco_dist = self.get_recom_distribution(reco_items, uid, compare_dist, alpha)  # sum wr(i)q˜(д|i),
        reco_dist = np.log(reco_dist)  # log sum wr(i)q˜(д|i),
        faireness_term = np.sum(compare_dist * reco_dist)
        sum_score = sum(scores[item] for item in reco_items)
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
        nonzeroarray = np.full(len(self.unique_genres), 0.001)
        tradeoff = 0.01
        self.sensitive_compare_dist = tradeoff * nonzeroarray + (1-tradeoff) * self.sensitive_compare_dist #just to ensure we never get log 0 error
        self.sensitive_compare_dist = (
            self.sensitive_compare_dist[self.unique_genres].sort_index().to_numpy()
        )

    def normalize_scores(self, diversity_scores, b):
        div_scores = np.array(diversity_scores)
        scores = div_scores[:, 0]
        fairness = div_scores[:, 1]
        min_score,min_fairness = np.min(scores), np.min(fairness)
        max_score,max_fairness = np.max(scores), np.max(fairness)
        score_norm = (scores- min_score) /(max_score-min_score)
        fairness_norm = (fairness-min_fairness)/(max_fairness-min_fairness)
        fair_score_norm = (1 - b) * score_norm + b * fairness_norm
        fair_score_norm[div_scores[:, 0] == 0] = -9999
        return fair_score_norm

    def get_new_recommendations(self, reco, scores, all_items):
        """reco is 6040x50 and scores is 6040x3416"""
        b = self.beta  # beta for the fairness term
        all_users = []
        top_k = self.top_k
        num_users = len(scores)
        upper_bound = min(num_users, rank * 25 + 25)
        for u in range(rank * 25, upper_bound):
           # remaining_items = list(range(20))
            # remaining_items = reco[u]
            remaining_items=all_items
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
                u_calibrated.append(best_item)
                remaining_items=np.delete(remaining_items, max_index)
                # print(f"after {remaining_items} len {remaining_items.__len__()}")
                # print(u_calibrated)


                
                # remaining_items.pop(max_index)
                # print(u_calibrated)
            print(f"user {u} u_calibrate {u_calibrated}")

            all_users.append(u_calibrated)

        return np.array(all_users)
