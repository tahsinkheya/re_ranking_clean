import numpy as np
import pandas as pd
from math import comb
from itertools import combinations

class GenreNDCGMulti:
    def __init__(self, gender_df, unique_genres,top_k, **kwargs):
        """
        initializating genders of the users
        Parameters
        ----------
        gender_mapping : dict
            A dictionary mapping user IDs to their genders.
        """
        super().__init__(name="GenrePrecision", **kwargs)
        self.gender_df = gender_df
        self.unique_genres = unique_genres
        self.top_k = top_k

    def compute(self, reco_matrix, item_df, sensitive_attr):
        """
        reco_matrix : n_userxk np array containing the ranked recommended list for users
        item_df : pd df containing all items with ids and genre info as ohe
        returns the abs diff for each gender genre distribution
        sensitive_attr: the sensitive attribute for which we wanna find unfairness
        
        """

        rank_val = np.arange(2, self.top_k+2)
        logVal = np.log2(rank_val)

        df_reco = pd.DataFrame(
            {
                "userID": np.repeat(np.arange(reco_matrix.shape[0]), self.top_k),
                "itemID": reco_matrix.flatten(),
                "rank": np.tile(np.arange(1, self.top_k + 1), reco_matrix.shape[0]),
                "logVal": np.tile(logVal, reco_matrix.shape[0]),
            }
        )
        merged_df = pd.merge(df_reco, item_df, on="itemID", how="inner")
        merged_df[self.unique_genres] = merged_df[self.unique_genres].div(
            merged_df[self.unique_genres].sum(axis=1), axis=0
        )

        merged_df[self.unique_genres] = merged_df.apply(
            lambda r: r[self.unique_genres] / r["logVal"], axis=1
        )
        reco_distribution = merged_df[["userID"] + self.unique_genres]

        reco_distribution = reco_distribution.groupby("userID")[
            self.unique_genres
        ].mean()

        g_reco_distribution = self.get_sensitive_attr_genre_dist(reco_distribution, sensitive_attr)

        return self.genre_result(g_reco_distribution)

    def get_sensitive_attr_genre_dist(self, user_reco,sensitive_attr):
        """
        user_reco : is the recommended genre distibution for all users
        sensitive_attr : the sensitive attribute for which we wanna find unfairness
        
        """
        recomen_df = pd.merge(user_reco, self.gender_df, on="userID")
        sensitive_attr_genre_weights_r = recomen_df.groupby(sensitive_attr)[self.unique_genres].mean()
        distribution_gender = sensitive_attr_genre_weights_r.sort_index()
        return distribution_gender

    def pairwise_abs_diff(self, sensitive_attr_genre_dist):
        """
        sensitive_attr_genre_dist : the genre distibution for each genre grouped by sensitive attribute given 
        """
        ret_val = 0
        genre_dist = []
        for g in self.unique_genres:
            genre_pref = sensitive_attr_genre_dist[g].values
            g_dist = 0
            for si, sj in combinations(range(len(genre_pref)), 2):
                # print(f"si {si} sj {sj}")
                val = genre_pref[si] - genre_pref[sj]
                ret_val += abs(val)
                g_dist = g_dist + abs(val)
            genre_dist.append(g_dist)
        gender_genre_dist = gender_genre_dist.to_numpy()
        possible_comb = comb(len(sensitive_attr_genre_dist), 2)
        print("::"*10)
        print(gender_genre_dist[0] - gender_genre_dist[1])
        print("::"*10)
        
        return ret_val / possible_comb, np.array(genre_dist) / possible_comb
        
        
