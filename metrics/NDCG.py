import numpy as np


class NDCG:
    def __init__(self, k):
        """
        initializating the metric
        Parameters
        ----------
        k = number upto which we evaluate
        """
        self.name = "NDCG@{}".format(k)
        self.k = k

    def dcg_score(self, gt_pos, pd_rank, k=-1):
        """Compute Discounted Cumulative Gain score.

        Parameters
        ----------
        gt_pos: Numpy array
            Vector of positive items.

        pd_rank: Numpy array
            Item ranking prediction.

        k: int, optional, default: -1 (all)
            The number of items in the top@k list.
            If None, all items will be considered.

        Returns
        -------
        dcg: A scalar
            Discounted Cumulative Gain score.

        """
        if k > 0:
            truncated_pd_rank = pd_rank[:k]
        else:
            truncated_pd_rank = pd_rank

        ranked_scores = np.in1d(truncated_pd_rank, gt_pos).astype(int)
        gain = 2**ranked_scores - 1
        discounts = np.log2(np.arange(len(ranked_scores)) + 2)

        return np.sum(gain / discounts)

    def compute(self, gt_pos, pd_rank):
        """Compute Hit Ratio.

        Parameters
        ----------
        gt_pos: items from test set for users. so the tps basically

        pd_rank: ranked items

        Returns
        -------
        res: A scalar
            Hit Ratio score (1.0 ground truth item(s) appear in top-k, 0 otherwise).

        """
        dcg = self.dcg_score(gt_pos, pd_rank, self.k)
        idcg = self.dcg_score(gt_pos, gt_pos, self.k)
        # print(f"{dcg} dcg and idcg {idcg}")
        ndcg = dcg / idcg

        return ndcg
