import numpy as np


class HitRatio:
    def __init__(self, k):
        """
        initializating the metric
        Parameters
        ----------
        k = number upto which we evaluate
        """
        self.name = "HitRatio@{}".format(k)
        self.k = k

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
        if self.k > 0:
            truncated_pd_rank = pd_rank[: self.k]
        else:
            truncated_pd_rank = pd_rank
        # print(":"*10)
        # print(truncated_pd_rank)
        # print(gt_pos)
        # print(":"*10)

        tp = np.sum(np.in1d(truncated_pd_rank, gt_pos))

        return 1.0 if tp > 0 else 0.0
