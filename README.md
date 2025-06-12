Code for paper titled Bias vs Bias - Dawn of Justice:
A Fair Fight in Recommendation Systems
published in ECMLPKDD 2025

<b>To get fair category-aware re-ranked items for users just follow these simple steps!</b>

## step 1

Get ur dataset ready. You need user demographich informatio and item genres. Additionally, you need to get the historical category distribution. Please check the alreasy existing datasets in this repo, for an example. the datasets are in the folder named _data_

## step 2

Once you have them, then you can run the models using the datasets. Please refer to the experiment folder for details. After running each of the models, we store them as npy file and also store the scores of items in a pkl file.

## step 3

Run the re-ranking scheme using the GreedyCalibration code. The code is stored in re_ranking.py for one of the datasets. The results are stored as a csv file.

## step 4

The results stored in the csv file and npy file are then evaluated for bias and performance. Examples are shown in the ipynb files stored in the metrics folder.
