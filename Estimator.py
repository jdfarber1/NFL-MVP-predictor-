import pandas as pd
import bs4
import requests
import time
from sklearn import linear_model

class Estimator:

    @staticmethod
    def mvp():
        mvp_df = pd.DataFrame()

        for i in range(2007, 2023):
            string_i = str(i)
            url = f'https://www.pro-football-reference.com/awards/awards_{string_i}.htm#voting_apmvp'
            df = pd.read_html(url)[0]
            df = df[df['Pos'] == "QB"]

            if df.empty:
                continue

            votes = df.iloc[0, 4]
            vote_share = df.iloc[0, 5]
            name = df.iloc[0, 2]
            new_url = f'https://www.pro-football-reference.com/years/{string_i}/passing.htm'
            stats_df = pd.read_html(new_url)[0]
            stats_df['Player'] = stats_df['Player'].str.rstrip('*+')
            player_stats = stats_df[stats_df['Player'] == name]

            if player_stats.empty:
                continue

            player_stats['Votes'] = votes
            player_stats['Share'] = vote_share
            mvp_df = mvp_df.append(player_stats, ignore_index=True)

        return mvp_df

    @staticmethod
    def linear(data, yd, td, Int, rate, qbr):
        X = data[['Yds', 'TD', 'Int', 'Rate', 'QBR']]
        y = data['Votes']
        regr = linear_model.LinearRegression()
        regr.fit(X, y)
        predicted_votes = regr.predict([[yd, td, Int, rate, qbr]])
        return predicted_votes[0]

# Example usage:
# est = Estimator()
# data = est.mvp()
# print(data)
# votes = est.linear(data, 5000, 40, 10, 100.0, 75.0)
# print("Predicted MVP votes:", votes)
