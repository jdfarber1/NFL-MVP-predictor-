import pandas as pd
import matplotlib.pyplot as plt
from Estimator import Estimator
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


def get_player_data(year, player):
    url = f'https://www.pro-football-reference.com/years/{year}/passing.htm'
    df = pd.read_html(url)[0]
    df['Player'] = df['Player'].str.rstrip('*+')
    player_data = df[df['Player'] == player]

    if player_data.empty:
        raise ValueError(f"Player {player} not found in year {year}")

    Yds = int(player_data.iloc[0, 11])
    TD = int(player_data.iloc[0, 12])
    Int = int(player_data.iloc[0, 14])
    Rate = float(player_data.iloc[0, 22])
    QBR = float(player_data.iloc[0, 23])

    return [Yds, TD, Int, Rate, QBR]


def main():
    year = input("Enter year (ex: 2007): ")
    print("Enter 3 players")
    players = [input(f"Enter player {i + 1}: ") for i in range(3)]

    data = Estimator.mvp()

    regr = LinearRegression()
    X = data[['Yds', 'TD', 'Int', 'Rate', 'QBR']]
    y = data['Votes']
    regr.fit(X, y)

    name_list = []
    vote_list = []

    while True:
        yr = input("Enter year or type 'stop' to stop: ")
        if yr.lower() == 'stop':
            break

        for player in players:
            try:
                player_stats = get_player_data(yr, player)
                name_list.append(f"{yr} {player}")
                predicted_votes = regr.predict([player_stats])[0]
                vote_list.append(predicted_votes)
            except ValueError as e:
                print(e)

    for name, votes in zip(name_list, vote_list):
        print(f"{name}: {votes}")


if __name__ == "__main__":
    main()
