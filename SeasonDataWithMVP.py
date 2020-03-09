import pandas as pd
import numpy as np


class SeasonDataWithMVP:
    players = []
    year = ''
    DATASET = "Season_Stats_MVP.csv"
    WINS_DATASET = "Wins.csv"

    def setYearToPredict(self, year):
        assert type(year) is int
        assert 2017 >= year >= 2013
        self.year = year

    def setPlayers(self, players):
        self.players = players

    def getMultiYearDataset(self, useNormalize=False):
        dataset = pd.read_csv(self.DATASET)
        dataset = dataset.loc[dataset['Year'].between(2013, 2017, True)]  # get stats from 2013 to 2017
        dataset = dataset.fillna(0)  # sets null/empty values to 0

        wins = pd.read_csv(self.WINS_DATASET)
        dataset = pd.merge(dataset, wins, on=["Tm", "Year"], how="outer")

        self.ensureTotalsColumns(dataset.columns)

        # convert raw stat totals into per game totals
        dataset['PPG'] = dataset['PTS'] / dataset['G']
        dataset['APG'] = dataset['AST'] / dataset['G']
        dataset['RPG'] = dataset['TRB'] / dataset['G']
        dataset['BPG'] = dataset['BLK'] / dataset['G']
        dataset['SPG'] = dataset['STL'] / dataset['G']

        '''
        traded players will have a 'TOT' category that gives their totals throughout the season, regardless of team
        thus, since 'TOT' is not a team, the wins will be nan -- so we replace them with an average win total
        '''
        avgWins = round(dataset['Wins'].mean(), 0)
        dataset['Wins'] = dataset['Wins'].fillna(avgWins)

        dataset = dataset[['Year', 'Player', 'Pos', 'Tm', 'G', 'PPG', 'RPG', 'APG', 'SPG', 'BPG',
                           '3P%', 'FG%', 'Wins', 'PER', 'MVP']]

        # eliminate all players who did not play at least 2/3 of the season
        dataset = dataset[dataset['G'] > 54]

        if useNormalize:
            dataset = self.scaleAllStats(dataset)

        return dataset.values

    def scaleFeature(self, column):
        min = np.min(column)
        max = np.max(column)
        range = max - min

        column = (column - min) / range
        return column

    def ensureTotalsColumns(self, columns):
        assert 'G' in columns
        assert 'Player' in columns
        assert 'Pos' in columns
        assert 'Tm' in columns
        assert 'PTS' in columns
        assert 'TRB' in columns
        assert 'AST' in columns
        assert 'STL' in columns
        assert 'BLK' in columns
        assert '3P%' in columns
        assert 'FG%' in columns
        assert 'Wins' in columns
        assert 'PER' in columns
        assert 'MVP' in columns

    def ensurePerGameColumns(self, columns):
        assert 'G' in columns
        assert 'PPG' in columns
        assert 'RPG' in columns
        assert 'APG' in columns
        assert 'SPG' in columns
        assert 'BPG' in columns
        assert '3P%' in columns
        assert 'FG%' in columns
        assert 'PER' in columns
        assert 'Wins' in columns
        assert 'MVP' in columns

    def scaleAllStats(self, dataset):
        self.ensurePerGameColumns(dataset.columns)

        dataset['G'] = self.scaleFeature(dataset['G'])
        dataset['PPG'] = self.scaleFeature(dataset['PPG'])
        dataset['RPG'] = self.scaleFeature(dataset['RPG'])
        dataset['APG'] = self.scaleFeature(dataset['APG'])
        dataset['SPG'] = self.scaleFeature(dataset['SPG'])
        dataset['BPG'] = self.scaleFeature(dataset['BPG'])
        dataset['3P%'] = self.scaleFeature(dataset['3P%'])
        dataset['FG%'] = self.scaleFeature(dataset['FG%'])
        dataset['PER'] = self.scaleFeature(dataset['PER'])
        dataset['Wins'] = self.scaleFeature(dataset['Wins'])

        return dataset

    def train(self, data):
        # isolate data for year to be predicted
        yearData = data[np.where(data[:, 0] == self.year)]

        # separate player biographical data (name, position, team) from statistics
        self.setPlayers(yearData[:, [1, 2, 3]])
        stats = np.delete(data, [0, 1, 2, 3], 1)

        # separate stats from MVP label, create training data
        xTrain = stats[:, 0:9]
        yTrain = stats[:, 10]
        yTrain = yTrain.astype('int')

        # create test data based on year to predict
        yearStats = np.delete(yearData, [0, 1, 2, 3], 1)
        xTest = yearStats[:, 0:9]
        yTest = yearStats[:, 10]
        yTest = yTest.astype('int')

        return {
            "xTrain": xTrain,
            "yTrain": yTrain,
            "xTest": xTest,
            "yTest": yTest
        }

    def gradePlayers(self, stats):
        playerProbs = np.concatenate((self.players, stats), axis=1)

        '''
        sorts players by probability (desc)
        player with highest MVP probability is first row
        '''
        return playerProbs[playerProbs[:, 3].argsort()[::-1]]
