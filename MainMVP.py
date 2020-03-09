import numpy as np
from SeasonDataWithMVP import SeasonDataWithMVP
from sklearn.linear_model import LogisticRegression


mvps = {
    "2013": "LeBron James",
    "2014": "Kevin Durant",
    "2015": "Stephen Curry",
    "2016": "Stephen Curry",
    "2017": "Russell Westbrook"
}

d = SeasonDataWithMVP()

try:
    # loads player data for seasons ending in 2013 to 2017 (inclusive)
    data = d.getMultiYearDataset(True)

    # set year to predict All-NBA Teams
    d.setYearToPredict(2013)

    # get training and test sets
    predictionVars = d.train(data)
    xTrain = predictionVars['xTrain']
    yTrain = predictionVars['yTrain']
    xTest = predictionVars['xTest']
    yTest = predictionVars['yTest']

    # make predictions w/ probability
    logreg = LogisticRegression()
    logreg.fit(xTrain, yTrain)
    probabilities = logreg.predict_proba(xTest)[:, 1]
    probabilities = np.reshape(probabilities, (len(probabilities), 1))

    # re-attach player info to probabilities
    playerProbs = d.gradePlayers(probabilities)

    mvp = playerProbs[0]  # MVP is player with the highest probability of being MVP
    print("Predicted MVP for " + str(d.year) + ": " + mvp[0])
    actualMVP = mvps[str(int(d.year))]
    print("Actual MVP for " + str(d.year) + ": " + actualMVP)
except AssertionError as e:
    print("An exception occurred")
