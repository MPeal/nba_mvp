This is a simple logistic regression exercise designed to predict the NBA MVP for a given season.  It is a spin-off of a 2017 graduate school project I did related to predicting All-NBA teams for a given season with linear regression.

The code uses two datasets: Season_Stats_MVP.csv and Wins.csv.  The first file contains unofficial regular season stats for every NBA player between 1956 and 2017; the dataset was obtained from Kaggle.  The second file simply holds records of NBA teams and their records for each season between the 2012-13 season and the 2016-17 season; I created this dataset manually.  Through code, these records are joined onto the main file so each player has a 'Wins' statistic, which correlates to how many games their team won in a season.  MVPs typically come from a handful of the best teams each year, which is why that statistic is an important factor.

This example only makes practical use of the 2013 through 2017 seasons.  When I did the original project in 2017, the hypothesis was, the playstyle and statistics of the NBA had changed so drastically since 2013 that the criteria for making an All-NBA team or winning the MVP in, say, 2015, was totally different from the criteria in, say, 1992.  Thus, to keep things as consistent as possible, I leveraged the most "modern" data.

Each player has a 'class' or 'label,' which simply conveys whether or not they won the MVP award for a given season.  All MVPs are labelled with a 1, all non-MVPs a 0.

The model runs through the MainMVP.py file.  The program predicts one year at a time.  That year can be set on MainMVP.py line 21, passing in an integer between 2013 and 2017 (inclusive).  The result simply prints the model's predicted MVP and the year's actual MVP for comparison.

Note:  the model makes use of a few packages, like numpy and pandas, so those will need to be present before running anything.