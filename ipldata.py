import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt

df_matches = pd.read_csv(r'matches.csv')
# df_matches = df_matches.isnull().sum()
# df_matches = df_matches.Season.unique()
pom = df_matches.player_of_match.value_counts().nlargest(5)
plt.figure(figsize=(10, 5))
sns.barplot(pom.index, pom)
plt.xlabel('Players Name')
plt.ylabel('Count')

result_label = df_matches.result.value_counts()
plt.figure(figsize=(10, 5))
sns.barplot(result_label.index, result_label, log=True)
plt.xlabel('Results')
plt.ylabel('Count')


plt.figure(figsize=(8, 5))
sns.countplot(df_matches.toss_decision)

# groupby by winner
toss_winner = df_matches.groupby('winner')['toss_decision'].value_counts()
print(toss_winner)

# barplot of most toss winners

toss_win_label = df_matches.toss_winner.value_counts()
plt.figure(figsize=(10, 8))
sns.barplot(toss_win_label, toss_win_label.index)
plt.ylabel('Teams')
plt.xlabel('Count'),
plt.title('Toss Winners')

# teams did fielding first and result was normal
field_first = df_matches[(df_matches['toss_decision'] == 'field') & (df_matches['result'] == 'normal')].head()
print(field_first)

# teams did batting first and won
win = df_matches[(df_matches.toss_decision == 'bat') & (df_matches.win_by_runs != 0)]['winner'].value_counts().sort_values(ascending=False)
plt.figure(figsize=(10, 5))
sns.barplot(win, win.index)
plt.xlabel('Count')
plt.ylabel('Teams')

# teams did bowling first and won
win = df_matches[(df_matches.toss_decision == 'field') & (df_matches.win_by_wickets != 0)]['winner'].value_counts().sort_values(ascending=False)
plt.figure(figsize=(10, 5))
sns.barplot(win, win.index)
plt.xlabel('Count')
plt.ylabel('Teams')

# barplot of won_by_wickets
plt.figure(figsize=(10, 5))
sns.barplot(df_matches.win_by_wickets.value_counts(), df_matches.win_by_wickets.value_counts().index)

# number of matches played each year
season = df_matches['Season'].value_counts()
plt.figure(figsize=(10, 5))
sns.barplot(season, season.index)
plt.ylabel('IPL Year')
plt.xlabel('Count of Numberof Matches Played')

# number of matches played in top 10 city
city = df_matches['city'].value_counts().sort_values().nlargest(10)
plt.figure(figsize=(10, 5))
sns.barplot(city, city.index)
plt.ylabel('City')
plt.xlabel('Count of Number Matches Played')

plt.show()