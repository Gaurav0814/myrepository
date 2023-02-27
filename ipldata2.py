import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt

# analysis on deliveries data file
df_deliveries = pd.read_csv(r'deliveries.csv')
# top 5 rows
df_deliveries = df_deliveries.head()
print(df_deliveries)
# shape of data
shape_of_data = df_deliveries.shape
print(shape_of_data)

# match id 1
df_match1 = df_deliveries[df_deliveries.match_id == 1]
print(df_match1)
# top 5 rows
df_match1 = df_match1.head()
print(df_match1)
# shape of match id 1
shape_of_m_id = df_match1.shape
print(shape_of_m_id)
# unique batting team
uniquebt = df_match1.batting_team.unique()
print(uniquebt)

# 1st inning
srh = df_match1[df_match1['inning'] == 1].head()
print(srh)
# dismissalkind
dismissal = srh['dismissal_kind'].value_counts()
print(dismissal)
# total number of balls bowled by srh
total_balls_by_srh = len(srh.ball)
print(total_balls_by_srh)

# count of 4s hit by srh
num_of_fours = len(srh[srh.total_runs == 4])
print(num_of_fours)

# count of 6s hit by srh
num_of_sixes = len(srh[srh.total_runs == 6])
print(num_of_sixes)
# 2nd innings
rcb = df_match1[df_match1.inning == 2].head()
print(rcb)
# dismissalkind
dismissal_by_rcb = rcb['dismissal_kind'].value_counts()
print(dismissal_by_rcb)
# total number of balls bowled by rcb
total_balls_by_rcb =  len(rcb.ball)
print(total_balls_by_rcb)
# count of 4s hit by rcb
count_of_fours = len(rcb[rcb.total_runs == 4])
print(count_of_fours)
# count of 6s hit by rcb
count_of_sixes = len(rcb[rcb.total_runs == 6])
print(count_of_sixes)