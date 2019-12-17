import os
import pandas as pd
import matplotlib.pyplot as plt

games_1 = 'experiments/all-games-1/run_002/'
games_2 = 'experiments/all-games/run_000/'

name_mapping = {}

for d in os.listdir(games_1):
    if d.startswith('logging_'):
        path = os.path.join(games_1, d)
        files = list(os.listdir(path))
        name_mapping[files[0]] = d[8:]

for d in os.listdir(games_2):
    if d.startswith('logging_'):
        path = os.path.join(games_2, d)
        files = list(os.listdir(path))
        name_mapping[files[0]] = d[8:]

results = pd.read_csv('all_result.csv')

for i, entry in enumerate(results['Unnamed: 0']):
    results.iloc[i, 0] = name_mapping[entry]

final_performance = results[['Unnamed: 0', 'Unnamed: 1', '19350']].loc[results['Unnamed: 2'] != 'wall_time']

loss = final_performance.loc[final_performance['Unnamed: 1'] == 'loss']
kl_loss = final_performance.loc[final_performance['Unnamed: 1'] == 'kl_loss']
p_x = final_performance.loc[final_performance['Unnamed: 1'] == 'p_x_loss']
mask_loss = final_performance.loc[final_performance['Unnamed: 1'] == 'mask_loss']

print(p_x)
p_x = p_x.sort_values(by='19350')

print('Worst games')
print(p_x.iloc[:10])

print('Best games')
print(p_x.iloc[-10:])
