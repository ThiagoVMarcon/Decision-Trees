import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

data = {
    'sunny': {
        '[79.39-93.79)': {'no': 1},
        '[93.79-108.18)': {'no': 2},
        '[65.00-79.39)': {'yes': 2}
    },
    'overcast': {
        'yes': 4
    },
    'rainy': {
        'False': {'yes': 3},
        'True': {'no': 2}
    }
}

weather_conditions = ['sunny', 'overcast', 'rainy']
play_labels = ['yes', 'no']
counts = defaultdict(lambda: [0, 0])  

for weather, details in data.items():
    if weather == 'overcast':
        counts[weather][0] = details['yes']
    else:
        for subkey, result in details.items():
            for play, count in result.items():
                if play == 'yes':
                    counts[weather][0] += count
                elif play == 'no':
                    counts[weather][1] += count

yes_counts = [counts[weather][0] for weather in weather_conditions]
no_counts = [counts[weather][1] for weather in weather_conditions]

bar_width = 0.35
bar_positions = range(len(weather_conditions))

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(bar_positions, yes_counts, bar_width, label='yes')
bars2 = ax.bar([p + bar_width for p in bar_positions], no_counts, bar_width, label='no')

ax.set_xlabel('Weather')
ax.set_ylabel('Número de Amostras')
ax.set_title('Contagem de decisões consoante a Condição Climática')
ax.set_xticks([p + bar_width / 2 for p in bar_positions])
ax.set_xticklabels(weather_conditions)
ax.legend()

plt.show()
