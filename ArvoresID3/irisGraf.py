import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

data = {
    '[1.00-2.16)': {'Iris-setosa': 37},
    '[2.16-3.32)': {'Iris-setosa': 13},
    '[3.32-4.48)': {'Iris-versicolor': 25},
    '[4.48-5.63)': {
        'Iris-versicolor': 23, 
        'Iris-virginica': 6    
    },
    '[5.63-6.79)': {
        'Iris-versicolor': 1,
        'Iris-virginica': 18  
    },
    '[6.79-7.95)': {'Iris-virginica': 25}
}

categories = list(data.keys())
class_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
counts = defaultdict(list)

for cat in categories:
    for label in class_labels:
        counts[label].append(data[cat].get(label, 0))

fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.25
bar_positions = range(len(categories))

for i, label in enumerate(class_labels):
    ax.bar(
        [p + bar_width * i for p in bar_positions], 
        counts[label], 
        width=bar_width, 
        label=label
    )

ax.set_xlabel('Petallength')
ax.set_ylabel('Número de Amostras')
ax.set_title('Distribuição das Iris por Petallength')
ax.set_xticks([p + bar_width for p in bar_positions])
ax.set_xticklabels(categories)
ax.legend()

plt.show()


