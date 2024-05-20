import matplotlib.pyplot as plt
from collections import defaultdict

data = {
    'Some': {'Yes': 4},
    'Full': {
        'Hun_Yes': {
            'Type_Thai': {'Fri_No': {'No': 1}, 'Fri_Yes': {'Yes': 1}},
            'Type_Italian': {'No': 1},
            'Type_Burger': {'Yes': 1}
        },
        'Hun_No': {'No': 2}
    },
    'Nan': {'No': 2}
}

pat_conditions = ['Some', 'Full', 'Nan']
class_labels = ['Yes', 'No']
counts = defaultdict(lambda: [0, 0]) 

for pat, details in data.items():
    if pat == 'Some':
        counts[pat][0] = details['Yes']
    elif pat == 'Full':
        for hun_key, hun_details in details.items():
            if isinstance(hun_details, dict):
                for type_key, type_details in hun_details.items():
                    if isinstance(type_details, dict):
                        for fri_key, fri_details in type_details.items():
                            if isinstance(fri_details, dict):
                                for play, count in fri_details.items():
                                    if play == 'Yes':
                                        counts[pat][0] += count
                                    elif play == 'No':
                                        counts[pat][1] += count
                            else:
                                if fri_key == 'Yes':
                                    counts[pat][0] += fri_details
                                elif fri_key == 'No':
                                    counts[pat][1] += fri_details
                    else:
                        if type_key == 'Yes':
                            counts[pat][0] += type_details
                        elif type_key == 'No':
                            counts[pat][1] += type_details
            else:
                if hun_key == 'Yes':
                    counts[pat][0] += hun_details
                elif hun_key == 'No':
                    counts[pat][1] += hun_details
    elif pat == 'Nan':
        counts[pat][1] = details['No']

yes_counts = [counts[pat][0] for pat in pat_conditions]
no_counts = [counts[pat][1] for pat in pat_conditions]

bar_width = 0.35
bar_positions = range(len(pat_conditions))

fig, ex = plt.subplots(figsize=(10, 6))

bars1 = ex.bar(bar_positions, yes_counts, bar_width, label='Yes')
bars2 = ex.bar([p + bar_width for p in bar_positions], no_counts, bar_width, label='No')

ex.set_xlabel('Pat')
ex.set_ylabel('Número de Amostras')
ex.set_title('Distribuição das Decisões por Pat')
ex.set_xticks([p + bar_width / 2 for p in bar_positions])
ex.set_xticklabels(pat_conditions)
ex.legend()

plt.show()



