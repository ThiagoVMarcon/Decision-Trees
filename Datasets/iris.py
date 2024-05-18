import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carrega o dataset
df = pd.read_csv('iris.csv')

# Exibe as primeiras linhas do dataset
print(df.head())

# Definir um dicionário de cores personalizado
custom_palette = {'Iris-setosa': '#1f77b4', 'Iris-versicolor': '#ff7f0e', 'Iris-virginica': '#2ca02c'}

# Configura a figura com 3 subplots em uma grade de 2x2
fig, eixos = plt.subplots(2, 2, figsize=(12, 12))

# Gráfico (sepallength, sepalwidth)
sns.scatterplot(data=df, x='sepallength', y='sepalwidth', hue='class', palette=custom_palette, ax=eixos[0, 0])
eixos[0, 0].set_title('Gráfico de relação entre Sepal Length/Sepal Width')
eixos[0, 0].set_xlabel('Sepal Length')
eixos[0, 0].set_ylabel('Sepal Width')

# Gráfico (petallength, petalwidth)
sns.scatterplot(data=df, x='petallength', y='petalwidth', hue='class', palette=custom_palette, ax=eixos[0, 1])
eixos[0, 1].set_title('Gráfico de relação entre Petal Length/Petal Width')
eixos[0, 1].set_xlabel('Petal Length')
eixos[0, 1].set_ylabel('Petal Width')

# Gráfico da distribuição de class
class_counts = df['class'].value_counts()
eixos[1, 0].pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', colors=[custom_palette[key] for key in class_counts.index])
eixos[1, 0].set_title('Distribuição de Class pelo tipo de Iris')
eixos[1, 0].legend(title='Class', loc='center left', bbox_to_anchor=(1, 0.5))  # Movendo a legenda para fora do gráfico

# Remove o subplot vazio
fig.delaxes(eixos[1, 1])

plt.tight_layout()
plt.show()