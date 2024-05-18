import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Carrega o dataset do arquivo CSV
df = pd.read_csv('restaurant.csv')

#Substitui os valores na coluna 'Price'
price = {'$': 'Barato', '$$': 'Médio', '$$$': 'Caro'}
df['Price'] = df['Price'].map(price)

#Cria uma figura e os subplots com 2 linhas e 2 colunas
fig, eixos = plt.subplots(2, 2, figsize=(8, 8))

#Gráfico de contagem dos tipos de Cozinha (Type)
sns.countplot(data=df, x='Type', ax=eixos[0, 0])
eixos[0, 0].set_title('Contagem por Tipo de Cozinha')
eixos[0, 0].set_xlabel('Tipo de Cozinha')
eixos[0, 0].set_ylabel('Contagem')
eixos[0, 0].tick_params(axis='x', rotation=45)

# Gráfico de Relação entre Price e Type
sns.boxplot(data=df, x='Type', y='Price', ax=eixos[0, 1])
eixos[0, 1].set_title('Relação entre Preço e Tipo de Cozinha')
eixos[0, 1].set_xlabel('Tipo de Cozinha')
eixos[0, 1].set_ylabel('Preço')

# Gráfico de Relação entre Price, Est e Class
sns.scatterplot(data=df, x='Price', y='Est', hue='Class', palette='Set1', ax=eixos[1, 0])
eixos[1, 0].set_title('Relação entre Preço, Tempo Estimado e Classificação')
eixos[1, 0].set_xlabel('Preço')
eixos[1, 0].set_ylabel('Tempo Estimado')
eixos[1, 0].legend(title='Classificação', bbox_to_anchor=(1.05, 1), loc='upper left')

# Remove o subplot que está vazio
fig.delaxes(eixos[1, 1])

plt.tight_layout()
plt.show()



