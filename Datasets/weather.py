import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset do arquivo CSV
weather_df = pd.read_csv('weather.csv')

# Cria uma figura e os subplots com 2 linhas e 2 colunas
fig, eixos = plt.subplots(2, 2, figsize=(8, 8))

#Gráfico de Relação entre Humidity e Temp
sns.scatterplot(data=weather_df, x='Humidity', y='Temp', ax=eixos[0, 0])
eixos[0, 0].set_title('Gráfico de Humidity e Temp')
eixos[0, 0].set_xlabel('Humidity')
eixos[0, 0].set_ylabel('Temp')
eixos[0, 0].grid(True)

#Gráfico circular de contagem de Weather
weather_counts = weather_df['Weather'].value_counts()
eixos[0, 1].pie(weather_counts, labels=weather_counts.index, autopct='%1.1f%%')
eixos[0, 1].set_title('Partição de Weather')
eixos[0, 1].axis('equal')  # Mantém o aspecto circular

#Gráfico de Relação entre Weather, Temp e Play
custom_palette = {'yes': 'green', 'no': 'red'}  # Define a cor verde para 'yes' e vermelho para 'no'
sns.scatterplot(data=weather_df, x='Temp', y='Weather', hue='Play', palette=custom_palette, ax=eixos[1, 0], s=100)
eixos[1, 0].set_title('Gráfico de relação entre Weather, Temp e Play')
eixos[1, 0].set_xlabel('Temp')
eixos[1, 0].set_ylabel('Weather')
eixos[1, 0].legend(title='Play', bbox_to_anchor=(1.05, 1), loc='upper left')

#Remove o subplot que está vazio
fig.delaxes(eixos[1, 1])

plt.tight_layout()
plt.show()

