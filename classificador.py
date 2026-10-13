import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

os.makedirs("figures", exist_ok=True)

df = sns.load_dataset('iris') # iris é tão classico que não precisa nem baixar o dataset, já vem incluido

print("--- Frequência de cada categoria ---")
print(df['species'].value_counts())

print("\n--- Estatísticas Gerais (Média, Desvio Padrão, etc.) ---")
print(df.describe())

print("\n--- Moda Geral ---")
print(df.drop(columns=['species']).mode().iloc[0])

print("\n--- Moda por Categoria ---")
colunas_numericas = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
moda_categoria = df.groupby('species')[colunas_numericas].apply(lambda x: x.mode().iloc[0])
print(moda_categoria)

print("\n--- Estatísticas por Categoria ---")
print(df.groupby('species').describe())
df.groupby('species').describe().to_csv("estatisticas_categoria.csv")


plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species', s=80)
plt.title('Relação: Comprimento vs Largura da Sépala')
plt.savefig("figures/sepalas.pdf", dpi=300, bbox_inches="tight")

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species', s=80)
plt.title('Relação: Comprimento vs Largura da Pétala')
plt.savefig("figures/petalas.pdf", dpi=300, bbox_inches="tight")

grid = sns.pairplot(df, hue='species', markers=["o", "s", "D"])
grid.figure.suptitle("Distribuição das 4 Dimensões do Dataset Iris", y=1.02)
grid.savefig("figures/pairplot.pdf", dpi=300, bbox_inches="tight")

features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = df[features]
y = df['species']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['species'] = y

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='species', palette='viridis', s=100)
plt.title(f"PCA - Iris Dataset\nVariância Explicada: {sum(pca.explained_variance_ratio_) * 100:.2f}%")
plt.savefig("figures/pca.pdf", dpi=300, bbox_inches="tight")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42, stratify=y)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("\n--- Resultados da Classificação K-NN ---")
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

plt.show()
