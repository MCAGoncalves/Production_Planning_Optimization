import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

# Carregamento do conjunto de dados
database = pd.read_excel('https://github.com/MCAGoncalves/dataset/blob/main/sales_data_sample_filtrada.xlsx?raw=true')

# Preparação dos dados: separação de variáveis independentes (X) e dependente (y)
X = database.drop(['SALES', 'Unnamed: 10', 'Unnamed: 11'], axis=1)
y = database['SALES']

# Divisão dos dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, shuffle=True, test_size=0.3)

# Inicialização dos modelos de Machine Learning
models = {
    "KNN": KNeighborsRegressor(n_neighbors=20),
    "DT": DecisionTreeRegressor(random_state=0)
}

# Lista para armazenar previsões e erros
predictions = {}
errors = {"MAE": [], "MAPE": [], "MSE": [], "RMSE": []}

# Treinamento dos modelos e cálculo das métricas de erro
for name, model in models.items():
    # Treinamento do modelo
    model.fit(X_train, y_train)
    # Previsão com o conjunto de teste
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    
    # Cálculo das métricas de erro
    errors["MAE"].append(metrics.mean_absolute_error(y_test, y_pred))
    errors["MAPE"].append(metrics.mean_absolute_percentage_error(y_test, y_pred))
    errors["MSE"].append(metrics.mean_squared_error(y_test, y_pred))
    errors["RMSE"].append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Criação do DataFrame para armazenar as métricas de erro
error_df = pd.DataFrame(errors, index=models.keys())
print("Métricas de erro para cada modelo:")
print(error_df)

# Seleção do modelo com o menor MAPE
best_model_name = error_df['MAPE'].idxmin()
best_model_prediction = predictions[best_model_name]

# Comparação do modelo selecionado com o conjunto de teste
plt.figure(figsize=(10, 6))
plt.plot(best_model_prediction, color='orange', linestyle='--', label='Previsão')
plt.plot(y_test.values, color='blue', label='Conjunto de Teste')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Sales')
plt.title(f'Comparação entre a previsão do {best_model_name} e o conjunto de teste')
plt.grid(True)
plt.show()

# Cálculo dos erros de previsão para o modelo selecionado
prediction_errors = y_test.values - best_model_prediction

# Gráfico dos erros de previsão
plt.figure(figsize=(10, 6))
plt.plot(prediction_errors, color='red', linestyle=':', label='Erro')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Erro de Previsão')
plt.title(f'Erros de previsão entre {best_model_name} e conjunto de teste')
plt.grid(True)
plt.show()
