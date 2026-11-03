import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# 1. Carregamento dos dados
df = pd.read_csv("base_sintetica_experimental_v2.csv")

# 2. Separação de Features (X) e Target (y)
X = df[['idade', 'queixa_principal', 'frequencia_cardiaca', 'frequencia_respiratoria', 'pressao_sistolica', 'pressao_diastolica', 'temperatura', 'saturacao_oxigenio']]
y = df['prioridade_sintetica']

# 3. One-Hot Encoding para variáveis categóricas
X_encoded = pd.get_dummies(X, columns=['queixa_principal'], drop_first=True)

# 4. Divisão Treino e Teste (80/20 com estratificação)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

# 5. Padronização das variáveis numéricas
num_cols = ['idade', 'frequencia_cardiaca', 'frequencia_respiratoria', 'pressao_sistolica', 'pressao_diastolica', 'temperatura', 'saturacao_oxigenio']
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# 6. Instanciação e Treinamento dos Modelos
models = {
    "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42),
    "Árvore de Decisão": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"================ {name} ================")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))