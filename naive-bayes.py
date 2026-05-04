import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# Carregar datasets

train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('validation.csv')
test_df = pd.read_csv('test.csv')

X_train = train_df.iloc[:, :9]
y_train = train_df['class']

X_val = val_df.iloc[:, :9]
y_val = val_df['class']

X_test = test_df.iloc[:, :9]
y_test = test_df['class']

# Juntar treino + validação

X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

# Pipeline

pipeline = Pipeline([
 ('rf', RandomForestClassifier(random_state=42))
])

# Grid Search (ajuste fino)

param_grid = {
 'rf__n_estimators': [100, 200],
 'rf__max_depth': [None, 10, 20],
 'rf__min_samples_split': [2, 5],
 'rf__min_samples_leaf': [1, 2],
 'rf__class_weight': ['balanced']
}

grid = GridSearchCV(
 pipeline,
 param_grid,
 cv=5,
 scoring='f1_weighted',
 n_jobs=-1,
 verbose=1
)

# Treinar modelo

grid.fit(X_train_val, y_train_val)

best_model = grid.best_estimator_

print("\nMelhores parâmetros encontrados:")
print(grid.best_params_)

# Avaliação final

y_test_pred = best_model.predict(X_test)

CLASSE_MAPEAMENTO = {
 0: "Tem jogo",
 1: "X venceu",
 2: "O venceu",
 3: "Empate"
}

target_names = [CLASSE_MAPEAMENTO[i] for i in sorted(CLASSE_MAPEAMENTO)]

print("\nTeste:")
print(classification_report(y_test, y_test_pred, target_names=target_names))
print(f"Acurácia: {accuracy_score(y_test, y_test_pred):.4f}")

# Salvar modelo

joblib.dump(best_model, 'random_forest_model.pkl')