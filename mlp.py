import pandas as pd
import joblib

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =========================
# 📥 Carregar datasets
# =========================
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('validation.csv')
test_df = pd.read_csv('test.csv')

X_train = train_df.iloc[:, :9]
y_train = train_df['class']

X_val = val_df.iloc[:, :9]
y_val = val_df['class']

X_test = test_df.iloc[:, :9]
y_test = test_df['class']

# =========================
# 🔗 Juntar treino + validação
# =========================
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

# =========================
# 🧠 Pipeline (NORMALIZAÇÃO + MLP)
# =========================
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(
        max_iter=2000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10
    ))
])

# =========================
# ⚙️ Grid de hiperparâmetros
# =========================
param_grid = {
    'mlp__hidden_layer_sizes': [
        (32,), (64,), (100,),
        (64, 32), (100, 50)
    ],
    'mlp__activation': ['relu', 'tanh'],
    'mlp__solver': ['adam', 'sgd'],
    'mlp__alpha': [0.0001, 0.001, 0.01, 0.1],
    'mlp__learning_rate': ['constant', 'adaptive'],
    'mlp__batch_size': [16, 32, 64]
}

# =========================
# 🔍 GridSearch
# =========================
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train_val, y_train_val)

print("\nMelhores parâmetros:")
print(grid.best_params_)

# =========================
# 🧪 Avaliação final
# =========================
best_model = grid.best_estimator_

y_test_pred = best_model.predict(X_test)

print("\nTeste:")
print(classification_report(y_test, y_test_pred))
print(f"Acurácia: {accuracy_score(y_test, y_test_pred):.4f}")

# =========================
# 💾 Salvar modelo
# =========================
joblib.dump(best_model, 'mlp_model.pkl')
