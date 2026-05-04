import pandas as pd
import joblib

from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =========================
# 📥 Carregar datasetsq
# =========================
train_df = pd.read_csv('train_32.csv')
val_df = pd.read_csv('validation_32.csv')
test_df = pd.read_csv('test_32.csv')

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
# 🧠 Pipeline (NORMALIZAÇÃO + PERCEPTRON)
# =========================
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('perceptron', Perceptron(
        max_iter=1000,
        eta0=0.1,
        tol=1e-3,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ))
])

# =========================
# 🚀 Treinar modelo
# =========================
pipeline.fit(X_train_val, y_train_val)

# =========================
# 🧪 Avaliação final
# =========================
y_test_pred = pipeline.predict(X_test)

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

# =========================
# 💾 Salvar modelo
# =========================
joblib.dump(pipeline, 'perceptron_model.pkl')