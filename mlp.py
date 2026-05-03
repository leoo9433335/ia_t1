import pandas as pd
import joblib

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =========================
# 📥 Carregar datasets
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
# 🧠 Pipeline (NORMALIZAÇÃO + MLP)
# =========================
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(32,),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=32,
        learning_rate='constant',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10
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

print("\nTeste:")
print(classification_report(y_test, y_test_pred))
print(f"Acurácia: {accuracy_score(y_test, y_test_pred):.4f}")

# =========================
# 💾 Salvar modelo
# =========================
joblib.dump(pipeline, 'mlp_model.pkl')