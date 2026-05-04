import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report


# ---------------------------
# Leitura dos datasets
# ---------------------------
def carregar_datasets():
    train_df = pd.read_csv('train_32.csv')
    val_df = pd.read_csv('validation_32.csv')
    test_df = pd.read_csv('test_32.csv')

    X_train = train_df.iloc[:, :9]
    y_train = train_df['class']

    X_val = val_df.iloc[:, :9]
    y_val = val_df['class']

    X_test = test_df.iloc[:, :9]
    y_test = test_df['class']

    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------
# Treinar modelo
# ---------------------------
def treinar_knn(X_train, y_train, k):
    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(X_train, y_train)
    
    return modelo


# ---------------------------
# Avaliar modelo
# ---------------------------
def avaliar_modelo(modelo, X, y):
    y_pred = modelo.predict(X)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

    return acc, prec, rec, f1


# ---------------------------
# Buscar melhor K (validação)
# ---------------------------
def buscar_melhor_k(X_train, y_train, X_val, y_val):
    melhor_k = 1
    melhor_f1 = 0

    print("\nBuscando melhor K...\n")

    for k in range(1, 21, 2):
        modelo = treinar_knn(X_train, y_train, k)
        _, _, _, f1 = avaliar_modelo(modelo, X_val, y_val)

        print(f"K={k} -> F1={f1:.4f}")

        if f1 > melhor_f1:
            melhor_f1 = f1
            melhor_k = k

    return melhor_k

# ---------------------------
# Função para criar modelo PKL para testes
# ---------------------------

def salvar_modelo_knn(modelo, caminho='knn_model.pkl'):
        joblib.dump(modelo, caminho)
        print(f"Modelo KNN salvo em: {caminho}")


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":

    #  Leitura dos dados 
    X_train, y_train, X_val, y_val, X_test, y_test = carregar_datasets()

    #  Buscar melhor K
    melhor_k = buscar_melhor_k(X_train, y_train, X_val, y_val)

    print(f"\nMelhor K encontrado: {melhor_k}")

    #  Treinar modelo final
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    modelo_final = treinar_knn(X_train_full, y_train_full, melhor_k)

    #  Avaliar no teste
    acc, prec, rec, f1 = avaliar_modelo(modelo_final, X_test, y_test)

    print("\nResultados no conjunto de TESTE:")
    print(f"Acurácia: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, modelo_final.predict(X_test)))

    # Salva modelo PKL para teste do jogo
    salvar_modelo_knn(modelo_final)