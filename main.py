import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random

# ==================== CONFIGURAÇÃO ====================
CLASSE_MAPEAMENTO = {
    0: "Tem jogo",
    1: "X venceu",
    2: "O venceu",
    3: "Empate"
}

random.seed(42)
np.random.seed(42)

# =========================
# 🧠 REGRAS DO JOGO
# =========================

def check_win(board):
    lines = [
        [0,1,2], [3,4,5], [6,7,8],
        [0,3,6], [1,4,7], [2,5,8],
        [0,4,8], [2,4,6]
    ]
    for line in lines:
        values = [board[i] for i in line]
        if values == [1,1,1]:
            return 1
        elif values == [-1,-1,-1]:
            return 2
    return 0

def classificar_estado(board):
    win = check_win(board)
    has_blank = 0 in board
    
    if win == 1:
        return 1
    elif win == 2:
        return 2
    elif not has_blank:
        return 3
    else:
        return 0

# =========================
# 🔥 GERAR ESTADOS VÁLIDOS (MELHORADO)
# =========================

def gerar_estados_validos(n=200):
    estados = []

    while len(estados) < n:
        board = [0] * 9
        jogador = 1

        # 🔥 ESSENCIAL: número aleatório de jogadas
        num_jogadas = random.randint(1, 9)

        for _ in range(num_jogadas):
            livres = [i for i in range(9) if board[i] == 0]
            if not livres:
                break

            pos = random.choice(livres)
            board[pos] = jogador

            # valida: diferença entre X e O no máximo 1
            if abs(board.count(1) - board.count(-1)) > 1:
                break

            # se alguém ganhou, parar
            if check_win(board) != 0:
                break

            jogador *= -1

        estados.append(board.copy())

    return estados
# =========================
# 🧠 FEATURE ENGINEERING (NOVO)
# =========================

def add_features(df):
    df = df.copy()
    df["count_X"] = (df.iloc[:, :9] == 1).sum(axis=1)
    df["count_O"] = (df.iloc[:, :9] == -1).sum(axis=1)
    df["diff"] = df["count_X"] - df["count_O"]
    return df

print("=" * 70)
print("PROCESSAMENTO DATASET - TIC TAC TOE (VERSÃO PROFISSIONAL)")
print("=" * 70)

# =========================
# 📥 CARREGAR DATASET ORIGINAL
# =========================

print("\n[1] Carregando dataset original...")
df = pd.read_csv('tic-tac-toe.data', header=None)

mapping = {'x': 1, 'o': -1, 'b': 0}
df_features = df.iloc[:, :9].apply(lambda x: x.map(mapping))

new_data = []

for _, row in df_features.iterrows():
    board = row.values.tolist()
    label = classificar_estado(board)
    new_data.append(board + [label])

print(f"    ✓ Dados originais: {len(new_data)}")

# =========================
# 🔥 GERAR DADOS VÁLIDOS
# =========================

print("\n[2] Gerando estados válidos adicionais...")
novos_estados = gerar_estados_validos(200)

for board in novos_estados:
    label = classificar_estado(board)
    new_data.append(board + [label])

# Criar DataFrame
new_df = pd.DataFrame(
    new_data,
    columns=[f'pos_{i}' for i in range(9)] + ['class']
)

# Remover duplicatas
new_df = new_df.drop_duplicates()

# =========================
# 📊 DISTRIBUIÇÃO ORIGINAL
# =========================

print("\n[3] Distribuição antes do balanceamento:")
print(new_df['class'].value_counts())

# =========================
# ⚖️ BALANCEAMENTO MELHORADO
# =========================

print("\n[4] Balanceamento inteligente...")

class_counts = new_df['class'].value_counts()
target_size = class_counts.max()

balanced_list = []

for c in class_counts.index:
    df_class = new_df[new_df['class'] == c]
    
    if len(df_class) < target_size:
        df_class = df_class.sample(target_size, replace=True, random_state=42)
    else:
        df_class = df_class.sample(target_size, random_state=42)
    
    balanced_list.append(df_class)

balanced_df = pd.concat(balanced_list)
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nDistribuição após balanceamento:")
print(balanced_df['class'].value_counts())

# =========================
# 🧠 FEATURE ENGINEERING
# =========================

balanced_df = add_features(balanced_df)

# =========================
# ✂️ SPLIT TREINO / VAL / TESTE
# =========================

print("\n[5] Dividindo dataset...")

X = balanced_df.drop(columns=["class"])
y = balanced_df["class"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
val_df = pd.concat([X_val.reset_index(drop=True), y_val.reset_index(drop=True)], axis=1)
test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

# =========================
# 💾 SALVAR
# =========================

train_df.to_csv('train_32.csv', index=False)
val_df.to_csv('validation_32.csv', index=False)
test_df.to_csv('test_32.csv', index=False)

print(f"\nTreino: {len(train_df)}")
print(f"Validação: {len(val_df)}")
print(f"Teste: {len(test_df)}")

# =========================
# 📊 DISTRIBUIÇÃO FINAL
# =========================

print("\n[6] Distribuição final:")

for nome, df_split in [("Treino", train_df), ("Validação", val_df), ("Teste", test_df)]:
    print(f"\n{nome}:")
    print(df_split['class'].value_counts())

print("\n" + "=" * 70)
print("✓ DATASET PRONTO PARA TREINAMENTO (VERSÃO FORTE)")
print("=" * 70)