import pandas as pd

# Analisar dataset original
data_path = 'tic-tac-toe.data'
df = pd.read_csv(data_path, header=None)

print("Dataset original:")
print(df.iloc[:, 9].value_counts())
print(f"\nTotal de linhas: {len(df)}")

# Analisar as colunas
mapping = {'x': 1, 'o': -1, 'b': 0}
df_features = df.iloc[:, :9].apply(lambda x: x.map(mapping))

def check_win(board):
    lines = [
        [0,1,2], [3,4,5], [6,7,8],
        [0,3,6], [1,4,7], [2,5,8],
        [0,4,8], [2,4,6]
    ]
    for line in lines:
        s = sum(board[i] for i in line)
        if s == 3:
            return 1
        elif s == -3:
            return 2
    return 0

class_count = {0: 0, 1: 0, 2: 0, 3: 0}
for idx, row in df_features.iterrows():
    board = row.values.tolist()
    win = check_win(board)
    has_blank = 0 in board
    if win == 1:
        label = 1
    elif win == 2:
        label = 2
    elif not has_blank:
        label = 3
    else:
        label = 0
    class_count[label] += 1

print("\nClasses reprocessadas:")
for cls, count in sorted(class_count.items()):
    class_names = {0: "Tem jogo", 1: "X venceu", 2: "O venceu", 3: "Empate"}
    print(f"  {cls} - {class_names[cls]}: {count}")
