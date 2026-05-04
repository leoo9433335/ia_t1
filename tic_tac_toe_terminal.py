import os
import joblib
import pandas as pd

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_board(board):
    print("\n")
    for i in range(3):
        print(" | ".join(board[i * 3:(i + 1) * 3]))
        if i < 2:
            print("---+---+---")

def check_winner(board, player):
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ]
    for condition in win_conditions:
        if all(board[pos] == player for pos in condition):
            return True
    return False

def is_draw(board):
    return all(cell != " " for cell in board)

# 🔁 Traduz saída do modelo
def traduzir_saida(pred):
    if pred == 0:
        return "Jogo em andamento"
    elif pred == 1:
        return "X vence"
    elif pred == 2:
        return "O vence"
    else:
        return "Desconhecido"

def predict_game_state(board):
    input_numeric = [1 if cell == "X" else -1 if cell == "O" else 0 for cell in board]

    # Sem colunas (igual ao treino com iloc)
    input_df = pd.DataFrame([input_numeric])

    #Descomentar o modelo que deseja testar
    prediction = mlp_model.predict(input_df)[0]
    #prediction = knn_model.predict(input_df)[0]
    return prediction

# 📦 Carregar modelo    
# Descomentar o modelo que deseja testar
mlp_model = joblib.load('mlp_model.pkl')
#knn_model = joblib.load('knn_model.pkl')

def tic_tac_toe():
    board = [" " for _ in range(9)]
    players = ["X", "O"]
    current_player = 0

    while True:
        clear_screen()
        print_board(board)

        # 🔥 Agora prevê sempre que houver alguma jogada
        if any(cell != " " for cell in board):
            state = predict_game_state(board)
            print(f"\n🧠 Estado previsto: {traduzir_saida(state)}\n")

        # Input do jogador
        try:
            move = int(input(f"Jogador {players[current_player]} escolha uma posição (1-9): ")) - 1
            if move < 0 or move >= 9 or board[move] != " ":
                raise ValueError("Movimento inválido!")
        except ValueError as e:
            print(e)
            input("Pressione Enter para tentar novamente...")
            continue

        # Atualizar o tabuleiro
        board[move] = players[current_player]

        # 🔥 Mostrar previsão após jogada
        state = predict_game_state(board)
        print(f"\n🤖 Após a jogada: {traduzir_saida(state)}\n")
        input("Pressione Enter para continuar...")

        # Verificar vitória
        if check_winner(board, players[current_player]):
            clear_screen()
            print_board(board)
            print(f"\n🏆 Jogador {players[current_player]} venceu!\n")
            break

        # Verificar empate
        if is_draw(board):
            clear_screen()
            print_board(board)
            print("\n🤝 Empate!\n")
            break

        # Trocar jogador
        current_player = 1 - current_player

if __name__ == "__main__":
    tic_tac_toe()