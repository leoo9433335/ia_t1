import os
import random

import joblib
import pandas as pd

mlp_model = joblib.load("mlp_model.pkl")

HUMANO = "X"
MAQUINA = "O"

CLASSES = {0: "Tem jogo", 1: "X venceu", 2: "O venceu", 3: "Empate"}


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_board(board):
    print()
    for i in range(3):
        print(" | ".join(board[i * 3:(i + 1) * 3]))
        if i < 2:
            print("---+---+---")


def check_winner(board, player):
    linhas = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6],
    ]
    for cond in linhas:
        if all(board[p] == player for p in cond):
            return True
    return False


def is_draw(board):
    return all(c != " " for c in board)


def estado_real(board):
    if check_winner(board, "X"):
        return 1
    if check_winner(board, "O"):
        return 2
    if is_draw(board):
        return 3
    return 0


def predict_game_state(board):
    numeric = [1 if c == "X" else -1 if c == "O" else 0 for c in board]
    df = pd.DataFrame([numeric], columns=[f"pos_{i}" for i in range(9)])
    return int(mlp_model.predict(df)[0])


def jogada_humano(board, player):
    while True:
        try:
            move = int(input(f"\nVocê ({player}), escolha uma posição (1-9): ")) - 1
            if move < 0 or move >= 9 or board[move] != " ":
                raise ValueError("Movimento inválido!")
            return move
        except ValueError as e:
            print(e)


def jogada_maquina(board):
    livres = [i for i, c in enumerate(board) if c == " "]
    return random.choice(livres)


def tic_tac_toe():
    board = [" "] * 9
    players = [HUMANO, MAQUINA]
    atual = 0

    acertos = 0
    erros = 0

    while True:
        clear_screen()
        print_board(board)
        total = acertos + erros
        if total:
            print(f"\nScore IA: {acertos}/{total} acertos ({acertos / total:.0%})")

        jogador = players[atual]
        if jogador == HUMANO:
            move = jogada_humano(board, jogador)
        else:
            move = jogada_maquina(board)
            print(f"\nMáquina ({jogador}) jogou na posição {move + 1}.")

        board[move] = jogador

        predito = predict_game_state(board)
        real = estado_real(board)
        acerto = predito == real
        if acerto:
            acertos += 1
        else:
            erros += 1

        clear_screen()
        print_board(board)
        print(f"\nIA prevê:    {CLASSES.get(predito, predito)}")
        print(f"Estado real: {CLASSES[real]}  [{'ACERTO' if acerto else 'ERRO'}]")

        # Encerra com base no estado REAL (regra do enunciado): se o jogo
        # acabou de fato, encerra mesmo que a IA não tenha detectado;
        # se a IA disse que acabou mas não acabou, continua jogando.
        if real != 0:
            total = acertos + erros
            acc = acertos / total if total else 0
            print(f"\n{CLASSES[real]}!")
            print(f"Score final da IA: {acertos}/{total} acertos ({acc:.2%})")
            input("\nPressione Enter para sair...")
            break

        input("\nPressione Enter para continuar...")
        atual = 1 - atual


if __name__ == "__main__":
    tic_tac_toe()
