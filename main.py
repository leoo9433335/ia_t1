import random
from itertools import combinations

import pandas as pd

SEED = 42
N_PER_CLASSE = 200

LINHAS_VITORIA = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]

CLASSES = {0: "tem_jogo", 1: "x_venceu", 2: "o_venceu", 3: "empate"}

COLUNAS_FEATURES = [f"pos_{i}" for i in range(9)]
COLUNAS = COLUNAS_FEATURES + ["class"]


def vencedor(b):
    for i, j, k in LINHAS_VITORIA:
        s = b[i] + b[j] + b[k]
        if s == 3:
            return 1
        if s == -3:
            return 2
    return 0


def rotular(b):
    v = vencedor(b)
    if v:
        return v
    return 0 if 0 in b else 3


def carregar_uci(path="tic-tac-toe.data"):
    df = pd.read_csv(path, header=None)
    m = {"x": 1, "o": -1, "b": 0}
    return df.iloc[:, :9].apply(lambda c: c.map(m)).values.tolist()


def simular_partida(rng):
    # Joga uma partida válida X-O-X-... escolhendo casas aleatórias
    # e devolve os estados intermediários (sem vencedor, com casa vazia).
    b = [0] * 9
    casas = list(range(9))
    rng.shuffle(casas)
    estados = []
    turno = 1
    for c in casas:
        b[c] = turno
        if vencedor(b):
            break
        if 0 in b:
            estados.append(tuple(b))
        turno = -turno
    return estados


def gerar_tem_jogo(n, seed=SEED):
    rng = random.Random(seed)
    pool = set()
    while len(pool) < n * 5 and len(pool) < 50_000:
        for s in simular_partida(rng):
            pool.add(s)
    out = [list(t) for t in pool]
    rng.shuffle(out)
    return out[:n]


def gerar_empates():
    # Enumera tabuleiros cheios sem três em linha. C(9,5)=126 disposições para
    # X; só 16 não têm vencedor. O caso 4X+5O é matematicamente equivalente
    # (jogo onde O começa) e dá outras 16 disposições válidas. Total base: 32.
    out = []
    for n_x in (4, 5):
        for x_pos in combinations(range(9), n_x):
            b = [-1] * 9
            for p in x_pos:
                b[p] = 1
            if not vencedor(b):
                out.append(b)
    return out


def dedup(lista):
    return [list(t) for t in {tuple(b) for b in lista}]


def split_estratificado(df, frac_treino=0.70, frac_val=0.15, seed=SEED):
    treino, val, teste = [], [], []
    for c in [0, 1, 2, 3]:
        sub = df[df["class"] == c].sample(frac=1, random_state=seed + 1).reset_index(drop=True)
        n = len(sub)
        n_t = round(n * frac_treino)
        n_v = round(n * frac_val)
        treino.append(sub.iloc[:n_t])
        val.append(sub.iloc[n_t:n_t + n_v])
        teste.append(sub.iloc[n_t + n_v:])
    treino = pd.concat(treino).sample(frac=1, random_state=seed + 2).reset_index(drop=True)
    val = pd.concat(val).sample(frac=1, random_state=seed + 3).reset_index(drop=True)
    teste = pd.concat(teste).sample(frac=1, random_state=seed + 4).reset_index(drop=True)
    return treino, val, teste


def montar_dataset():
    uci = carregar_uci()
    print(f"UCI: {len(uci)} linhas")

    pools = {0: [], 1: [], 2: [], 3: []}
    for b in uci:
        pools[rotular(b)].append(b)
    print("Após reclassificar UCI:")
    for c, lst in pools.items():
        print(f"  {c} {CLASSES[c]}: {len(lst)}")

    # UCI só tem fim de jogo, então tem_jogo precisa ser gerado.
    # Geramos por simulação de partidas válidas para garantir que os
    # estados sejam alcançáveis num jogo real (X joga primeiro e alterna).
    pools[0].extend(gerar_tem_jogo(N_PER_CLASSE * 2))

    # Empates do UCI são poucos. Enumera todos os 5X+4O sem 3-em-linha.
    pools[3].extend(gerar_empates())

    for c in pools:
        pools[c] = dedup(pools[c])

    # Por garantia, evita que o mesmo tabuleiro caia em duas classes.
    visto = {}
    for c in [1, 2, 3, 0]:
        for b in pools[c]:
            visto.setdefault(tuple(b), c)
    pools = {0: [], 1: [], 2: [], 3: []}
    for tup, c in visto.items():
        pools[c].append(list(tup))

    print("Pool após dedup global:")
    for c, lst in pools.items():
        print(f"  {c} {CLASSES[c]}: {len(lst)}")

    rng = random.Random(SEED)
    linhas = []
    for c in [0, 1, 2, 3]:
        pool = pools[c][:]
        rng.shuffle(pool)
        n = min(N_PER_CLASSE, len(pool))
        if n < N_PER_CLASSE:
            print(f"  AVISO: classe {c} ({CLASSES[c]}) com {n} amostras")
        for b in pool[:n]:
            linhas.append(b + [c])

    return pd.DataFrame(linhas, columns=COLUNAS)


if __name__ == "__main__":
    df = montar_dataset()
    df.to_csv("tic-tac-toe_balanced.csv", index=False)
    print(f"\nDataset balanceado: {len(df)} amostras")
    print(df["class"].value_counts().sort_index().to_dict())

    treino, val, teste = split_estratificado(df)
    treino.to_csv("train.csv", index=False)
    val.to_csv("validation.csv", index=False)
    teste.to_csv("test.csv", index=False)

    for nome, d in [("treino", treino), ("validação", val), ("teste", teste)]:
        print(f"  {nome}: {len(d)}  {d['class'].value_counts().sort_index().to_dict()}")

    erros = sum(
        rotular(row[COLUNAS_FEATURES].tolist()) != row["class"]
        for _, row in df.iterrows()
    )
    print(f"\nInconsistências rótulo vs regra: {erros}")
