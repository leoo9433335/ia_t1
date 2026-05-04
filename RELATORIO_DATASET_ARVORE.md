# Dataset e Árvore de Decisão — Material para o relatório

Documento curto com o que foi feito no dataset e como a árvore de decisão funciona. Pode ser usado direto como base pros slides do PPT.

---

## Part 1 - O que mudei

- Os nomes de variáveis e arquivos contendo o sufixo 32 foram removidos (incluindo o nome das colunas pos_32_*, ajustadas para pos_*), uma vez que esse identificador não consta no enunciado oficial do trabalho. A alteração foi propagada também aos demais arquivos que liam ou geravam esses nomes (como o builder de dataset e os CSVs treino/validação/teste), mantendo a consistência do projeto.                                                                                                                

- A função estado_real(board) foi adicionada para calcular a classe verdadeira do tabuleiro a partir das regras do jogo, servindo como referência para validar a predição do modelo a cada jogada.                                  

- A cada turno, a previsão da IA é comparada com o estado real e contabilizada como acerto ou erro. O score corrente é exibido durante a partida e o score final ao término, em formato de fração e percentual.                                                                                                                                         

- O fim do jogo passou a ser controlado pelo estado real do tabuleiro, conforme a regra do enunciado: a partida é encerrada quando o jogo de fato termina, mesmo que a IA não detecte; e segue normalmente quando a IA aponta um fim incorretamente.

- Um dos jogadores foi convertido em máquina com jogadas aleatórias (humano joga como X, máquina como O), implementando a interação humano vs. máquina exigida pelo enunciado. A escolha de cada jogada da máquina é feita por amostragem uniforme entre as casas livres do tabuleiro.                                                        

- A leitura do input do humano foi extraída para a função jogada_humano, e a jogada aleatória para jogada_maquina, mantendo o loop principal mais legível.                  

---

## Parte 2 — Árvore de Decisão

### Como funciona (em uma frase)

A árvore aprende uma sequência de perguntas binárias sobre as posições do tabuleiro (ex: "a casa do meio é X?") até chegar a uma classe. Cada nó interno é uma pergunta, cada folha é uma classe predita.

### Como ela é construída

1. **Pega todas as amostras de treino** e mede quão "misturadas" estão as classes (impureza).
2. **Testa todas as features e todos os pontos de corte** pra achar a divisão que mais reduz a impureza.
3. **Divide as amostras** em duas partes (que satisfazem ou não a pergunta).
4. **Repete recursivamente** em cada parte até parar (limite de profundidade ou poucos exemplos restantes).

A "impureza" é medida com **Gini** ou **entropia** (são duas fórmulas diferentes que medem a mesma coisa: quão homogênea é a distribuição de classes em um nó).

### Por que árvore de decisão é boa pra esse problema

- **Interpretabilidade:** dá pra olhar a árvore e entender as regras (ex: "se a diagonal principal é toda X, então x_venceu").
- **Não precisa normalizar** as features (X=1, O=-1, vazio=0 já funciona direto).
- **Lida bem com features categóricas** disfarçadas de numéricas, que é o nosso caso.

### Hiperparâmetros tunados


| Parâmetro           | Valores testados      | O que controla                                                |
| ------------------- | --------------------- | ------------------------------------------------------------- |
| `criterion`         | gini, entropy         | Como medir a impureza                                         |
| `max_depth`         | None, 4, 6, 8, 10, 15 | Profundidade máxima da árvore (controla overfitting)          |
| `min_samples_split` | 2, 5, 10              | Mínimo de amostras para dividir um nó                         |
| `min_samples_leaf`  | 1, 2, 5               | Mínimo de amostras numa folha                                 |
| `class_weight`      | None, balanced        | `balanced` dá mais peso à classe `empate` (que é minoritária) |


### Como o tuning foi feito

`GridSearchCV` com **validação cruzada de 5 folds** sobre treino+validação. A métrica de seleção foi **F1 macro**, não acurácia, porque a classe `empate` está desbalanceada (32 vs 200) e a acurácia mascararia o problema.

### Configuração escolhida

```
criterion = gini
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
class_weight = None
```

Ou seja, a árvore com defaults venceu o tuning. Isso significa que o problema é simples o suficiente para a árvore não precisar de regularização adicional.

### Resultados no conjunto de teste


| Métrica           | Valor      |
| ----------------- | ---------- |
| Acurácia          | **0.7474** |
| Precision (macro) | 0.7496     |
| Recall (macro)    | 0.7167     |
| F1 (macro)        | **0.7306** |


Por classe:


| Classe   | Precision | Recall | F1   | Suporte |
| -------- | --------- | ------ | ---- | ------- |
| tem_jogo | 0.66      | 0.70   | 0.68 | 30      |
| x_venceu | 0.79      | 0.73   | 0.76 | 30      |
| o_venceu | 0.81      | 0.83   | 0.82 | 30      |
| empate   | 0.75      | 0.60   | 0.67 | 5       |


### Análise rápida

- A árvore acerta melhor `o_venceu` (F1 0.82) e tem mais dificuldade com `tem_jogo` (F1 0.68), que é a classe mais "difusa" (qualquer estado em andamento).
- `empate` tem o menor recall (0.60), mas com só 5 amostras no teste o número não é estatisticamente confiável.
- Comparando com MLP (acurácia 83.16%), a árvore fica abaixo. Faz sentido: redes neurais capturam padrões não-lineares melhor. Mas a árvore é mais interpretável.

### Para o PPT, vale incluir

- A configuração escolhida (tabela acima).
- A matriz de confusão (gerada na célula 5 do notebook).
- A visualização da árvore (gerada na célula 6).
- A importância das features (gerada na célula 7) — mostra que o centro do tabuleiro (`pos_4`) costuma ser a feature mais relevante.

---

## Como rodar

```bash
# 1. (Opcional) Regerar o dataset
python main.py

# 2. Abrir o notebook da árvore
jupyter notebook arvore_decisao.ipynb
# (ou abrir no VS Code/Cursor)
```

