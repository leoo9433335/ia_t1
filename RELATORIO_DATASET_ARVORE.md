# Dataset e Árvore de Decisão — Material para o relatório

Documento curto com o que foi feito no dataset e como a árvore de decisão funciona. Pode ser usado direto como base pros slides do PPT.

---

## Parte 1 — O dataset

### Origem

Dataset UCI **Tic-Tac-Toe Endgame** (https://archive.ics.uci.edu/dataset/101).
- 958 instâncias.
- Cada instância é um tabuleiro 3×3.
- Original tem só 2 rótulos: `positive` (X venceu) e `negative` (X não venceu).

### Problemas encontrados no dataset original

1. **Só tem 2 classes**, mas o trabalho pede 4 (Tem jogo, X venceu, O venceu, Empate).
   `negative` mistura "O venceu" e "Empate" no mesmo rótulo.
2. **Só tem estados de fim de jogo** (tabuleiros com 5 X + 4 O).
   Não existe nenhuma instância de "Tem jogo" (jogo em andamento).
3. **Distribuição enviesada** (~65% positive, 35% negative).

### O que foi feito (passo a passo)

Tudo está no `main.py`. Resumo do pipeline:

| Passo | O que faz | Resultado |
|---|---|---|
| 1 | Lê o UCI bruto e converte X/O/vazio para 1/-1/0 | 958 linhas com encoding numérico |
| 2 | Re-rotula cada linha lendo o tabuleiro (regra do jogo) | Separa `o_venceu` de `empate` corretamente |
| 3 | Gera estados de `tem_jogo` simulando partidas válidas | 400 estados intermediários alcançáveis |
| 4 | Enumera todos os empates possíveis (5X+4O e 4X+5O sem 3-em-linha) | 32 empates únicos |
| 5 | Remove duplicatas dentro e entre classes | Pool limpo |
| 6 | Balanceia em ~200 amostras por classe | 200/200/200/32 |
| 7 | Divide fisicamente em treino/validação/teste (70/15/15 estratificado) | Splits prontos |

### Detalhes importantes

**Como a classe `tem_jogo` foi gerada:** simulação de partidas válidas. Começa com tabuleiro vazio, joga X, depois O, depois X... a cada jogada salva o estado atual se ainda não há vencedor e há casa vazia. Isso garante que todo estado seja **alcançável num jogo real**, não um tabuleiro inventado (ex: 7 X's e 0 O's, que nunca acontece).

**Por que só 32 empates?** Existem `C(9,5) = 126` formas de distribuir 5 X's no tabuleiro. Dessas, **só 16** não formam três em linha (de X ou de O). Somando o caso simétrico 4X+5O, total = **32 empates únicos no universo**. Não é falha do dataset — é uma propriedade matemática do jogo da velha 3×3. Por isso o enunciado deixou aberto: "200 amostras de cada classe, **quando possível**".

### Resultado final

- **Total: 632 amostras**
- 200 tem_jogo, 200 x_venceu, 200 o_venceu, 32 empate
- Split: 442 treino / 95 validação / 95 teste (estratificado)
- 0 inconsistências (todo rótulo bate com a regra do jogo)

Arquivos gerados:
- `tic-tac-toe_balanced.csv` — dataset completo
- `train.csv`, `validation.csv`, `test.csv` — splits físicos

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

| Parâmetro | Valores testados | O que controla |
|---|---|---|
| `criterion` | gini, entropy | Como medir a impureza |
| `max_depth` | None, 4, 6, 8, 10, 15 | Profundidade máxima da árvore (controla overfitting) |
| `min_samples_split` | 2, 5, 10 | Mínimo de amostras para dividir um nó |
| `min_samples_leaf` | 1, 2, 5 | Mínimo de amostras numa folha |
| `class_weight` | None, balanced | `balanced` dá mais peso à classe `empate` (que é minoritária) |

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

| Métrica | Valor |
|---|---|
| Acurácia | **0.7474** |
| Precision (macro) | 0.7496 |
| Recall (macro) | 0.7167 |
| F1 (macro) | **0.7306** |

Por classe:

| Classe | Precision | Recall | F1 | Suporte |
|---|---|---|---|---|
| tem_jogo | 0.66 | 0.70 | 0.68 | 30 |
| x_venceu | 0.79 | 0.73 | 0.76 | 30 |
| o_venceu | 0.81 | 0.83 | 0.82 | 30 |
| empate | 0.75 | 0.60 | 0.67 | 5 |

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
