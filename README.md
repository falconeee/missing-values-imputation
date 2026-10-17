# Imputação de valores faltantes para o MSCVAE

Experimentos de imputação de valores faltantes e de outliers de sensor **antes** da predição de um modelo de detecção de anomalias multivariado (MSCVAE). Os testes usam dados de dois processos industriais reais, **descaracterizados** como *Processo A* e *Processo B*.

A pergunta é: quando uma TAG fica sem dado (NaN) ou um sensor falha com um valor absurdo, qual estratégia de imputação reconstrói melhor o sinal sem gerar alarmes falsos e sem esconder uma falha de processo de verdade?

## Estrutura

```
.
├── main_processo_A_MSCVAE.ipynb   # experimento no Processo A
├── main_processo_B_MSCVAE.ipynb   # experimento no Processo B
├── src/
│   ├── ppd.py          # carga do dataset, TAGs do subsistema, remoção de desligado/transitório
│   ├── MSCVAE.py       # modelo MSCVAE (ConvLSTM + atenção + VAE) e limiar por SPOT/POT
│   ├── imputation.py   # experimento de imputação: treino, outliers, lacunas, métricas e gráficos
│   ├── model.py        # train_imputer: treino do IterativeImputer no formato de deploy
│   └── utils.py        # gráfico do índice de anomalia (graph_predict)
├── data/               # datasets (não versionados)
├── models/             # .pkl dos imputers gerados (não versionados)
├── pyproject.toml      # dependências (uv)
└── uv.lock
```

## Instalação

Python 3.12. Com [uv](https://docs.astral.sh/uv/):

```bash
uv sync                 # cria o .venv a partir do pyproject.toml + uv.lock
source .venv/bin/activate

python -m ipykernel install --user --name missing-values-imputation --display-name "Python (missing-values-imputation)"
```

O torch vem do PyPI: no macOS usa MPS e, no Linux, o wheel padrão já inclui CUDA (precisa de um driver NVIDIA compatível). Para fixar outro build (só CPU ou outra versão do CUDA), configure o índice do PyTorch com `[[tool.uv.index]]` + `[tool.uv.sources]` no `pyproject.toml`.

Abra os notebooks **a partir da raiz do repositório**, porque eles importam `from src import ...` e leem `data/`. O MSCVAE usa GPU (CUDA ou MPS) se houver. Os imputers rodam na CPU.

## Dados

Os dados não são versionados (`data/` está no `.gitignore`). Cada processo precisa de dois arquivos em `data/`:

| Arquivo | Conteúdo |
|---|---|
| `processo_X.parquet` | Uma coluna `Timestamp` (amostragem de 1 min) e uma coluna numérica por TAG (`TAG_001`, `TAG_002`, …). |
| `processo_X_subsistema.csv` | Separador `;`, colunas `VARIAVEL;DESC;SISTEMA`: o nome da TAG, a descrição e o subsistema. |

As TAGs, as descrições e os subsistemas foram anonimizados. Os valores e os timestamps são os originais.

## Fluxo do notebook

Os dois notebooks têm a mesma estrutura. Só mudam os parâmetros do processo.

1. **Carga dos dados**: `ppd.load_dataset_principal` descarta colunas, converte para numérico e ordena por tempo. `ppd.set_tags_config` cruza o dataset com o subsistema.
2. **ON/OFF**: `ppd.drop_transitorio_desligado` remove os períodos de equipamento desligado, com uma margem antes e depois (transitório). A referência é uma TAG (`pp_var_ref_desligado`) e um limite.
3. **Períodos de treino**: um ou mais intervalos de operação normal. Repetir um período em `df_train_list` aumenta o peso dele no treino.
4. **MSCVAE**: `MSCVAE().fit(df_train_list, gain, epochs)`. O limiar vem do SPOT (Peaks-Over-Threshold) e é multiplicado por `gain`.
5. **Treino dos imputers**: usa os mesmos períodos do MSCVAE, sem duplicatas. Antes, `fimp.limpar_falhas_treino` tira as linhas com falha grosseira de sensor. `fimp.treinar_imputers` treina um `IterativeImputer` por estimador com o `train_imputer` de deploy, que inclui um holdout temporal com lacunas de 10–240 min, uma TAG por vez.
6. **Outliers extremos no teste**: `fimp.detectar_outliers` marca as falhas reais de sensor no período de teste, seja valor absurdo, seja fundo de escala. Para conferência, `fimp.listar_desvios` lista os desvios de processo que **não** são tratados.
7. **Lacunas sintéticas**: `fimp.injetar_lacunas` cria lacunas NaN numa cópia do teste, longe dos outliers reais. Como o valor real é conhecido, dá para medir o erro.
8. **Imputação**: `fimp.aplicar_imputacao` aplica a regra de produção (abaixo) com cada método.
9. **Métricas**: erro nas lacunas sintéticas (`fimp.metricas_lacunas`) e efeito no índice de anomalia (`fimp.resumo_efeito_predicao`), com gráficos antes × depois por TAG.

## Métodos comparados

| Método | Descrição |
|---|---|
| `ffill (baseline)` | `ffill`/`bfill` em todas as lacunas. |
| `BayesianRidge` | `IterativeImputer` linear. |
| `HistGB (produção)` | `IterativeImputer` com `HistGradientBoostingRegressor` (estimador padrão do deploy). |
| `ExtraTrees` | `IterativeImputer` com `ExtraTreesRegressor` raso, para manter o `.pkl` pequeno. |
| `XGBoost` | `IterativeImputer` com `XGBRegressor`. |
| `LightGBM` | `IterativeImputer` com `LGBMRegressor`. |

Os cenários de predição também incluem a **Referência** (dados originais, sem lacunas sintéticas) e **Sem imputação (dropna)**, que descarta as linhas com NaN.

Para testar outro estimador, edite o dict `estimadores` no notebook.

## Regras principais

**Outlier extremo (falha de sensor).** Só é imputado o que é falha de sensor. Os desvios de processo, mesmo grandes, são justamente o que o MSCVAE precisa detectar e vão crus para ele. Há duas regras.

*Valor absurdo.* O valor é marcado quando cumpre **as duas** condições:

1. `|x − mediana| > FATOR_EXTREMO × max(faixa p1–p99, |mediana|)` do treino. É um valor absurdo para o nível da TAG.
2. `|x[t] − x[t−1]| > SALTO_EXTREMO × p99,9` dos saltos do treino. O valor foi alcançado num salto que o sensor nunca deu.

Depois do disparo, a marcação segue para frente até o valor voltar à faixa normal (± `K_RETORNO` faixas). Um desvio de processo evolui aos poucos e não cumpre a condição 2. Essa regra pega o sensor que vai para ~6e6 ou para 10× a faixa.

*Fundo de escala.* Vale só para as TAGs configuradas em `LIMITES_ESCALA = {tag: (limite_inferior, limite_superior)}`, com `None` no lado sem limite. Cobre a falha que fica perto da faixa operacional, onde a regra anterior não chega: um sensor a ~1000 que vai para −58 tem fator ~1. O valor é marcado quando cumpre **as duas** condições:

1. Está na zona do limite: a menos de `TOL_ESCALA` da distância entre a faixa normal (p1/p99) e o limite.
2. Chegou lá de forma abrupta: em até 2 amostras, um salto > `SALTO_ESCALA` × p99,9 dos saltos do treino. O processo que sobe aos poucos até saturar o instrumento é desvio real e fica cru.

Depois do disparo, a marcação segue enquanto o valor estiver mais perto do limite do que da faixa normal. Isso inclui a rampa interpolada que o histórico grava depois da falha, mas não a parada de equipamento logo depois de um pico no limite.

O ideal é preencher `LIMITES_ESCALA` com o range real dos instrumentos. `fimp.candidatos_fundo_escala` lista os valores em que cada sensor parou depois de saídas abruptas repetidas, como ponto de partida, mas também traz falsos candidatos (TAGs inteiras, paradas).

Nas duas regras, o que vem antes do disparo nunca é marcado, como no runtime, que só vê o passado. Por isso a primeira amostra da transição (a média do minuto em que o sensor travou) fica crua.

**Regra de aplicação (igual à produção):**

1. outlier vira NaN;
2. lacuna menor que `START_MINUTES` com valor anterior válido é preenchida com `ffill`;
3. o restante vai para o `IterativeImputer`;
4. linha com ≥ `MAX_BAD_RATIO` das variáveis inválidas não é reconstruída.

## Parâmetros

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `FATOR_FALHA_GROSSEIRA` | 50 | Limpeza do treino do imputer: \|x − mediana\| > fator × escala robusta. |
| `FATOR_EXTREMO` | 3.0 | Condição 1 do outlier extremo. |
| `SALTO_EXTREMO` | 20.0 | Condição 2 do outlier extremo. |
| `K_RETORNO` | 1.0 | Faixa de retorno que encerra a marcação do valor absurdo. |
| `LIMITES_ESCALA` | por processo | Limites do instrumento por TAG para a regra de fundo de escala. |
| `TOL_ESCALA` | 0.05 | Zona do limite: fração da distância entre a faixa normal e o limite. |
| `SALTO_ESCALA` | 3.0 | Salto mínimo (× p99,9), em até 2 amostras, para disparar o fundo de escala. |
| `START_MINUTES` | 10 | Lacunas mais curtas que isso (com âncora) vão para `ffill`. |
| `MAX_BAD_RATIO` | 0.7 | Fração de variáveis inválidas a partir da qual a linha não é reconstruída. |
| `LACUNAS_POR_TAG` | 3 | Lacunas sintéticas por TAG. |
| `DUR_MIN`, `DUR_MAX` | 10, 240 | Duração das lacunas sintéticas (min). |
| `FRAC_SIMULTANEAS` | 0.3 | Fração das lacunas que derruba 2–3 TAGs juntas. |
| `PASSO_AMOSTRAGEM` | 1 | Subamostragem do treino dos imputers (> 1 acelera). |

## Principais resultados

Os números abaixo são da execução anterior à regra de fundo de escala. Com ela, o Processo A passa a imputar 83 pontos de falha de sensor no teste (antes, nenhum), e o Processo B não muda. O MAE e o RMSE são normalizados pelo desvio-padrão da TAG no treino, para comparar TAGs de escalas diferentes. Os valores são medianas entre as TAGs.

### Lacunas sintéticas no teste

| Método | Proc. A: MAE norm. | Proc. A: R² | Proc. B: MAE norm. | Proc. B: R² |
|---|---|---|---|---|
| XGBoost | **0,184** | **0,763** | 0,244 | 0,823 |
| LightGBM | 0,190 | 0,728 | **0,221** | 0,814 |
| HistGB (produção) | 0,197 | 0,693 | 0,253 | **0,835** |
| ExtraTrees | 0,267 | 0,606 | 0,260 | 0,798 |
| BayesianRidge | 0,298 | 0,520 | 0,240 | 0,824 |
| ffill (baseline) | 0,584 | −0,126 | 0,284 | 0,683 |

### Efeito na predição do MSCVAE

| Cenário | Proc. A: cobertura nas lacunas | Proc. A: \|Δ phi/limiar\| | Proc. B: cobertura nas lacunas | Proc. B: % alarmes nos outliers reais |
|---|---|---|---|---|
| Referência | 100% | — | 100% | 36,0% |
| Sem imputação (dropna) | 5,7% | 4,95 | 6,3% | 36,1% |
| ffill (baseline) | 100% | 0,18 | 100% | 8,0% |
| IterativeImputer (5 métodos) | 100% | 0,08–0,09 | 100% | 7,9–8,0% |

### Conclusões

- **Sem imputação, o monitoramento se perde.** O `dropna` mantém só ~6% dos instantes dentro das lacunas.
- **O IterativeImputer ganha do ffill.** No Processo A, o MAE normalizado cai de 0,58 para 0,18–0,20 com os métodos de árvore, e o desvio do índice de anomalia em relação à referência cai pela metade (0,18 → ~0,08). No Processo B a diferença é menor (MAE de 0,28 para 0,22–0,26).
- **Imputar o outlier de sensor elimina alarmes falsos.** No Processo B, os alarmes nos trechos com falha de sensor caem de 36% para ~8% com qualquer método.
- **Entre os métodos de árvore, a diferença é pequena.** O `HistGB (produção)` fica perto dos melhores e tem o `.pkl` menor entre as árvores (1,7–3,7 MB, contra 6–8 MB de XGBoost/LightGBM). Isso importa no deploy, porque o runtime baixa o imputer a cada ciclo.
- **O holdout do treino mostra TAGs pouco previsíveis** (cerca de 20 por processo com R² < 0,5). Para essas TAGs, a imputação deve ser usada com cautela.

## Adicionar um novo processo

1. Coloque `processo_X.parquet` e `processo_X_subsistema.csv` em `data/`.
2. Copie um dos notebooks e ajuste:
   - `process_name`;
   - as colunas descartadas na carga;
   - os parâmetros de ON/OFF (`pp_var_ref_desligado`, limite e cortes);
   - os limites de escala dos instrumentos (`LIMITES_ESCALA`);
   - os períodos de treino;
   - o `gain` do MSCVAE.
3. Rode o notebook inteiro. Os imputers são salvos em `models/`.
