"""Experimento de imputação de valores faltantes antes da predição do MSCVAE.

Treina um IterativeImputer por estimador com o mesmo `train_imputer` do deploy
(src/model.py) e aplica a mesma regra do preprocessing de produção:

- outlier extremo (falha de sensor) vira NaN;
- lacuna curta (< start_minutes) com valor anterior válido -> ffill;
- lacuna a partir daí -> imputer;
- linha com >= max_bad_ratio das variáveis inválidas não é reconstruída.

Diferenças para a produção:
- a regra padrão de outlier é a de falha de sensor (`detectar_outliers`, metodo="extremo"):
  só marca valor fisicamente absurdo para o sensor ou sensor em fundo de escala
  (`limites_escala`). O desvio robusto (MAD) da produção e o
  envelope da faixa de treino marcam também desvio de processo (equipamento parando, setpoint
  mudando), e imputar isso mascara justamente o início de uma falha. As duas regras
  continuam disponíveis para comparação (metodo="robust_z" / "envelope");
- antes de treinar, as linhas com falha grosseira de sensor saem do treino do imputer
  (`limpar_falhas_treino`): em alguns processos os períodos de treino do MSCVAE contêm
  ~10% de linhas com sensor em 6e6.
"""
import contextlib
import io
import os
import re
import time
import unicodedata

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_MAD_TO_SIGMA = 1.4826
_IQR_TO_SIGMA = 1.349

COR_SERIE = "#0F293A"
COR_LIMIAR = "#FB8102"
COR_GRID = "#CECFD1"
COR_NAN = "#9E9E9E"
COR_OUTLIER = "#D62828"
CORES_METODOS = {
    "Referência": "#0F293A",
    "Sem imputação (dropna)": "#BDBDBD",
    "ffill (baseline)": "#8D99AE",
    "BayesianRidge": "#2E86AB",
    "HistGB (produção)": "#FB8102",
    "ExtraTrees": "#6A994E",
    "XGBoost": "#C1121F",
    "LightGBM": "#7B2CBF",
}
_CORES_EXTRA = ["#E9C46A", "#264653", "#F4A261", "#2A9D8F", "#8338EC"]


def _cor(nome, i=0):
    return CORES_METODOS.get(nome, _CORES_EXTRA[i % len(_CORES_EXTRA)])


def _slug(texto):
    texto = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "-", texto).strip("-")


def _layout(fig, titulo, altura=450):
    fig.update_layout(
        title=titulo,
        height=altura,
        # Fundo branco explícito: com fundo transparente o gráfico some no tema escuro do Jupyter/VS Code.
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=80, b=40),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=COR_GRID)
    return fig


def _corridas(flag):
    """Corridas contíguas de True como array [[inicio, fim_exclusivo], ...]."""
    f = np.asarray(flag, dtype=bool)
    if not f.any():
        return np.empty((0, 2), dtype=int)
    d = np.diff(np.r_[0, f.astype(np.int8), 0])
    return np.c_[np.flatnonzero(d == 1), np.flatnonzero(d == -1)]


def _passo_segundos(ts):
    passo = pd.Series(pd.to_datetime(ts)).diff().dt.total_seconds().median()
    return float(passo) if passo and passo > 0 else 60.0


# ---------------------------------------------------------------- estimadores e treino


def estimadores_padrao(random_state=0, n_jobs=None):
    """Estimadores testados no IterativeImputer. Edite o dict no notebook para incluir outros.

    Todos rodam na CPU: com o IterativeImputer o XGBoost em CUDA ficou mais lento (um fit
    curto por TAG e por iteração, e o predict com numpy volta para a CPU) e o LightGBM em
    GPU (OpenCL) também. `n_jobs` vale para o treino e fica gravado no .pkl: no runtime da
    AWS ajuste para o número de vCPUs, senão as threads disputam CPU.
    """
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import BayesianRidge
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 8) - 4)
    return {
        "BayesianRidge": BayesianRidge(),
        # Mesmo estimador padrão do train_imputer (produção).
        "HistGB (produção)": HistGradientBoostingRegressor(
            max_iter=50, max_leaf_nodes=15, random_state=random_state
        ),
        # Raso de propósito: o imputer guarda um estimador por TAG e por iteração (~220) e,
        # com max_depth=12, o .pkl passava de 600 MB. Assim fica no porte do LightGBM.
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=20, max_depth=8, min_samples_leaf=10, n_jobs=n_jobs, random_state=random_state
        ),
        "XGBoost": XGBRegressor(
            n_estimators=150, max_depth=6, learning_rate=0.1, subsample=0.8,
            tree_method="hist", n_jobs=n_jobs, random_state=random_state,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=150, num_leaves=31, learning_rate=0.1, subsample=0.8, subsample_freq=1,
            n_jobs=n_jobs, random_state=random_state, verbose=-1,
        ),
    }


def treinar_imputers(df_train_imputer, timestamp, process_name, estimadores, passo_amostragem=1, verbose=False):
    """Roda o `train_imputer` de produção uma vez por estimador.

    Cada `.pkl` é renomeado para `<processo>_<metodo>_<nome>.pkl` em DIR_MODELS: o nome do
    train_imputer tem resolução de segundo e dois treinos rápidos se sobrescreveriam.
    Retorna (imputers, blocos, tempos); `blocos[metodo]` segue o formato do bloco `imputer` do deploy.
    """
    import joblib
    from . import model

    df = df_train_imputer.iloc[::passo_amostragem] if passo_amostragem > 1 else df_train_imputer
    colunas = [c for c in df.columns if c != timestamp]
    imputers, blocos, tempos = {}, {}, {}
    for nome, estimador in estimadores.items():
        inicio = time.time()
        saida = io.StringIO()
        with contextlib.redirect_stdout(saida) if not verbose else contextlib.nullcontext():
            bloco = model.train_imputer(df, timestamp, process_name, estimator=estimador)
        tempos[nome] = time.time() - inicio
        if bloco["columns"] != colunas:
            raise RuntimeError("train_imputer devolveu colunas fora da ordem do df_train_imputer")

        nome_arquivo = "%s_%s_%s" % (_slug(process_name), _slug(nome), bloco["name_model"])
        os.replace(model.DIR_MODELS + bloco["name_model"], model.DIR_MODELS + nome_arquivo)
        bloco["name_model"] = nome_arquivo
        imputers[nome] = joblib.load(model.DIR_MODELS + nome_arquivo)
        blocos[nome] = bloco
        print("%-20s treinado em %6.1fs -> %s" % (nome, tempos[nome], nome_arquivo))
    return imputers, blocos, tempos


def resumo_holdout(blocos, std_treino, tempos=None):
    """Tabela por método com as métricas de holdout do train_imputer + matrizes TAG x método.

    O tamanho do .pkl entra porque o runtime baixa o imputer a cada ciclo.
    """
    from . import model

    std = std_treino.replace(0, np.nan)
    linhas, r2_tag, mae_tag = [], {}, {}
    for nome, bloco in blocos.items():
        mae = pd.Series(bloco["holdout_mae"], dtype=float)
        r2 = pd.Series(bloco["holdout_r2"], dtype=float)
        mae_norm = mae / std.reindex(mae.index)
        r2_tag[nome], mae_tag[nome] = r2, mae_norm
        linhas.append({
            "método": nome,
            "MAE norm. (média)": mae_norm.mean(),
            "MAE norm. (mediana)": mae_norm.median(),
            "R² (mediana)": r2.median(),
            "TAGs com R² < 0.5": int((r2 < 0.5).sum()),
            "tempo treino (s)": (tempos or {}).get(nome, np.nan),
            ".pkl (MB)": os.path.getsize(model.DIR_MODELS + bloco["name_model"]) / 2**20,
        })
    tabela = pd.DataFrame(linhas).set_index("método").sort_values("MAE norm. (mediana)")
    return tabela, pd.DataFrame(r2_tag), pd.DataFrame(mae_tag)


# ------------------------------------------------------------------- outlier extremo


def _escala_robusta(valores):
    """Mediana e escala robusta com piso: max(IQR, 1.4826*MAD, 5% da |mediana|).

    O piso relativo evita que uma TAG quantizada (MAD ~ 0) tenha toda variação normal
    tratada como falha.
    """
    mediana = valores.median()
    iqr = valores.quantile(0.75) - valores.quantile(0.25)
    mad = (valores - mediana).abs().median() * _MAD_TO_SIGMA
    escala = pd.concat([iqr, mad, 0.05 * mediana.abs()], axis=1).max(axis=1)
    return mediana, escala.replace(0, np.nan)


def limpar_falhas_treino(df, colunas, fator=50.0):
    """Remove do treino do imputer as linhas com falha grosseira de sensor.

    Falha grosseira = |x - mediana| > fator * escala robusta (ver `_escala_robusta`):
    temperatura em 6e6, vazão em 1e6, valor x10. Medianas e quartis aguentam até ~25% de
    contaminação, então o critério não é arrastado pela própria falha. Sem essa limpeza o
    imputer aprende a reproduzir a falha e o envelope de outlier fica largo demais.
    Retorna (df_limpo, tabela por TAG com a fração de linhas com falha grosseira).
    """
    valores = df[colunas].apply(pd.to_numeric, errors="coerce")
    mediana, escala = _escala_robusta(valores)
    grosseira = (valores - mediana).abs().div(escala) > fator
    linhas_ruins = grosseira.any(axis=1)
    tabela = pd.DataFrame({
        "linhas com falha grosseira": grosseira.sum(),
        "% do treino": (100 * grosseira.mean()).round(2),
        "mediana": mediana,
        "min": valores.min(),
        "max": valores.max(),
    })
    tabela = tabela[tabela["linhas com falha grosseira"] > 0].sort_values("linhas com falha grosseira", ascending=False)
    print("Linhas removidas do treino do imputer: %d de %d (%.1f%%)" % (linhas_ruins.sum(), len(df), 100 * linhas_ruins.mean()))
    return df[~linhas_ruins.to_numpy()].reset_index(drop=True), tabela


def estatisticas_outlier(df_treino, ts=None, q_baixo=0.01, q_alto=0.99):
    """Estatísticas por TAG do período de treino usadas na detecção de outlier.

    `passo_p999` = p99.9 de |x[t] - x[t-1]|, o maior salto normal entre amostras. Com `ts`,
    o salto entre períodos de treino (ou sobre linhas removidas) fica de fora.
    """
    valores = df_treino.apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    mediana = valores.median()
    saltos = valores.diff().abs()
    if ts is not None:
        tempos = pd.to_datetime(pd.Series(ts)).reset_index(drop=True)
        contiguo = (tempos.diff().dt.total_seconds() <= 1.5 * _passo_segundos(tempos)).to_numpy()
        saltos = saltos[contiguo]
    return pd.DataFrame({
        "median": mediana,
        "mad": (valores - mediana).abs().median(),
        "q1": valores.quantile(0.25),
        "q3": valores.quantile(0.75),
        "p_baixo": valores.quantile(q_baixo),
        "p_alto": valores.quantile(q_alto),
        "std": valores.std(),
        "passo_p999": saltos.quantile(0.999),
    })


def _filtra_corridas(flag, min_amostras):
    if min_amostras <= 1:
        return flag
    saida = np.zeros(len(flag), dtype=bool)
    for ini, fim in _corridas(flag):
        if fim - ini >= min_amostras:
            saida[ini:fim] = True
    return saida


def fator_desvio(df, stats):
    """|x - mediana| / max(faixa, |mediana|) por célula, com mediana e faixa (p_alto - p_baixo) do treino.

    Quantas vezes o valor se afasta do comportamento usual da TAG, medido pelo maior entre a
    faixa normal e o próprio nível da TAG. O nível no denominador faz uma queda a zero valer
    no máximo ~1: parada de equipamento e trip não contam como falha de sensor. A faixa
    cobre a TAG que opera perto de zero. TAG sem faixa no treino (constante) fica NaN.
    """
    valores = df.apply(pd.to_numeric, errors="coerce")
    mediana = stats["median"].reindex(df.columns)
    faixa = (stats["p_alto"] - stats["p_baixo"]).reindex(df.columns)
    escala = pd.concat([faixa, mediana.abs()], axis=1).max(axis=1).where(faixa > 0)
    return (valores - mediana).abs().div(escala)


def salto_relativo(df, stats):
    """|x[t] - x[t-1]| / passo_p999 do treino: quantas vezes o salto supera o maior salto normal da TAG.

    NaN onde não há amostra anterior válida (primeira linha do teste ou logo depois de um NaN).
    """
    valores = df.apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    saltos = valores.diff().abs()
    passo = stats["passo_p999"].reindex(df.columns)
    return saltos.div(passo.where(passo > 0, np.finfo(float).tiny)).set_axis(df.index)


def _fundo_de_escala(x, s, limites, tol=0.05, salto=3.0):
    """Marcação de fundo de escala de uma TAG (`x` em array), com `limites` = (inferior, superior) do instrumento.

    Dispara na amostra que cumpre as duas condições
      1. está na zona do limite: a menos de `tol` da distância entre a borda da faixa normal
         (p_baixo/p_alto do treino) e o limite;
      2. chegou lá de forma abrupta: max(|x[t] - x[t-1]|, |x[t] - x[t-2]|) > `salto` * passo_p999
         (ou sem amostra anterior). O processo que sobe aos poucos até saturar o instrumento é
         desvio real e fica cru.
    Depois segue marcando, só para frente, enquanto o valor estiver mais perto do limite do que da
    faixa normal. Por isso a rampa interpolada que o histórico grava depois de um termopar ir a
    fundo de escala também é marcada, mas o equipamento que para logo após um pico no limite
    (queda a zero do lado oposto) não é.
    """
    flag = np.zeros(len(x), dtype=bool)
    if not limites:
        return flag
    serie = pd.Series(x)
    passos = pd.concat([(serie - serie.shift(1)).abs(), (serie - serie.shift(2)).abs()], axis=1).max(axis=1)
    passo = s["passo_p999"] if s["passo_p999"] > 0 else np.finfo(float).tiny
    abrupto = ((passos > salto * passo) | passos.isna()).to_numpy()
    for limite, borda, sinal in ((limites[0], s["p_baixo"], -1.0), (limites[1], s["p_alto"], 1.0)):
        if limite is None:
            continue
        distancia = (limite - borda) * sinal
        if not distancia > 0:
            raise ValueError("limite de escala %g está dentro da faixa normal do treino (%.4g a %.4g)"
                             % (limite, s["p_baixo"], s["p_alto"]))
        perto = (x - limite) * sinal >= -tol * distancia
        zona = (x - borda) * sinal >= distancia / 2
        gatilho = perto & abrupto
        for ini, fim in _corridas(zona):
            disparo = np.flatnonzero(gatilho[ini:fim])
            if disparo.size:
                flag[ini + disparo[0]:fim] = True
    return flag


def detectar_outliers(df, stats, metodo="extremo", fator=3.0, salto=20.0, k_retorno=1.0, k=2.0, z=8.0,
                      min_amostras=1, limites_escala=None, tol_escala=0.05, salto_escala=3.0):
    """Máscara booleana (mesmo shape de `df`) dos outliers extremos (falha de sensor).

    - "extremo" (padrão): une duas regras de falha de sensor. A primeira é o valor absurdo, que
      dispara na amostra que cumpre as duas condições
        1. `fator_desvio` > `fator`: valor absurdo para o nível da TAG;
        2. `salto_relativo` > `salto`: chegou lá num salto que o sensor nunca deu no treino.
      Depois segue marcando, só para frente, até o valor voltar para
      [p_baixo - k_retorno*R, p_alto + k_retorno*R]. Falha de processo evolui aos poucos e não
      cumpre a condição 2, por maior que fique; parada, trip e troca de setpoint ficam abaixo
      de `fator`. O que vem antes do disparo nunca é marcado, como no runtime, que só vê o
      passado. Nos notebooks: desvio de processo chegou a fator 2.5 com salto 1.6x o normal;
      falha de sensor teve salto acima de 130x o normal.
      A segunda é o fundo de escala (`_fundo_de_escala`), só para as TAGs de `limites_escala`
      = {tag: (limite_inferior, limite_superior)}, com None no lado sem limite. Ela cobre a
      falha que fica perto da faixa operacional, onde a primeira regra não alcança: termopar
      a ~1000 °C que vai para -58 °C tem fator ~1.
    - "envelope": fora de [p_baixo - k*R, p_alto + k*R]. Marca também desvio de processo.
    - "robust_z": |x - mediana| / (1.4826*MAD) > z, com IQR quando MAD = 0 (produção).
    TAG sem escala no treino (constante) não é avaliada, como na produção.
    """
    mascara = pd.DataFrame(False, index=df.index, columns=df.columns)
    if metodo == "extremo":
        # Sem amostra anterior não dá para medir o salto: valor absurdo ali já basta (teste que começa em falha).
        saltos = salto_relativo(df, stats)
        gatilho = ((fator_desvio(df, stats) > fator) & ((saltos > salto) | saltos.isna())).to_numpy()
    for j, tag in enumerate(df.columns):
        if tag not in stats.index:
            continue
        s = stats.loc[tag]
        x = pd.to_numeric(df[tag], errors="coerce")
        if metodo == "extremo":
            faixa = s["p_alto"] - s["p_baixo"]
            if not faixa > 0:
                continue
            flag = _fundo_de_escala(x.to_numpy(dtype=float), s, (limites_escala or {}).get(tag),
                                    tol=tol_escala, salto=salto_escala)
            if gatilho[:, j].any():
                fora = ~((x >= s["p_baixo"] - k_retorno * faixa) & (x <= s["p_alto"] + k_retorno * faixa)).to_numpy()
                for ini, fim in _corridas(fora):
                    disparo = np.flatnonzero(gatilho[ini:fim, j])
                    if disparo.size:
                        flag[ini + disparo[0]:fim] = True
            mascara[tag] = _filtra_corridas(flag & x.notna().to_numpy(), min_amostras)
            continue
        if metodo == "envelope":
            faixa = s["p_alto"] - s["p_baixo"]
            if not faixa > 0:
                continue
            flag = (x < s["p_baixo"] - k * faixa) | (x > s["p_alto"] + k * faixa)
        elif metodo == "robust_z":
            if s["mad"] > 0:
                escala = s["mad"] * _MAD_TO_SIGMA
            elif s["q3"] > s["q1"]:
                escala = (s["q3"] - s["q1"]) / _IQR_TO_SIGMA
            else:
                continue
            flag = (x - s["median"]).abs() / escala > z
        else:
            raise ValueError("metodo deve ser 'extremo', 'envelope' ou 'robust_z'")
        mascara[tag] = _filtra_corridas(flag.fillna(False).to_numpy(), min_amostras)
    return mascara


def comparar_regras_outlier(df, stats, fator=3.0, salto=20.0, k_retorno=1.0, k=2.0, z=8.0, min_amostras_z=3,
                            limites_escala=None, tol_escala=0.05, salto_escala=3.0):
    """Pontos marcados por TAG na regra extrema (usada) e nas regras que marcam desvio de processo.

    "só fundo de escala" é a parte da regra usada que vem de `limites_escala`.
    """
    escala = dict(limites_escala=limites_escala, tol_escala=tol_escala, salto_escala=salto_escala)
    contagens = {
        "extremo (usada)": detectar_outliers(df, stats, "extremo", fator=fator, salto=salto, k_retorno=k_retorno,
                                             **escala).sum(),
        "só fundo de escala": detectar_outliers(df, stats, "extremo", fator=np.inf, **escala).sum(),
        "envelope k=%g" % k: detectar_outliers(df, stats, "envelope", k=k).sum(),
        "robust_z produção": detectar_outliers(df, stats, "robust_z", z=z, min_amostras=min_amostras_z).sum(),
    }
    tabela = pd.DataFrame(contagens)
    tabela["fator de desvio máx"] = fator_desvio(df, stats).max()
    tabela["salto máx (x normal)"] = salto_relativo(df, stats).max()
    tabela["mediana treino"] = stats["median"].reindex(tabela.index)
    tabela["faixa treino p_baixo"] = stats["p_baixo"].reindex(tabela.index)
    tabela["faixa treino p_alto"] = stats["p_alto"].reindex(tabela.index)
    tabela["min teste"] = df.min()
    tabela["max teste"] = df.max()
    tabela["limites de escala"] = [str((limites_escala or {}).get(tag, "")) for tag in tabela.index]
    tabela = tabela[tabela[list(contagens)].sum(axis=1) > 0]
    return tabela.sort_values(["extremo (usada)", "fator de desvio máx"], ascending=False)


def candidatos_fundo_escala(df, stats, k_retorno=1.0, min_eventos=2):
    """Valores em que o sensor travou depois de sair abruptamente da faixa normal, para ajudar a montar `limites_escala`.

    Para cada saída de [p_baixo - k_retorno*R, p_alto + k_retorno*R] que começa até 2 amostras
    depois de um valor dentro de [p_baixo, p_alto], guarda o valor mais frequente do trecho
    (arredondado a 0,2% da faixa R). Valor que se repete em `min_eventos` saídas ou mais é
    candidato: um desvio de processo raramente para duas vezes no mesmo valor, e um instrumento
    em fundo de escala para sempre no mesmo. A tabela é só sugestão. TAG quantizada (valor
    inteiro) e parada de equipamento também aparecem, então confirme com o range do instrumento.
    """
    valores = df.apply(pd.to_numeric, errors="coerce")
    linhas = []
    for tag in valores.columns:
        if tag not in stats.index:
            continue
        s = stats.loc[tag]
        faixa = s["p_alto"] - s["p_baixo"]
        if not faixa > 0:
            continue
        x = valores[tag].to_numpy(dtype=float)
        dentro = (x >= s["p_baixo"]) & (x <= s["p_alto"])
        fora = (x < s["p_baixo"] - k_retorno * faixa) | (x > s["p_alto"] + k_retorno * faixa)
        paradas = []
        for ini, fim in _corridas(fora):
            if not ((ini >= 1 and dentro[ini - 1]) or (ini >= 2 and dentro[ini - 2])):
                continue
            trecho = x[ini:fim][~np.isnan(x[ini:fim])]
            if trecho.size:
                v, n = np.unique(np.round(trecho / (0.002 * faixa)) * 0.002 * faixa, return_counts=True)
                paradas.append(float(v[n.argmax()]))
        for valor, eventos in pd.Series(paradas, dtype=float).value_counts().items():
            if eventos >= min_eventos and valor != 0:
                linhas.append({"tag": tag, "lado": "inferior" if valor < s["p_baixo"] else "superior",
                               "valor": valor, "eventos": int(eventos), "saídas abruptas": len(paradas),
                               "faixa normal": "%.4g a %.4g" % (s["p_baixo"], s["p_alto"])})
    colunas = ["tag", "lado", "valor", "eventos", "saídas abruptas", "faixa normal"]
    return pd.DataFrame(linhas).reindex(columns=colunas).sort_values(["eventos", "tag"], ascending=[False, True]).reset_index(drop=True)


def listar_desvios(df, ts, stats, mask_outlier, k_retorno=1.0):
    """Cada trecho contíguo fora de [p_baixo - k_retorno*R, p_alto + k_retorno*R], por TAG.

    "imputado" = o trecho contém outlier extremo (pelo menos parte dele vira NaN e é
    reconstruída). "mantido" = desvio que o MSCVAE vê cru, sem imputação.
    """
    ts = pd.to_datetime(pd.Series(ts)).reset_index(drop=True)
    fatores = fator_desvio(df, stats)
    saltos = salto_relativo(df, stats)
    linhas = []
    for tag in df.columns:
        if tag not in stats.index:
            continue
        s = stats.loc[tag]
        faixa = s["p_alto"] - s["p_baixo"]
        if not faixa > 0:
            continue
        x = pd.to_numeric(df[tag], errors="coerce").to_numpy()
        fora = (x < s["p_baixo"] - k_retorno * faixa) | (x > s["p_alto"] + k_retorno * faixa)
        marcado = np.asarray(mask_outlier[tag], dtype=bool)
        for ini, fim in _corridas(fora):
            linhas.append({
                "tag": tag,
                "tratamento": "imputado" if marcado[ini:fim].any() else "mantido",
                "inicio": ts.iloc[ini],
                "fim": ts.iloc[fim - 1],
                "amostras": int(fim - ini),
                "amostras imputadas": int(marcado[ini:fim].sum()),
                "valor min": np.nanmin(x[ini:fim]),
                "valor max": np.nanmax(x[ini:fim]),
                "fator de desvio máx": float(np.nanmax(fatores[tag].to_numpy()[ini:fim])),
                "salto máx (x normal)": saltos[tag].iloc[ini:fim].max(),
                "faixa normal": "%.4g a %.4g" % (s["p_baixo"], s["p_alto"]),
            })
    colunas = ["tag", "tratamento", "inicio", "fim", "amostras", "amostras imputadas", "valor min", "valor max",
               "fator de desvio máx", "salto máx (x normal)", "faixa normal"]
    desvios = pd.DataFrame(linhas).reindex(columns=colunas)
    return desvios.sort_values("fator de desvio máx", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- lacunas sintéticas


def injetar_lacunas(df, ts, lacunas_por_tag=3, dur_min=10, dur_max=240, frac_simultaneas=0.3,
                    tags=None, evitar=None, margem=60, seed=42):
    """Injeta lacunas NaN contíguas com o valor real conhecido.

    Cada TAG recebe `lacunas_por_tag` lacunas de `dur_min` a `dur_max` minutos; uma fração
    `frac_simultaneas` delas derruba também 1-2 outras TAGs no mesmo trecho (falha de cartão
    ou de comunicação). As lacunas ficam a `margem` amostras das linhas de `evitar` (outliers
    reais), para a predição de referência ser limpa nesses trechos. Retorna
    (df_com_nan, mascara_sintetica).
    """
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    n = len(df)
    tags = list(df.columns) if tags is None else list(tags)
    indice_tag = {tag: j for j, tag in enumerate(df.columns)}
    passo = _passo_segundos(ts)
    nmin = max(1, int(round(dur_min * 60.0 / passo)))
    nmax = max(nmin, int(round(dur_max * 60.0 / passo)))

    livre = np.ones(n, dtype=bool)
    if evitar is not None:
        ocupado = np.asarray(evitar, dtype=bool).any(axis=1).astype(float)
        livre = np.convolve(ocupado, np.ones(2 * margem + 1), mode="same") == 0
    livre[0] = False  # a primeira linha não tem âncora

    mascara = np.zeros(df.shape, dtype=bool)
    nao_colocadas = 0
    for tag in tags:
        j = indice_tag[tag]
        for _ in range(lacunas_por_tag):
            for _tentativa in range(200):
                dur = int(rng.integers(nmin, nmax + 1))
                if dur >= n - 1:
                    break
                ini = int(rng.integers(1, n - dur))
                if livre[ini:ini + dur].all() and not mascara[ini:ini + dur, j].any():
                    alvo = [j]
                    outras = [indice_tag[t] for t in tags if t != tag]
                    if outras and rng.random() < frac_simultaneas:
                        extra = rng.choice(outras, size=min(len(outras), int(rng.integers(1, 3))), replace=False)
                        alvo += [int(e) for e in extra]
                    mascara[ini:ini + dur, alvo] = True
                    break
            else:
                nao_colocadas += 1
    if nao_colocadas:
        print("Aviso: %d lacuna(s) não couberam fora dos trechos com outlier" % nao_colocadas)

    mascara = pd.DataFrame(mascara, columns=df.columns)
    return df.mask(mascara), mascara


# ------------------------------------------------------------------------- aplicação


def aplicar_imputacao(df, ts, imputer=None, mask_outlier=None, start_minutes=10, max_bad_ratio=0.7):
    """Reconstrói NaN + outliers de `df` com a regra de produção.

    `imputer=None` é o baseline: ffill (e bfill no início) de tudo. As colunas de `df`
    precisam estar na ordem do treino do imputer. Linhas bloqueadas (>= max_bad_ratio das
    variáveis inválidas) continuam NaN no caminho do imputer. Retorna (df_imputado, info).
    """
    inicio = time.time()
    df = df.reset_index(drop=True)
    X = df.to_numpy(dtype=float, copy=True)
    if mask_outlier is not None:
        X[np.asarray(mask_outlier, dtype=bool)] = np.nan
    invalido = np.isnan(X)
    mask_ffill = np.zeros_like(invalido)
    mask_imputer = np.zeros_like(invalido)
    bloqueada = np.zeros(len(X), dtype=bool)

    if imputer is None:
        X = pd.DataFrame(X).ffill().bfill().to_numpy()
        mask_ffill = invalido.copy()
    else:
        tempos = pd.to_datetime(pd.Series(ts)).reset_index(drop=True)
        # total_seconds e nao astype(int64): no pandas 3 a resolucao do datetime64 varia (s/ms/us/ns).
        segundos = (tempos - tempos.iloc[0]).dt.total_seconds().to_numpy()
        passo = _passo_segundos(ts)
        bloqueada = invalido.mean(axis=1) >= max_bad_ratio
        for j in range(X.shape[1]):
            for ini, fim in _corridas(invalido[:, j]):
                minutos = (segundos[fim - 1] - segundos[ini] + passo) / 60.0
                if ini > 0 and minutos < start_minutes:
                    X[ini:fim, j] = X[ini - 1, j]
                    mask_ffill[ini:fim, j] = True
                else:
                    mask_imputer[ini:fim, j] = True
        mask_imputer &= ~bloqueada[:, None]
        if mask_imputer.any():
            reconstruido = imputer.transform(X.copy())
            X[mask_imputer] = reconstruido[mask_imputer]

    info = {
        "mask_ffill": pd.DataFrame(mask_ffill, columns=df.columns),
        "mask_imputer": pd.DataFrame(mask_imputer, columns=df.columns),
        "linhas_bloqueadas": bloqueada,
        "tempo (s)": time.time() - inicio,
    }
    return pd.DataFrame(X, columns=df.columns), info


def resumo_imputacao(infos):
    return pd.DataFrame({
        nome: {
            "células ffill": int(info["mask_ffill"].values.sum()),
            "células imputer": int(info["mask_imputer"].values.sum()),
            "linhas bloqueadas": int(info["linhas_bloqueadas"].sum()),
            "tempo (s)": round(info["tempo (s)"], 2),
        }
        for nome, info in infos.items()
    }).T


def preparar_predicao(df, ts):
    """Descarta as linhas que ficaram com NaN e devolve (df, timestamps) alinhados."""
    ok = ~df.isna().any(axis=1).to_numpy()
    ts = pd.Series(ts).reset_index(drop=True)
    return df.reset_index(drop=True)[ok].reset_index(drop=True), ts[ok].reset_index(drop=True)


def metricas_lacunas(df_verdade, df_imputados, mask_sintetico, std_treino):
    """Erro nas células das lacunas sintéticas (valor real conhecido), por método e por TAG."""
    verdade = df_verdade.reset_index(drop=True).to_numpy(dtype=float)
    mascara = np.asarray(mask_sintetico, dtype=bool)
    linhas = []
    for nome, df_imp in df_imputados.items():
        previsto = df_imp.to_numpy(dtype=float)
        for j, tag in enumerate(df_verdade.columns):
            alvo = mascara[:, j] & ~np.isnan(verdade[:, j])
            sel = alvo & ~np.isnan(previsto[:, j])
            if sel.sum() < 2:
                continue
            erro = previsto[sel, j] - verdade[sel, j]
            variancia = np.sum((verdade[sel, j] - verdade[sel, j].mean()) ** 2)
            std = std_treino.get(tag, np.nan) or np.nan
            linhas.append({
                "método": nome,
                "tag": tag,
                "MAE": np.mean(np.abs(erro)),
                "RMSE": np.sqrt(np.mean(erro ** 2)),
                "R²": 1 - np.sum(erro ** 2) / variancia if variancia > 0 else np.nan,
                "MAE norm.": np.mean(np.abs(erro)) / std,
                "RMSE norm.": np.sqrt(np.mean(erro ** 2)) / std,
                "cobertura": sel.sum() / alvo.sum(),
            })
    por_tag = pd.DataFrame(linhas)
    resumo = por_tag.groupby("método").agg(**{
        "MAE norm. (média)": ("MAE norm.", "mean"),
        "MAE norm. (mediana)": ("MAE norm.", "median"),
        "RMSE norm. (mediana)": ("RMSE norm.", "median"),
        "R² (mediana)": ("R²", "median"),
        "cobertura (%)": ("cobertura", lambda c: 100 * c.mean()),
    }).sort_values("MAE norm. (mediana)")
    return resumo, por_tag


# ------------------------------------------------------------------------ eventos


def listar_eventos(mask_nan, mask_outlier, ts, df_original=None):
    """Uma linha por corrida contígua de NaN sintético ou outlier, por TAG."""
    ts = pd.to_datetime(pd.Series(ts)).reset_index(drop=True)
    passo = _passo_segundos(ts)
    linhas = []
    for tipo, mascara in (("NaN sintético", mask_nan), ("Outlier (falha de sensor)", mask_outlier)):
        if mascara is None:
            continue
        for tag in mascara.columns:
            for ini, fim in _corridas(mascara[tag].to_numpy()):
                linha = {
                    "tag": tag,
                    "tipo": tipo,
                    "inicio": ts.iloc[ini],
                    "fim": ts.iloc[fim - 1],
                    "amostras": int(fim - ini),
                    "duração (min)": (ts.iloc[fim - 1] - ts.iloc[ini]).total_seconds() / 60.0 + passo / 60.0,
                }
                if df_original is not None and tipo.startswith("Outlier"):
                    trecho = df_original[tag].iloc[ini:fim]
                    linha["valor min"], linha["valor max"] = trecho.min(), trecho.max()
                linhas.append(linha)
    colunas = ["tag", "tipo", "inicio", "fim", "amostras", "duração (min)", "valor min", "valor max"]
    eventos = pd.DataFrame(linhas).reindex(columns=colunas)
    return eventos.sort_values("amostras", ascending=False).reset_index(drop=True)


def _escolher_diversos(eventos, n):
    """As `n` primeiras linhas com TAG diferente e sem sobreposição no tempo.

    Uma falha que derruba 11 TAGs juntas renderia 11 gráficos do mesmo trecho; se não
    houver eventos separados suficientes, completa com os que sobrarem.
    """
    escolhidos, intervalos = [], []
    for i, ev in eventos.iterrows():
        if len(escolhidos) == n:
            break
        if ev["tag"] in set(eventos.loc[escolhidos, "tag"]):
            continue
        if any(ev["inicio"] <= f and i0 <= ev["fim"] for i0, f in intervalos):
            continue
        escolhidos.append(i)
        intervalos.append((ev["inicio"], ev["fim"]))
    resto = eventos.drop(index=escolhidos).drop_duplicates("tag")
    resto = resto[~resto["tag"].isin(eventos.loc[escolhidos, "tag"])]
    return pd.concat([eventos.loc[escolhidos], resto.head(n - len(escolhidos))])


def selecionar_eventos(eventos, n=6):
    """Os maiores eventos de cada tipo, em TAGs e trechos diferentes, para os gráficos antes x depois."""
    tipos = list(eventos["tipo"].unique())
    if not tipos:
        return eventos
    por_tipo = max(1, n // len(tipos))
    partes = [_escolher_diversos(eventos[eventos["tipo"] == tipo], por_tipo) for tipo in tipos]
    return pd.concat(partes).reset_index(drop=True)


def selecionar_desvios_mantidos(desvios, n=4):
    """Desvios que ficaram sem imputação, em TAGs e trechos diferentes.

    Alterna o mais perto do limite (maior fator) com o mais longo (evento de processo que
    uma regra mais frouxa apagaria inteiro).
    """
    mantidos = desvios[desvios["tratamento"] == "mantido"]
    por_fator = mantidos.sort_values("fator de desvio máx", ascending=False).index
    por_duracao = mantidos.sort_values("amostras", ascending=False).index
    ordem = pd.unique(np.ravel(np.column_stack([por_fator, por_duracao])))
    return _escolher_diversos(mantidos.loc[ordem], n).reset_index(drop=True)


# ------------------------------------------------------------------------- gráficos


def plot_metricas(tabela, metricas, titulo):
    fig = make_subplots(rows=1, cols=len(metricas), subplot_titles=metricas)
    for c, metrica in enumerate(metricas, start=1):
        fig.add_trace(go.Bar(
            x=tabela.index, y=tabela[metrica],
            marker_color=[_cor(m, i) for i, m in enumerate(tabela.index)],
            text=tabela[metrica].round(3), textposition="outside", showlegend=False,
        ), row=1, col=c)
    return _layout(fig, titulo, altura=420)


def plot_heatmap_tags(matriz, titulo, zmin=0, zmax=1, escala="RdYlGn"):
    matriz = matriz.sort_index()
    fig = go.Figure(go.Heatmap(
        z=matriz.values, x=matriz.columns, y=matriz.index, zmin=zmin, zmax=zmax,
        colorscale=escala, text=np.round(matriz.values, 2), texttemplate="%{text}",
    ))
    fig = _layout(fig, titulo, altura=max(400, 22 * len(matriz) + 120))
    fig.update_layout(hovermode="closest")
    return fig


def _grade(ts, valores, passo):
    """Insere um NaN em cada buraco de tempo (desligado removido) para a linha quebrar ali."""
    serie = pd.Series(np.asarray(valores, dtype=float), index=pd.to_datetime(pd.Series(ts)).to_numpy())
    serie = serie[~serie.index.duplicated()]
    if len(serie) < 2:
        return serie
    tempos = serie.index.to_series()
    buraco = (tempos.diff().dt.total_seconds() > 1.5 * passo).to_numpy()
    if not buraco.any():
        return serie
    quebras = pd.Series(np.nan, index=tempos.shift(1)[buraco].to_numpy() + pd.Timedelta(seconds=passo))
    return pd.concat([serie, quebras]).sort_index()


def _faixas(fig, ts, mascara, cor, opacidade):
    if mascara is None:
        return
    for ini, fim in _corridas(mascara):
        fig.add_vrect(x0=ts.iloc[ini], x1=ts.iloc[fim - 1], fillcolor=cor, opacity=opacidade,
                      line_width=0, row="all", col=1)


def plot_imputacao_tag(tag, ts, df_antes, df_depois, mask_nan=None, mask_outlier=None, df_verdade=None,
                       inicio=None, fim=None, margem_min=60, descricao=None):
    """TAG antes (com NaN/outlier) x depois da imputação, um traço por método.

    `df_depois` é {metodo: df_imputado}. Linha de cima: série como chegou, com a
    magnitude real do outlier. Linha de baixo: série reconstruída, com o valor real
    tracejado nas lacunas sintéticas.
    """
    ts = pd.to_datetime(pd.Series(ts)).reset_index(drop=True)
    passo = _passo_segundos(ts)
    inicio = ts.iloc[0] if inicio is None else pd.Timestamp(inicio)
    fim = ts.iloc[-1] if fim is None else pd.Timestamp(fim)
    margem = pd.Timedelta(minutes=margem_min)
    sel = ((ts >= inicio - margem) & (ts <= fim + margem)).to_numpy()
    ts_sel = ts[sel].reset_index(drop=True)

    m_nan = np.zeros(sel.sum(), dtype=bool) if mask_nan is None else mask_nan[tag].to_numpy()[sel]
    m_out = np.zeros(sel.sum(), dtype=bool) if mask_outlier is None else mask_outlier[tag].to_numpy()[sel]
    evento = m_nan | m_out
    # Estende 1 amostra para cada lado para o trecho imputado encostar na série.
    borda = evento | np.r_[evento[1:], False] | np.r_[False, evento[:-1]]

    antes = df_antes[tag].reset_index(drop=True).to_numpy(dtype=float)[sel]
    titulo_y = tag if not descricao else "%s<br><sup>%s</sup>" % (tag, descricao)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Antes da imputação", "Depois da imputação"))

    serie = _grade(ts_sel, antes, passo)
    fig.add_trace(go.Scatter(x=serie.index, y=serie.values, name="Original", line_color=COR_SERIE,
                             connectgaps=False), row=1, col=1)
    if m_out.any():
        fig.add_trace(go.Scatter(x=ts_sel[m_out], y=antes[m_out], mode="markers", name="Outlier (falha de sensor)",
                                 marker=dict(color=COR_OUTLIER, symbol="x", size=7)), row=1, col=1)

    base = np.where(evento, np.nan, antes)
    serie = _grade(ts_sel, base, passo)
    fig.add_trace(go.Scatter(x=serie.index, y=serie.values, name="Valores válidos", line_color=COR_SERIE,
                             connectgaps=False, showlegend=False), row=2, col=1)
    if df_verdade is not None and m_nan.any():
        verdade = df_verdade[tag].reset_index(drop=True).to_numpy(dtype=float)[sel]
        serie = _grade(ts_sel, np.where(m_nan | (borda & ~m_out), verdade, np.nan), passo)
        fig.add_trace(go.Scatter(x=serie.index, y=serie.values, name="Valor real (oculto)", connectgaps=False,
                                 line=dict(color="black", dash="dash", width=2)), row=2, col=1)
    for i, (nome, df_imp) in enumerate(df_depois.items()):
        depois = df_imp[tag].reset_index(drop=True).to_numpy(dtype=float)[sel]
        serie = _grade(ts_sel, np.where(borda, depois, np.nan), passo)
        fig.add_trace(go.Scatter(x=serie.index, y=serie.values, name=nome, mode="lines+markers",
                                 marker=dict(size=3), line=dict(color=_cor(nome, i), width=1.5),
                                 connectgaps=False), row=2, col=1)

    _faixas(fig, ts_sel, m_nan, COR_NAN, 0.2)
    _faixas(fig, ts_sel, m_out, COR_OUTLIER, 0.12)
    fig = _layout(fig, titulo_y, altura=650)
    # Outlier ordens de grandeza acima achata a série normal: escala log no painel de cima.
    validos = antes[np.isfinite(antes)]
    normais = antes[~m_out & np.isfinite(antes)]
    if m_out.any() and normais.size and (validos > 0).all() and validos.max() > 100 * np.median(normais):
        fig.update_yaxes(type="log", row=1, col=1)
    return fig


def plot_desvio_mantido(tag, ts, df, stats, inicio, fim, fator=3.0, salto=20.0, margem_min=120, descricao=None,
                        limites=None):
    """Desvio que ficou sem imputação: a série crua, a faixa normal do treino e o limite de outlier.

    Serve para conferir que a regra extrema não tocou em desvio de processo.
    """
    ts = pd.to_datetime(pd.Series(ts)).reset_index(drop=True)
    passo = _passo_segundos(ts)
    margem = pd.Timedelta(minutes=margem_min)
    inicio, fim = pd.Timestamp(inicio), pd.Timestamp(fim)
    sel = ((ts >= inicio - margem) & (ts <= fim + margem)).to_numpy()
    no_desvio = ((ts >= inicio) & (ts <= fim)).to_numpy()
    x = pd.to_numeric(df[tag], errors="coerce").reset_index(drop=True).to_numpy(dtype=float)[sel]
    s = stats.loc[tag]
    escala = max(s["p_alto"] - s["p_baixo"], abs(s["median"]))
    limite_baixo, limite_alto = s["median"] - fator * escala, s["median"] + fator * escala
    fmax = float(fator_desvio(df[[tag]], stats)[tag].to_numpy()[no_desvio].max())
    smax = salto_relativo(df[[tag]], stats)[tag][no_desvio].max()
    texto_salto = "salto %.0fx o normal" % smax if np.isfinite(smax) else "sem amostra anterior"

    fig = go.Figure()
    fig.add_hrect(y0=s["p_baixo"], y1=s["p_alto"], fillcolor="#6A994E", opacity=0.15, line_width=0,
                  annotation_text="faixa normal do treino (p1–p99)", annotation_position="top left")
    fig.add_vrect(x0=inicio, x1=fim, fillcolor=COR_LIMIAR, opacity=0.12, line_width=0)
    for limite in (limites or ()):
        if limite is not None:
            fig.add_hline(y=limite, line_dash="dash", line_color=COR_OUTLIER,
                          annotation_text="fundo de escala %g" % limite, annotation_position="bottom right")
    serie = _grade(ts[sel].reset_index(drop=True), x, passo)
    fig.add_trace(go.Scatter(x=serie.index, y=serie.values, name="Valor cru (vai para o MSCVAE)",
                             line_color=COR_SERIE, connectgaps=False))
    titulo = "%s — mantido: desvio %.1fx (limite %gx), %s (limite %gx)" % (tag, fmax, fator, texto_salto, salto)
    titulo += "<br><sup>%soutlier extremo exige valor < %.4g ou > %.4g, alcançado num salto</sup>" % (
        (descricao + " · ") if descricao else "", limite_baixo, limite_alto)
    return _layout(fig, titulo, altura=420)


def _afetado(mascara, janela):
    """Linhas cujo índice de anomalia enxerga alguma célula da máscara (janela do MSCVAE)."""
    if mascara is None:
        return None
    uniao = pd.Series(np.asarray(mascara, dtype=bool).any(axis=1).astype(float))
    return uniao.rolling(janela + 1, min_periods=1).max().astype(bool).to_numpy()


def _serie_phi(predicao, threshold):
    serie = pd.Series(np.asarray(predicao["phi"], dtype=float) / threshold,
                      index=pd.to_datetime(pd.Series(predicao["timestamp"])).to_numpy())
    # O predict repete o último timestamp (janela que termina no fim do df).
    return serie[~serie.index.duplicated()]


def resumo_efeito_predicao(predicoes, threshold, ts, mask_sintetico=None, mask_outlier=None,
                           janela=10, referencia="Referência"):
    """Alarmes (phi/limiar > 1) por cenário: total, nas lacunas sintéticas e nos outliers reais.

    "cobertura" = fração dos instantes afetados que ainda tem predição (o dropna perde
    monitoramento nas lacunas). O |Δ| é contra a referência sem lacunas sintéticas.
    """
    ts = pd.to_datetime(pd.Series(ts)).reset_index(drop=True)

    def por_tempo(mascara):
        if mascara is None:
            return None
        serie = pd.Series(_afetado(mascara, janela), index=ts.to_numpy())
        return serie[~serie.index.duplicated()]

    em_lacuna, em_outlier = por_tempo(mask_sintetico), por_tempo(mask_outlier)
    ref = _serie_phi(predicoes[referencia], threshold) if referencia in predicoes else None
    linhas = []
    for nome, predicao in predicoes.items():
        s = _serie_phi(predicao, threshold)
        linha = {"cenário": nome, "pontos previstos": len(s), "% alarmes (total)": 100 * (s > 1).mean()}
        for rotulo, afetado in (("nas lacunas sintéticas", em_lacuna), ("nos outliers reais", em_outlier)):
            if afetado is None or not afetado.any():
                continue
            marca = afetado.reindex(s.index).fillna(False).astype(bool)
            linha["cobertura %s (%%)" % rotulo] = 100 * marca.sum() / afetado.sum()
            linha["%% alarmes %s" % rotulo] = 100 * (s[marca] > 1).mean() if marca.any() else np.nan
        if ref is not None and em_lacuna is not None and nome != referencia:
            comum = s.index.intersection(ref.index)
            comum = comum[em_lacuna.reindex(comum).fillna(False).astype(bool).to_numpy()]
            linha["|Δ phi/limiar| nas lacunas vs ref."] = (s[comum] - ref[comum]).abs().mean() if len(comum) else np.nan
        linhas.append(linha)
    return pd.DataFrame(linhas).set_index("cenário").round(3)


def plot_phi_comparacao(predicoes, threshold, ts, mask_sintetico=None, mask_outlier=None, freq="5min",
                        inicio=None, fim=None, max_faixas=300):
    """phi/limiar de cada cenário (máximo por `freq`), com as faixas de lacuna e de outlier."""
    ts = pd.to_datetime(pd.Series(ts)).reset_index(drop=True)
    fig = go.Figure()
    for i, (nome, predicao) in enumerate(predicoes.items()):
        s = _serie_phi(predicao, threshold)
        if inicio is not None or fim is not None:
            s = s[pd.Timestamp(inicio or s.index.min()):pd.Timestamp(fim or s.index.max())]
        s = s.resample(freq).max()
        fig.add_trace(go.Scattergl(x=s.index, y=s.values, name=nome, connectgaps=False,
                                   line=dict(color=_cor(nome, i), width=1.2)))
    fig.add_hline(y=1, line_color=COR_LIMIAR, line_dash="dash", annotation_text="Limiar")
    for mascara, cor in ((mask_sintetico, COR_NAN), (mask_outlier, COR_OUTLIER)):
        if mascara is None:
            continue
        for ini, fim_ in _corridas(np.asarray(mascara, dtype=bool).any(axis=1))[:max_faixas]:
            if (inicio is not None and ts.iloc[fim_ - 1] < pd.Timestamp(inicio)) or (fim is not None and ts.iloc[ini] > pd.Timestamp(fim)):
                continue
            fig.add_vrect(x0=ts.iloc[ini], x1=ts.iloc[fim_ - 1], fillcolor=cor, opacity=0.15, line_width=0)
    fig = _layout(fig, "Índice de anomalia normalizado (phi/limiar) por cenário — cinza: NaN sintético, vermelho: outlier", 500)
    fig.update_yaxes(range=[0, 4])
    return fig
