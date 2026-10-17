import os
from datetime import datetime

import joblib
import pandas as pd

DIR_MODELS = os.getcwd()+"/models/"


def _iso_z(value):
    """RFC3339 UTC com milissegundos e sufixo Z — mesmo formato do operation_log."""
    return pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _sorteia_lacunas(n_linhas, passo_segundos, gap_min_minutes, gap_max_minutes, rng):
    """Lacunas contiguas dentro do holdout, como [(inicio, fim_exclusivo), ...].

    Mascarar pontos isolados superestima a qualidade: o padrao real de falha e um trecho
    contiguo. As lacunas cobrem ~20% do holdout e nunca encostam na primeira linha.
    """
    minimo = max(1, int(round(gap_min_minutes * 60.0 / passo_segundos)))
    maximo = max(minimo, int(round(gap_max_minutes * 60.0 / passo_segundos)))
    alvo = max(minimo, int(n_linhas * 0.2))
    lacunas = []
    coberto = 0
    tentativas = 0
    while coberto < alvo and tentativas < 200:
        tentativas += 1
        tamanho = int(rng.integers(minimo, maximo + 1))
        if tamanho >= n_linhas - 1:
            tamanho = max(1, n_linhas // 4)
        inicio = int(rng.integers(1, max(2, n_linhas - tamanho)))
        fim = min(n_linhas, inicio + tamanho)
        if any(inicio < b and a < fim for a, b in lacunas):
            continue
        lacunas.append((inicio, fim))
        coberto += fim - inicio
    if not lacunas:
        lacunas.append((1, min(n_linhas, 1 + minimo)))
    return lacunas


def _remove_estado_aleatorio(imputer):
    """Tira do imputer fitado os geradores aleatorios, que o runtime nao consegue ler.

    O pickle de RandomState/Generator nao e portavel entre numpy 2.x e 1.x: o 2.x grava a
    CLASSE do BitGenerator e o 1.26 do Lambda so aceita o NOME, falhando com "<class
    'numpy.random._mt19937.MT19937'> is not a known BitGenerator module". Nenhum dos dois
    atributos e usado no transform/predict - `random_state_` so entra com
    sample_posterior=True e `_feature_subsample_rng` so no fit do HGB - e o sklearn
    recria ambos num refit. Depois de remover, verifica o objeto inteiro: se uma versao
    nova do sklearn (ou um `estimator` customizado) trouxer outro gerador, o treino falha
    aqui, e nao em producao.
    """
    import io
    import pickle
    import numpy as np

    if hasattr(imputer, "random_state_"):
        del imputer.random_state_
    for triplet in getattr(imputer, "imputation_sequence_", []):
        if hasattr(triplet.estimator, "_feature_subsample_rng"):
            del triplet.estimator._feature_subsample_rng

    restantes = []
    tipos = (np.random.RandomState, np.random.Generator, np.random.BitGenerator)

    class Verificador(pickle.Pickler):
        def reducer_override(self, obj):
            if isinstance(obj, tipos):
                restantes.append(type(obj).__name__)
            return NotImplemented

    Verificador(io.BytesIO()).dump(imputer)
    if restantes:
        raise RuntimeError(
            "imputer ainda carrega gerador aleatorio (%s): o .pkl nao carregaria no "
            "runtime com numpy 1.x" % sorted(set(restantes))
        )


def train_imputer(
    df_train_imputer,
    timestamp,
    process_id,
    estimator=None,
    max_iter=5,
    tol=1e-3,
    holdout_frac=0.2,
    gap_min_minutes=10,
    gap_max_minutes=240,
    random_state=0,
):
    """Treina o IterativeImputer e devolve o bloco `imputer` no formato usado pelo deploy.

    Uma chamada faz tudo: fit, metrica de holdout, estatisticas robustas para a deteccao
    de outlier, gravacao do `.pkl` em DIR_MODELS e montagem do dict. Uso no notebook:

        imputer = train_imputer(df_train_imputer, timestamp, process_id)

    `df_train_imputer` e UM dataframe, recortado no periodo que voce escolher. O periodo
    pode diferir do df_train do modelo; o CONJUNTO DE COLUNAS nao pode — o deploy valida
    isso, porque em runtime o imputer e aplicado sobre as colunas do modelo.

    Sobre o periodo: quanto mais ele incluir trechos com anomalia, melhor o imputer
    reconstroi durante uma falha, mas mais ele tende a devolver um valor "anomalo
    plausivel". Recortar so operacao limpa da o oposto: reconstrucao enviesada para o
    normal, que pode suavizar uma anomalia em curso. O `holdout_mae` por variavel e o
    instrumento para calibrar isso entre treinos.

    Sobre o estimador: o padrao e HistGradientBoostingRegressor. Em dado real (6
    janelas de 2 meses de um processo industrial) ele reduziu o MAE de holdout entre 9% e 64% frente
    ao BayesianRidge em todas as janelas, inclusive com o equipamento parando no holdout - os
    processos tem regime liga/desliga e saturacoes que o linear nao captura. Contrapartida:
    arvore nao extrapola, entao o valor imputado fica limitado a faixa vista no treino.
    Outro motivo para nao recortar o periodo demais.
    """
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    if not isinstance(df_train_imputer, pd.DataFrame):
        raise TypeError(
            "train_imputer espera UM DataFrame; recebeu %s"
            % type(df_train_imputer).__name__
        )
    if timestamp not in df_train_imputer.columns:
        raise ValueError("coluna de timestamp '%s' ausente no df_train_imputer" % timestamp)

    df = df_train_imputer.sort_values(timestamp).reset_index(drop=True)
    # As colunas do df_train, nessa ordem. Derivado aqui de proposito: aceitar a lista por
    # parametro abriria espaco para passar algo fora de ordem ou desatualizado.
    columns = [column for column in df.columns if column != timestamp]
    if not columns:
        raise ValueError("df_train_imputer nao tem coluna de variavel")

    valores = df[columns].apply(pd.to_numeric, errors="coerce")
    tempos = pd.to_datetime(df[timestamp])
    passo = tempos.diff().dropna().dt.total_seconds().median() or 60.0

    if estimator is None:
        # max_leaf_nodes=15 (padrao 31) e max_iter=5 no IterativeImputer deram o mesmo erro
        # da configuracao cheia com .pkl ~3x menor: o runtime baixa o .pkl a cada ciclo.
        estimator = HistGradientBoostingRegressor(
            max_iter=50, max_leaf_nodes=15, random_state=random_state
        )

    def novo_imputer():
        # keep_empty_features=True e obrigatorio: garante que a saida do transform tenha
        # sempre a mesma contagem de colunas de `columns`, mesmo se alguma for toda-NaN.
        # Sem isso o alinhamento posicional quebra no runtime.
        return IterativeImputer(
            estimator=estimator,
            max_iter=max_iter,
            tol=tol,
            keep_empty_features=True,
            random_state=random_state,
        )

    # --- holdout temporal: recorte final, nunca aleatorio ---
    corte = int(len(valores) * (1.0 - holdout_frac))
    holdout_mae, holdout_r2 = {}, {}
    if corte > len(columns) + 1 and len(valores) - corte > 2:
        treino = valores.iloc[:corte]
        holdout = valores.iloc[corte:].reset_index(drop=True)
        medidor = novo_imputer()
        medidor.fit(treino.to_numpy(dtype=float, copy=True))

        rng = np.random.default_rng(random_state)
        # Mascara UMA coluna por vez, com as demais intactas. Mascarar todas de uma vez
        # media um cenario que nao acontece - varios sensores caindo no mesmo instante -
        # e o multivariado ficava sem nada de onde puxar, produzindo R2 negativo e um
        # MAE pessimista que nao servia para comparar variaveis entre si. Uma falha real
        # e um sensor de cada vez, e e isso que o numero precisa refletir.
        for indice, column in enumerate(columns):
            lacunas = _sorteia_lacunas(
                len(holdout), passo, gap_min_minutes, gap_max_minutes, rng
            )
            mascarado = holdout.to_numpy(dtype=float, copy=True)
            trechos = []
            for inicio, fim in lacunas:
                reais = holdout.iloc[inicio:fim, indice].to_numpy(dtype=float)
                if np.isnan(reais).all():
                    continue
                trechos.append((inicio, fim, reais))
                mascarado[inicio:fim, indice] = np.nan
            if not trechos:
                continue

            reconstruido = medidor.transform(mascarado)
            verdadeiros, previstos = [], []
            for inicio, fim, reais in trechos:
                previsto = reconstruido[inicio:fim, indice]
                valido = ~np.isnan(reais)
                verdadeiros.extend(reais[valido].tolist())
                previstos.extend(previsto[valido].tolist())
            if not verdadeiros:
                continue
            verdadeiros = np.asarray(verdadeiros)
            previstos = np.asarray(previstos)
            holdout_mae[column] = float(np.mean(np.abs(verdadeiros - previstos)))
            variancia = float(np.sum((verdadeiros - verdadeiros.mean()) ** 2))
            residuo = float(np.sum((verdadeiros - previstos) ** 2))
            holdout_r2[column] = float(1.0 - residuo / variancia) if variancia > 0 else 0.0
    else:
        print("*** Poucas linhas para holdout: metricas nao calculadas ***")

    # --- artefato de producao: refit no periodo inteiro ---
    # Fitar duas vezes e de proposito. O holdout existe so para produzir a metrica; o
    # artefato que vai a producao precisa ter visto todo o periodo escolhido.
    imputer = novo_imputer()
    imputer.fit(valores.to_numpy(dtype=float, copy=True))
    _remove_estado_aleatorio(imputer)

    # --- estatisticas robustas: alimentam a deteccao de outlier no runtime ---
    robust_stats = {}
    for column in columns:
        serie = valores[column].dropna()
        if serie.empty:
            continue
        mediana = float(serie.median())
        robust_stats[column] = {
            "median": mediana,
            "mad": float((serie - mediana).abs().median()),
            "q1": float(serie.quantile(0.25)),
            "q3": float(serie.quantile(0.75)),
        }

    name_model = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_imputer.pkl"
    if not os.path.isdir(DIR_MODELS):
        os.makedirs(DIR_MODELS)
    # compress=3 corta ~45% do .pkl de arvores; o joblib.load do runtime le transparente.
    joblib.dump(imputer, DIR_MODELS + name_model, compress=3)

    import sklearn

    bloco = {
        "enabled": True,
        "name_model": name_model,
        "columns": columns,
        # Lidas do proprio ambiente: os campos de versao nunca ficam desatualizados a
        # mao, que e o modo de falha mais provavel deste bloco.
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "trained_at": _iso_z(datetime.utcnow()),
        "train_rows": int(len(valores)),
        "train_period_start": _iso_z(tempos.min()),
        "train_period_end": _iso_z(tempos.max()),
        "estimator": type(estimator).__name__,
        "max_iter": int(max_iter),
        "keep_empty_features": True,
        "holdout_mae": holdout_mae,
        "holdout_r2": holdout_r2,
        "robust_stats": robust_stats,
    }

    print("Imputer treinado: %s" % name_model)
    print("  processo .......: %s" % process_id)
    print("  periodo ........: %s -> %s" % (bloco["train_period_start"], bloco["train_period_end"]))
    print("  linhas / colunas: %s / %s" % (bloco["train_rows"], len(columns)))
    print("  sklearn / numpy : %s / %s" % (bloco["sklearn_version"], bloco["numpy_version"]))
    if holdout_mae:
        print("  qualidade por variavel (pior -> melhor):")
        for column, mae in sorted(holdout_mae.items(), key=lambda item: -item[1]):
            print("    %-30s MAE=%10.4f  R2=%7.4f" % (column, mae, holdout_r2.get(column, 0.0)))
        print("  MAE alto e sinal de que a variavel nao deveria ser imputada.")
    return bloco
