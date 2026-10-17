import pandas as pd
from datetime import timedelta


def select_periods_nan(dataset, timestamp):
    df = dataset.copy()
    df.set_index(timestamp, inplace=True)

    # Crie uma máscara booleana para identificar as linhas com NaN
    mask = df.isna().any(axis=1)

    # Obtenha os índices das linhas com NaN
    indices_com_nan = df.index[mask]

    # Crie a lista de dicionários
    lista_de_dicionarios = []
    for indice in indices_com_nan:
        linha_nan = df.loc[indice]
        colunas_com_nan = linha_nan[linha_nan.isna()].index.tolist()

        dicionario = {
            "date_ini": indice.strftime("%Y/%m/%d %H:%M:%S"),
            "date_end": indice.strftime("%Y/%m/%d %H:%M:%S"),
            "type": "tag_bad",
            "tags": colunas_com_nan,
        }
        lista_de_dicionarios.append(dicionario)

    df.reset_index(inplace=True)
    df.dropna(inplace=True)

    return lista_de_dicionarios


def drop_transitorio_desligado(
    dataset,
    variavel,
    limite,
    intervalo,
    timestamp,
    pre_corte=0,
    pos_corte=0,
    pp_residual=0,
):
    final = False
    df_aux = dataset.loc[:, [variavel, timestamp]].copy()
    df_aux["status"] = 1  # Ligado
    ############################   Descida  ########################################
    ## Se uma amostra está abaixo do limite e permance nesse nivel por
    ## por um intervalo, o status desse periodo é setado como 0, o que indicada
    ## que o motor está desligado
    ###############################################################################
    data_aux = df_aux[timestamp].min()  # Primeira data do dataframe
    list_periods = []
    while True:
        # Pega a data da primeira amostra com o valor abaixo do limite
        df_amostra = df_aux[
            (df_aux[variavel] <= limite) & (df_aux[timestamp] >= data_aux)
        ]

        if not df_amostra.empty:
            data_min = df_amostra[timestamp].min()
        else:
            break

        # Pega a primeira data da amostra acima do valor limite depois da amostra acima
        df_amostra = df_aux[
            (df_aux[variavel] > limite) & (df_aux[timestamp] > data_min)
        ]

        if not df_amostra.empty:
            data_aux = df_amostra[timestamp].min()
        else:
            data_aux = df_aux[timestamp].max()
            final = True

        # Tira a diferença entre as duas amostras,
        dif_date = (data_aux - data_min).total_seconds()
        dif_date = dif_date/60

        # Caso o valor da diferença seja maior que >= 3600s (1 hora) o motor está desligado
        if dif_date >= intervalo:
            list_periods.append(
                {"date_ini": data_min, "date_end": data_aux, "type": "desligado"}
            )
            mask = (df_aux[timestamp] >= data_min) & (df_aux[timestamp] <= data_aux)
            df_aux.loc[mask, "status"] = 0  # Status Desligado

            if pre_corte != 0:
                date_ini_pre_corte = data_min - timedelta(minutes=pre_corte)
                date_end_pre_corte = data_min

                mask_pre_corte = (df_aux[timestamp] >= date_ini_pre_corte) & (
                    df_aux[timestamp] <= date_end_pre_corte
                )
                df_aux.loc[mask_pre_corte, "status"] = 0  # Status Desligado
                list_periods.append(
                    {
                        "date_ini": date_ini_pre_corte,
                        "date_end": date_end_pre_corte,
                        "type": "transitorio",
                    }
                )

            if pos_corte != 0:
                date_ini_pos_corte = data_aux
                date_end_pos_corte = data_aux + timedelta(minutes=pos_corte)

                mask_pos_corte = (df_aux[timestamp] >= date_ini_pos_corte) & (
                    df_aux[timestamp] <= date_end_pos_corte
                )
                df_aux.loc[mask_pos_corte, "status"] = 0  # Status Desligado
                list_periods.append(
                    {
                        "date_ini": date_ini_pos_corte,
                        "date_end": date_end_pos_corte,
                        "type": "transitorio",
                    }
                )

        if final:
            break

    ############################   Subida  ########################################
    ## Essa parte do código serve para pegar picos onde o motor volta a "funcionar"
    ## por menos do intervalo, ou seja, ele estava desligado, deu um pique de menos
    ## de uma hora e voltou a ficar desligado
    ###############################################################################
    data_aux = df_aux[timestamp].min()
    while True:
        # Pega a primeira data da amostra com status ligado
        df_amostra = df_aux[(df_aux["status"] == 1) & (df_aux[timestamp] >= data_aux)]

        if not df_amostra.empty:
            data_min = df_amostra[timestamp].min()
        else:
            break

        # Pega a amostra com o status deligado após a data acima
        df_amostra = df_aux[(df_aux["status"] == 0) & (df_aux[timestamp] > data_min)]

        if not df_amostra.empty:
            data_aux = df_amostra[timestamp].min()
        else:
            break

        # Tira a diferença entre as duas amostras,
        dif_date = (data_aux - data_min).total_seconds()

        # Caso o valor da diferença seja maior que < 3600s (1 hora) o motor está desligado
        if dif_date < intervalo:
            list_periods.append(
                {"date_ini": data_min, "date_end": data_aux, "type": "desligado"}
            )
            mask = (df_aux[timestamp] >= data_min) & (df_aux[timestamp] <= data_aux)
            df_aux.loc[mask, "status"] = 0

    if pp_residual:
        df_aux.iloc[:pp_residual, df_aux.columns.get_loc("status")] = 0
    df_return = dataset.copy()
    df_return.drop(df_aux[df_aux["status"] == 0].index, inplace=True)
    return df_return, df_aux, list_periods


def set_tags_config(df_dataset, dir_file_tags):
    # importação do arquivo de susbsistema
    df_sistema = pd.read_csv(dir_file_tags, sep=";", decimal=".")
    df_sistema.drop_duplicates(subset=['VARIAVEL'],inplace=True)
    df_local = df_sistema.drop_duplicates(subset=["SISTEMA"])
    df_local = df_local.drop(["VARIAVEL", "DESC"], axis=1)
    df_local = df_local.reset_index(drop=True)

    df_sistema_drop = df_sistema[
        df_sistema["VARIAVEL"].isin(list(df_dataset.columns)[0:]) == False
    ]

    df_sistema_drop.reset_index(inplace=True, drop=True)

    df_sistema = df_sistema[
        df_sistema["VARIAVEL"].isin(list(df_dataset.columns)[0:]) == True
    ]

    df_sistema.reset_index(inplace=True, drop=True)

    df_aux = df_dataset[df_sistema["VARIAVEL"].to_list()]

    listIQR = []
    listQ3 = []
    for col in df_aux.columns:
        Q1 = df_aux[col].quantile(0.25)
        Q3 = df_aux[col].quantile(0.75)
        IQR = Q3 - Q1
        listIQR.append(IQR)
        listQ3.append(Q3)

    df_sistema["IQR"] = listIQR
    df_sistema["Q3"] = listQ3
    
    df_sistema.fillna('NoData',inplace=True)
    df_sistema_drop.fillna('NoData',inplace=True)

    return df_sistema, df_sistema_drop


def load_dataset_principal(
    dir_file, list_columns_drop, timestamp, dropna=False, use_chunks=False, chunksize=1000, sep=";", decimal=".", mask="%Y-%m-%d %H:%M:%S"
):
    """
    Carrega o dataset principal aceitando tanto .csv quanto .parquet.
    O formato é detectado pelo sufixo de 'dir_file'. Para .parquet, os parâmetros
    específicos de CSV (sep, decimal, use_chunks, chunksize, mask) são ignorados,
    pois o Parquet já traz o schema/tipos das colunas.
    """
    # ---- Parquet: colunas já tipadas; sep/decimal/chunks não se aplicam ----
    if str(dir_file).lower().endswith(".parquet"):
        df_aux = pd.read_parquet(dir_file)

        # Remover colunas especificadas
        df_aux = df_aux.drop(columns=list_columns_drop, errors="ignore")

        # Conversão da coluna de timestamp (pode vir como datetime ou string ISO)
        df_aux[timestamp] = pd.to_datetime(df_aux[timestamp])

        # Convertendo as demais colunas para numérico
        cols = list(df_aux.columns)
        cols.remove(timestamp)
        df_aux[cols] = df_aux[cols].apply(pd.to_numeric, errors='coerce')

        # Retirando valores NaN
        if dropna:
            df_aux = df_aux.dropna()

    elif use_chunks:
        # Importação do arquivo CSV em chunks
        chunks = pd.read_csv(dir_file, sep=sep, decimal=decimal, chunksize=chunksize) #engine="pyarrow"
        processed_chunks = []
        for chunk in chunks:
            # Remover colunas especificadas
            chunk = chunk.drop(columns=list_columns_drop, errors="ignore")
            
            # Conversão da coluna de timestamp para Datetime object
            chunk[timestamp] = pd.to_datetime(chunk[timestamp],format=mask)
            
            # Listando todas as colunas exceto timestamp
            cols = list(chunk.columns)
            cols.remove(timestamp)
            
            # Convertendo para numérico
            chunk[cols] = chunk[cols].apply(pd.to_numeric, errors='coerce')
            
            # Retirando valores NaN
            if dropna:
                chunk.dropna(inplace=True)
            
            processed_chunks.append(chunk)
        
        df_aux = pd.concat(processed_chunks)
    else:
        # Importação do arquivo CSV completo
        df_aux = pd.read_csv(dir_file, sep=sep, decimal=decimal, engine="pyarrow")
        
        # Remover colunas especificadas
        df_aux = df_aux.drop(columns=list_columns_drop, errors="ignore")

        # Conversão da coluna de timestamp para Datetime object
        df_aux[timestamp] = pd.to_datetime(df_aux[timestamp])

        # Listando todas as colunas exceto timestamp
        cols = list(df_aux.columns)
        cols.remove(timestamp)
        
        # Convertendo para numérico
        df_aux[cols] = df_aux[cols].apply(pd.to_numeric, errors='coerce')

        # Retirando valores NaN
        if dropna:
            df_aux = df_aux.dropna()


    df_aux.sort_values(by=timestamp, inplace=True)
    return df_aux


def merge_periods(periods):
    # Sort the periods by 'date_ini'
    periods.sort(key=lambda x: x["date_ini"])

    merged_periods = []

    for period in periods:
        if not merged_periods:
            merged_periods.append(period)
        else:
            last_period = merged_periods[-1]

            # If the current period overlaps with the last period
            if period["date_ini"] <= last_period["date_end"]:
                # If both periods are of type 'transitorio'
                if period["type"] == last_period["type"] == "transitorio":
                    # Extend the 'date_end' of the last period if necessary
                    last_period["date_end"] = max(
                        last_period["date_end"], period["date_end"]
                    )
                # If the last period is 'desligado' and the current period is 'transitorio'
                elif (
                    last_period["type"] == "desligado"
                    and period["type"] == "transitorio"
                ):
                    # Adjust the 'date_ini' of the current period
                    period["date_ini"] = last_period["date_end"]
                    merged_periods.append(period)
                # If the last period is 'transitorio' and the current period is 'desligado'
                elif (
                    last_period["type"] == "transitorio"
                    and period["type"] == "desligado"
                ):
                    # Adjust the 'date_end' of the last period
                    last_period["date_end"] = period["date_ini"]
                    merged_periods.append(period)
            else:
                merged_periods.append(period)

    return merged_periods
