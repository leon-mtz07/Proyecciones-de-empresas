from openbb import obb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import date
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from sklearn.model_selection import TimeSeriesSplit
import ta

# Creación de la función monte carlo que se utilizará para predecir el precio de una acción
def monte_carlo(ticker: str, fecha_f = date.today(), N = 100000, tiempo : int = 22):
    """
    Se usará el modelo de montecarlo para tratar de predecir el precio de la acción específicada en el ticker
    :param ticker:
    :param fecha_f:
    :param N:
    :return:
    """

    # Calculando la volatilidad anual
    stock = obb.equity.price.historical(ticker, start_date="2020-01-01", end_date=fecha_f, interval="1d",
                                        provider="yfinance")
    df_stock = stock.to_df()

    # Calcula log returns
    df_stock["logret"] = np.log(df_stock["close"] / df_stock["close"].shift(1))

    # Elimina filas con NaN
    df_stock = df_stock.dropna(subset=["logret"])

    # Volatilidad anualizada histórica total
    vol_diaria = df_stock["logret"].std()
    vol_anual = vol_diaria * np.sqrt(252)

    # Calculamos el rendimiento esperado
    # 1. Obtener precios históricos del S&P 500 (VOO)
    sp500 = obb.equity.price.historical("VOO", start_date="1900-01-01", end_date=fecha_f, provider="yfinance").to_df()

    # 2. Calcular rendimientos diarios logarítmicos
    df_stock["ret"] = np.log(df_stock["close"] / df_stock["close"].shift(1))
    sp500["ret"] = np.log(sp500["close"] / sp500["close"].shift(1))

    # 3. Alinear fechas
    data = df_stock[["ret"]].join(sp500[["ret"]], lsuffix="_stock", rsuffix="_mkt").dropna()

    # 4. Estimar beta mediante regresión lineal (CAPM)
    X = sm.add_constant(data["ret_mkt"])  # variable independiente (mercado)
    y = data["ret_stock"]  # variable dependiente (activo)

    model = sm.OLS(y, X).fit()
    alpha, beta = model.params

    # 5. Supongamos que la tasa libre de riesgo es 4% anual (~0.04/252 diario)
    rf_daily = 0.04 / 252

    # Rendimiento esperado del mercado (media diaria * 252)
    market_exp = data["ret_mkt"].mean() * 252

    # 6. Rendimiento esperado de la acción según CAPM
    expected_return = rf_daily * 252 + beta * (market_exp - rf_daily * 252)

    # Calculando el Monte Carlo
    T = tiempo / 252  # 22 días de trading

    # Simulación GBM (méto-do exacto para S_T):
    Z = np.random.normal(0, 1, size=N)
    ST = df_stock['close'].iloc[-1] * np.exp((expected_return - 0.5 * vol_anual ** 2) * T + vol_anual * np.sqrt(T) * Z)

    # Estadísticas
    mean_ST = ST.mean()
    median_ST = np.median(ST)
    p05, p95 = np.percentile(ST, [5, 95])

    # Resultados
    return mean_ST

# Creando un modelo de machine learning para predecir el precio de una acción
def random_forest(ticker: str, fecha_f: str = date.today(), tiempo: int = 22, model_type: str = "Regresor"):
    import warnings

    # Ignoramos warnings
    warnings.filterwarnings(action='ignore')

    # Descargar datos históricos
    df_stock = obb.equity.price.historical(
        ticker, start_date="2020-01-01", end_date=fecha_f, interval="1d", provider="yfinance").to_df()

    # Guardamos el precio actual real
    current_price = df_stock["close"].iloc[-1]
    print("------------------------")
    print(f"Acción {ticker}")
    print(f"Precio actual: {current_price}")
    print("------------------------")

    # Crear variables/features
    df_stock["Return"] = df_stock["close"].pct_change()
    df_stock["MA20"] = df_stock["close"].rolling(20).mean()
    df_stock["MA50"] = df_stock["close"].rolling(50).mean()
    df_stock["Volatility"] = df_stock["Return"].rolling(20).std()
    df_stock["RSI"] = ta.momentum.RSIIndicator(df_stock["close"], window=14).rsi()

    # Definir Target y Target_Price
    df_stock["Target"] = (df_stock["close"].shift(-tiempo) > df_stock["close"]).astype(int)
    df_stock["Target_Price"] = df_stock["close"].shift(-tiempo)

    # Crear features adicionales
    horizons = [5, 10, 20]
    new_predictors = ["Return", "MA20", "MA50", "Volatility", "RSI"]
    for horizon in horizons:
        rolling_avg = df_stock["close"].rolling(horizon).mean()
        df_stock[f"Close_Ratio_{horizon}"] = df_stock["close"] / rolling_avg
        df_stock[f"Trend_{horizon}"] = df_stock["Target"].shift(1).rolling(horizon).sum()
        new_predictors += [f"Close_Ratio_{horizon}", f"Trend_{horizon}"]

    # Guardamos las features actuales para predicción (precio real)
    X_current = df_stock[new_predictors].iloc[[-1]].copy()

    # Preparamos datos de entrenamiento quitando filas con NaN en Target_Price
    df_train = df_stock.dropna(subset=["Target_Price"])

    if model_type == "Clasificador":
        # Modelo de Random Forest Clasificador
        # Definir modelo
        model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

        # Función predict con umbral de probabilidad
        def predict(train, test, predictors, model, umbral = 0.5):
            model.fit(train[predictors], train["Target"])
            predictions = model.predict_proba(test[predictors])[:, 1]

            # Aplicar umbral
            predictions[predictions >= umbral] = 1
            predictions[predictions < umbral] = 0

            predictions = pd.Series(predictions, index=test.index, name="Predictions")
            combined = pd.concat([test["Target"], predictions], axis=1)
            return combined

        # Función de backtesting
        def backtest(data, model, predictors, start=250, step=250):
            all_predictions = []
            for i in range(start, data.shape[0], step):
                train = data.iloc[0:i].copy()
                test = data.iloc[i:(i + step)].copy()
                predictions = predict(train, test, predictors, model)
                all_predictions.append(predictions)
            return pd.concat(all_predictions)

        # Ejecutar backtest
        predictions = backtest(df_stock, model, new_predictors)

        # Evaluar resultados
        print("-------------------------------------------")
        print("Puntuación del Clasificador:")

        y_true = predictions["Target"]
        y_pred = predictions["Predictions"]

        print("Accuracy del backtest: ", accuracy_score(y_true, y_pred))
        print("Precision del backtest: ", precision_score(y_true, y_pred))
        print("Recall: ", recall_score(y_true, y_pred))
        print("F1 Score: ", f1_score(y_true, y_pred))
        print("Matriz de confusión:\n", confusion_matrix(y_true, y_pred))
        print("-------------------------------------------")

    elif model_type == "Regresor":
        X_train = df_train[new_predictors]
        y_train = df_train["Target_Price"]

        # Entrenar Random Forest Regressor
        rf_reg = RandomForestRegressor(n_estimators=500, min_samples_split=50, random_state=1)
        rf_reg.fit(X_train, y_train)

        # Predicción sobre la fila actual
        predicted_price = rf_reg.predict(X_current)[0]

        # Evaluación de error en test set
        mae_list, rmse_list = [], []
        tscv = TimeSeriesSplit(n_splits=5)

        for train_index, test_index in tscv.split(X_train):
            X_train_split, X_test_split = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_split, y_test_split = y_train.iloc[train_index], y_train.iloc[test_index]

            rf_reg.fit(X_train_split, y_train_split)
            y_pred_split = rf_reg.predict(X_test_split)

            mae_list.append(mean_absolute_error(y_test_split, y_pred_split))
            rmse_list.append(np.sqrt(mean_squared_error(y_test_split, y_pred_split)))

        # Intervalos del 5%-95% usando predicciones de todos los árboles
        all_tree_preds = np.array([tree.predict(X_current)[0] for tree in rf_reg.estimators_])
        p05, p95 = np.percentile(all_tree_preds, [5, 95])

        print("-------------------------------------------")
        print("Puntuación del Regressor:")
        print("MAE promedio:", np.mean(mae_list))
        print("RMSE promedio:", np.mean(rmse_list))
        print("-------------------------------------------")

        # Mostrar resultados
        return predicted_price
    return None


def predict_price(ticker: str, fecha_f: str = date.today(), tiempo: int = 22, model_type: str = "Regresor"):

    if model_type == "Regresor":

        # Obteniendo el precio de monte carlo
        precio_MN = monte_carlo(ticker, fecha_f, tiempo)

        # Obteniendo el precio con el modelo de Random Forest
        precio_RF = random_forest(ticker, fecha_f, tiempo, model_type=model_type)

        print(f"Precio de monte carlo: {precio_MN}")
        print(f"Precio del modelo de Random Forest: {precio_RF}")

    elif model_type == "Clasificador":

        # Obteniendo el precio con el modelo de Random Forest
        random_forest(ticker, fecha_f, tiempo, model_type=model_type)