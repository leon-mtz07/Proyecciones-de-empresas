from openbb import obb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
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
    stock = obb.equity.price.historical(ticker, start_date="1900-01-01", end_date=fecha_f, interval="1d",
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
    ST = df_stock['close'].iloc[-1] * np.exp((vol_anual - 0.5 * expected_return ** 2) * T + expected_return * np.sqrt(T) * Z)

    # Estadísticas
    mean_ST = ST.mean()
    median_ST = np.median(ST)
    p05, p95 = np.percentile(ST, [5, 95])

    # Resultados
    print(f"Acción: {ticker}")
    print(f"Precio Actual: {df_stock['close'].iloc[-1]}")
    print(f"Rendimiento E[S_T] probable {np.round((mean_ST / df_stock['close'].iloc[-1] - 1) * 100, 2)}%")
    print("E[S_T] ≈", mean_ST)
    print("Mediana ≈", median_ST)
    print("P5 ≈", p05, "  P95 ≈", p95)
    print(f"Rendimiento P5: {np.round((p05 / df_stock['close'].iloc[-1] - 1) * 100, 2)}%", f" P95: {np.round((p95 / df_stock['close'].iloc[-1] - 1) * 100, 2)}%")

    # Histograma
    plt.title("")
    sns.histplot(ST, bins=100, stat="density", alpha=0.7, color='green')
    plt.axvline(np.percentile(ST, 5), color='blue', linestyle='--', label="90% info")
    plt.axvline(mean_ST, color='k', linestyle='--', label=f"mean {mean_ST:.2f}")
    plt.axvline(median_ST, color='r', linestyle=':', label=f"median {median_ST:.2f}")
    plt.axvline(np.percentile(ST, 95), color='blue', linestyle='--')
    plt.legend()
    plt.xlabel("Precio S_T")
    plt.ylabel("Densidad")
    plt.title(f"Distribución simulada de S_T (1 mes) de la empresa {ticker}")
    plt.show()

# Creando un modelo de machine learning para predecir el precio de una acción
def predict_forest(ticker: str, fecha_f: str = date.today(), tiempo: int = 22, model_type: str = "Regresor"):
    import warnings

    # Ignoramos warnings
    warnings.filterwarnings(action='ignore')

    # Descargar datos históricos
    df_stock = obb.equity.price.historical(
        ticker, start_date="2020-01-01", end_date=fecha_f, interval="1d", provider="yfinance").to_df()

    # Guardamos el precio actual real
    current_price = df_stock["close"].iloc[-1]
    print(f"Precio actual: {current_price}")

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
        accuracy = (predictions["Target"] == predictions["Predictions"]).mean()
        print("Accuracy del backtest:", accuracy)
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
        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
            X_train, y_train, test_size=0.2, shuffle=False
        )
        y_pred_split = rf_reg.predict(X_test_split)
        mae = mean_absolute_error(y_test_split, y_pred_split)
        rmse = np.sqrt(mean_squared_error(y_test_split, y_pred_split))

        # Intervalos del 5%-95% usando predicciones de todos los árboles
        all_tree_preds = np.array([tree.predict(X_current)[0] for tree in rf_reg.estimators_])
        p05, p95 = np.percentile(all_tree_preds, [5, 95])

        # Mostrar resultados
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"Precio actual usado para predicción: {current_price:.2f}")
        print(f"Precio probable a {tiempo} días: {predicted_price:.2f}")
        print(f"Rango probable de precio: ${p05:.2f} - ${p95:.2f}")