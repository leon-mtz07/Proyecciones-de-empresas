from openbb import obb
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

def monte_carlo(ticker: str, fecha_f = date.today(), N = 100000):
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
    # 1. Obtener precios históricos del S&P 500 (SPXUIV)
    sp500 = obb.equity.price.historical("SPXUIV", start_date="2020-01-01", end_date="2025-09-30").to_df()

    # 2. Calcular rendimientos diarios logarítmicos
    df_stock["ret"] = np.log(df_stock["close"] / df_stock["close"].shift(1))
    sp500["ret"] = np.log(sp500["close"] / sp500["close"].shift(1))

    # 3. Alinear fechas
    data = df_stock[["ret"]].join(sp500[["ret"]], lsuffix="_aapl", rsuffix="_mkt").dropna()

    # 4. Estimar beta mediante regresión lineal (CAPM)
    X = sm.add_constant(data["ret_mkt"])  # variable independiente (mercado)
    y = data["ret_aapl"]  # variable dependiente (activo)

    model = sm.OLS(y, X).fit()
    alpha, beta = model.params

    # 5. Supongamos que la tasa libre de riesgo es 4% anual (~0.04/252 diario)
    rf_daily = 0.04 / 252

    # Rendimiento esperado del mercado (media diaria * 252)
    market_exp = data["ret_mkt"].mean() * 252

    # 6. Rendimiento esperado de AAPL según CAPM
    expected_return = rf_daily * 252 + beta * (market_exp - rf_daily * 252)

    # Calculando el Monte Carlo
    T = 22 / 252  # 22 días de trading

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