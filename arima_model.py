import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller


def adf_test(series, title=''):
    print(f'\n=== UJI ADF {title} ===')

    if len(series) < 3:
        print("⚠️ Data terlalu sedikit untuk ADF")
        return 1.0

    result = adfuller(series)

    print(f'Statistik ADF : {result[0]}')
    print(f'p-value       : {result[1]}')
    print(f'Lags          : {result[2]}')
    print(f'Observasi     : {result[3]}')

    if result[1] <= 0.05:
        print("✅ Data STASIONER")
    else:
        print("❌ Data TIDAK STASIONER")

    return result[1]


def make_stationary(series):
    d = 0
    temp_series = series.copy()

    while True:
        p_value = adf_test(temp_series, f"(d={d})")

        if p_value <= 0.05:
            break

        temp_series = temp_series.diff().dropna()
        d += 1

        print(f"➡️ Differencing ke-{d}")

        if d == 3:
            break

    return temp_series, d


def forecast_arima(series, steps=8):

    print("=== DATA AWAL ===")
    print(series)
    print("Jumlah data:", len(series))

    # ==========================
    # PASTIKAN INDEX PERIOD
    # ==========================
    if not isinstance(series.index, pd.PeriodIndex):
        series.index = pd.PeriodIndex(series.index, freq='Q')

    series = series.astype(float)

    # ==========================
    # UJI ADF → DAPATKAN d
    # ==========================
    _, d_manual = make_stationary(series)

    print(f"\n📌 Nilai differencing optimal (ADF): d = {d_manual}")

    # ==========================
    # HANDLE DATA PENDEK
    # ==========================
    if len(series) < 6:
        print("⚠️ Data terlalu sedikit, pakai fallback")

        last = series.iloc[-1]
        forecast = [last for _ in range(steps)]
        order = (0, d_manual, 0)
        aic = None

    else:
        try:
            print("\n=== PROSES AUTO ARIMA (PURE ARIMA) ===")

            model = auto_arima(
                series,
                d=d_manual,          # dari ADF
                seasonal=False,      # 🔥 WAJIB NON-SEASONAL
                start_p=0,
                start_q=0,
                max_p=5,
                max_q=5,
                trace=True,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore'
            )

            forecast = model.predict(n_periods=steps)

            order = model.order
            aic = model.aic()

            print("\n=== HASIL MODEL TERBAIK ===")
            print("✅ ORDER ARIMA :", order)
            print("✅ AIC TERBAIK :", aic)

        except Exception as e:
            print("❌ ERROR ARIMA:", e)

            forecast = [series.iloc[-1] for _ in range(steps)]
            order = (0, d_manual, 0)
            aic = None

    # ==========================
    # LABEL WAKTU
    # ==========================
    last_period = series.index[-1]

    future_labels = []
    next_period = last_period

    for _ in range(steps):
        next_period = next_period + 1
        future_labels.append(f"{next_period.year} Q{next_period.quarter}")

    # ==========================
    # RETURN
    # ==========================
    return {
        "labels": future_labels,
        "forecast": list(forecast),
        "order": order,
        "aic": aic,
        "d_manual": d_manual
    }