import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import os
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import time

from arima_model import forecast_arima
BASE_URL = "https://forecast-api-production-7d02.up.railway.app"

app = Flask(__name__)
os.makedirs("static", exist_ok=True)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.get_json()
    series = data.get('series', [])

    if not isinstance(series, list) or len(series) == 0:
        return jsonify({"forecast": []})

    try:
        df = pd.DataFrame(series)

        required_cols = {'tahun', 'triwulan', 'total_luas'}
        if not required_cols.issubset(df.columns):
            return jsonify({"forecast": []})

        # ======================
        # CLEANING
        # ======================
        map_tri = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}
        df['triwulan'] = df['triwulan'].replace(map_tri)

        df['tahun'] = pd.to_numeric(df['tahun'], errors='coerce')
        df['triwulan'] = pd.to_numeric(df['triwulan'], errors='coerce')
        df['total_luas'] = pd.to_numeric(df['total_luas'], errors='coerce')

        df = df.dropna()
        df = df.drop_duplicates(subset=['tahun', 'triwulan'])
        df = df.sort_values(by=['tahun', 'triwulan'])

        # ======================
        # PERIOD INDEX
        # ======================
        df['periode'] = df['tahun'].astype(int).astype(str) + 'Q' + df['triwulan'].astype(int).astype(str)
        df['periode'] = pd.PeriodIndex(df['periode'], freq='Q')

        df = df.set_index('periode')
        series_ts = df['total_luas']

        # ======================
        # ACF PACF
        # ======================
        timestamp = str(int(time.time()))

        lags_acf = max(1, min(8, len(series_ts) - 1))
        lags_pacf = max(1, min(5, max(1, len(series_ts)//2 - 1)))

        if len(series_ts) > 2:
            plt.figure()
            plot_acf(series_ts, lags=lags_acf)
            acf_path = f"static/acf_{timestamp}.png"
            plt.savefig(acf_path)
            plt.close()

            plt.figure()
            plot_pacf(series_ts, lags=lags_pacf, method='ywm')
            pacf_path = f"static/pacf_{timestamp}.png"
            plt.savefig(pacf_path)
            plt.close()
        else:
            acf_path = "static/acf.png"
            pacf_path = "static/pacf.png"

        # ======================
        # FORECAST (ARIMA ONLY)
        # ======================
        if len(series_ts) < 6:
            last = float(series_ts.iloc[-1])
            forecast_values = [last] * 8
            order = [0, 1, 0]
            aic = None
            d_manual = 0
        else:
            result = forecast_arima(series_ts)

            forecast_values = result.get("forecast", [])
            order = result.get("order", [0,0,0])
            aic = result.get("aic", None)
            d_manual = result.get("d_manual", None)

        # ======================
        # RETURN (NO SEASONAL)
        # ======================
        return jsonify({
            "forecast": list(map(float, forecast_values)),
            "order": order,
            "aic": aic,
            "d_manual": d_manual,
            "acf_img": f"{BASE_URL}/{acf_path}",
            "pacf_img": f"{BASE_URL}/{pacf_path}"
        })

    except Exception as e:
        print("ERROR:", e)

        return jsonify({
            "forecast": [],
            "order": [0,0,0],
            "aic": None,
            "d_manual": None,
            "acf_img": "",
            "pacf_img": ""
        })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
