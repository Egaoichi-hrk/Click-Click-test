import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from PIL import Image
from streamlit_extras.stylable_container import stylable_container
import yfinance as yf
from pandas_datareader import data
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#実行させるにはstreamlit run C:\ウェブ企業\開発コード\WUM\pages\株価分析 個別銘柄.py
image = Image.open("Click Click LOGO 1.jpg")
st.set_page_config(
    page_title="株価分析",
    page_icon=image,
)
#######
HIDE_ST_STYLE = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
				        .appview-container .main .block-container{
                            padding-top: 1rem;
                            padding-right: 3rem;
                            padding-left: 3rem;
                            padding-bottom: 1rem;
                        }  
                        .reportview-container {
                            padding-top: 0rem;
                            padding-right: 3rem;
                            padding-left: 3rem;
                            padding-bottom: 0rem;
                        }
                        header[data-testid="stHeader"] {
                            z-index: -1;
                        }
                        div[data-testid="stToolbar"] {
                        z-index: 100;
                        }
                        div[data-testid="stDecoration"] {
                        z-index: 100;
                        }
                </style>
"""

st.markdown(HIDE_ST_STYLE, unsafe_allow_html=True)
########
st.markdown(
    """
    <div class = "head-back">
     <div class = "head-wrapper">
      <div class = "head1-text"> Click Click</div>
     </div>
    </div>
    <style>

      .head1-text{
         font-weight: 1000;
         text-align:center;
         font-size: 40px;
         font-family: 'Sacramento', cursive;
      }
    </style>
    <br><br>
    """
    , unsafe_allow_html=True
)
########
st.markdown(
    """
    <br><br><br>
    """
    , unsafe_allow_html=True)
######
st.subheader('株価分析')
#######
st.markdown(
    """
    <br>
    """
    , unsafe_allow_html=True)
########
with st.form(key='profile_form1'):
    st.subheader('個別銘柄の株価分析')
    kinds = st.text_input('日本の個別銘柄を指定（証券コード）', key='kinds_input_1')
    st.link_button('コードの参考', "https://quote.jpx.co.jp/jpx/template/quote.cgi?F=tmp/stock_search")
    st.text("表示するデータの範囲を決める")
    start = st.text_input('始まり(0000-00-00(年-月-日))', key='start_date_2')
    end = st.text_input('終わり(0000-00-00(年-月-日))', key='end_date_2')
    name = st.selectbox('表示させたい値', ('Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'),
                            key='value_selector_2')
    bunseki = st.selectbox('分析', (
    'ゴールデンクロスの検出', 'ARIMA', 'ランダムフォレストモデル', '指数平滑移動平均線（EMA）','ストキャスティクス','ボリンジャーバンド','相対力指数（RSI）'),
                               key='analysis_selector_1')
    btn = st.form_submit_button('実行')  # 修正したキー
    if btn:
        if len(start) == 0 or len(end) == 0:
            st.write("範囲が入力されていません。")
        else:
            if kinds:
                try:
                    # データを取得
                    df = data.DataReader(kinds, 'stooq', start=start, end=end)
                    st.write(df[[name]])  # ユーザーが選んだ値を表示
                    if bunseki == 'ゴールデンクロスの検出':
                        # データがSeriesなので、DataFrameに変換
                        data = df[[name]].copy()

                        # 移動平均線の計算
                        short_window = 50  # 短期移動平均線 (50日)
                        long_window = 200  # 長期移動平均線 (200日)

                        data['short_mavg'] = data[name].rolling(window=short_window, min_periods=1).mean()
                        data['long_mavg'] = data[name].rolling(window=long_window, min_periods=1).mean()

                        # ゴールデンクロスを確認する条件
                        data['signal'] = 0.0
                        data['signal'][short_window:] = np.where(
                            data['short_mavg'][short_window:] > data['long_mavg'][short_window:], 1.0, 0.0)
                        data['positions'] = data['signal'].diff()

                        # ゴールデンクロスのポイントをプロット
                        plt.figure(figsize=(10, 5))
                        plt.plot(data[name], label=f'{name} Price', alpha=0.5)
                        plt.plot(data['short_mavg'], label='50-Day Moving Average', alpha=0.85)
                        plt.plot(data['long_mavg'], label='200-Day Moving Average', alpha=0.85)

                        # ゴールデンクロスに対応する箇所をプロット
                        plt.plot(data[data['positions'] == 1.0].index,
                                 data['short_mavg'][data['positions'] == 1.0],
                                 '^', markersize=10, color='g', lw=0, label='Golden Cross')
                        plt.title(f'{kinds}　Golden Crosses')
                        plt.legend()
                        st.pyplot(plt)
                    elif bunseki == 'ARIMA':
                          price = df[[name]].copy()
                          model = ARIMA(price, order=(5, 1, 0))
                          model_fit = model.fit()

                          steps = len(price) // 2
                          forecast = model_fit.get_forecast(steps=steps)
                          forecast_index = pd.date_range(start=price.index[-1], periods=steps + 1, freq='B')[1:]
                          forecast_values = forecast.predicted_mean

                          # 結果をプロット
                          plt.figure(figsize=(10, 5))
                          plt.plot(price.index, price, label='Actual')
                          plt.plot(forecast_index, forecast_values, label='Forecast', color='red')
                          plt.title("ARIMA予測")
                          plt.xlabel("日付")
                          plt.ylabel(name)
                          plt.legend()
                          st.pyplot(plt)
                    elif bunseki == 'ランダムフォレストモデル':
                         price = df[[name]].copy()  # priceの定義を追加
                         scaler = StandardScaler()
                         scaled_price = scaler.fit_transform(price.values.reshape(-1, 1))

                         window_size = 10
                         X = []
                         y = []
                         for i in range(len(scaled_price) - window_size):
                           X.append(scaled_price[i:i + window_size])
                           y.append(scaled_price[i + window_size])

                         X, y = np.array(X), np.array(y)

                         train_size = int(len(X) * 0.8)
                         X_train, X_test = X[:train_size], X[train_size:]
                         y_train, y_test = y[:train_size], y[train_size:]
                       # ランダムフォレストモデルの構築
                         model = RandomForestRegressor(n_estimators=100)
                         model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
                        # 予測
                         predictions = model.predict(X_test.reshape(X_test.shape[0], -1))
                         predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
                         y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
                         plt.figure(figsize=(10, 5))
                         plt.plot(np.arange(len(y_test)), y_test, label='Actual')
                         plt.plot(np.arange(len(predictions)), predictions, label='Predicted')
                         plt.legend()
                         st.pyplot(plt)
                    elif bunseki == '指数平滑移動平均線（EMA）':
                        def calculate_ema(data, span):
                            """
                            指数平滑移動平均 (EMA) を計算する関数。

                            Parameters:
                            - data: 計算対象となるデータのリストまたはPandas Series
                            - span: EMAの期間（例: 12, 26など)
                            Returns:
                            - ema: 計算されたEMAの値を含むPandas Series
                            """
                            ema = data.ewm(span=span, adjust=False).mean()
                            return ema


                        price = df[[name]].copy()

                        # EMAの計算
                        price['EMA_12'] = calculate_ema(price[name], span=12)

                        # グラフの描画
                        plt.figure(figsize=(10, 6))
                        plt.plot(price[name], label='株価', color='blue', marker='o')  # 株価の線
                        plt.plot(price['EMA_12'], label='EMA (12期間)', color='red', linestyle='--')  # EMAの線
                        plt.title(f'{name}とEMA (12期間)')
                        plt.xlabel('日数')
                        plt.ylabel('価格')
                        plt.legend()
                        plt.grid(True)

                        st.pyplot(plt)
                    elif bunseki == 'ストキャスティクス':
                        def calculate_stochastic(df, k_window=14, d_window=3):
                            df['L14'] = df['Low'].rolling(window=k_window).min()  # 過去14日の安値
                            df['H14'] = df['High'].rolling(window=k_window).max()  # 過去14日の高値
                            df['%K'] = (df['Close'] - df['L14']) / (df['H14'] - df['L14']) * 100  # %Kの計算
                            df['%D'] = df['%K'].rolling(window=d_window).mean()  # %Dの計算（%Kの3期間移動平均）
                            return df


                        price = df.copy()
                        # ストキャスティクスを計算
                        df = calculate_stochastic(price)

                        # グラフの表示
                        st.write(f"{kinds}とストキャスティクス")
                        plt.figure(figsize=(10, 5))

                        # 株価のグラフ
                        plt.subplot(2, 1, 1)
                        plt.plot(df['Close'], label=f'{kinds}', color='blue')
                        plt.title(f'{kinds}')

                        # ストキャスティクスのグラフ
                        plt.subplot(2, 1, 2)
                        plt.plot(df.index, df['%K'], label='%K', color='green')
                        plt.plot(df.index, df['%D'], label='%D', color='red')
                        plt.axhline(80, color='grey', linestyle='--')  # 80%ライン（買われ過ぎ）
                        plt.axhline(20, color='grey', linestyle='--')  # 20%ライン（売られ過ぎ）
                        plt.title('ストキャスティクス')
                        plt.legend()

                        st.pyplot(plt)
                    elif bunseki == 'ボリンジャーバンド':
                        data = df.copy()
                        # 2. 移動平均と標準偏差を計算
                        window = 20  # 移動平均の期間
                        data['SMA'] = data['Close'].rolling(window=window).mean()
                        data['STD'] = data['Close'].rolling(window=window).std()
                        # 3. ボリンジャーバンドの上限と下限を計算
                        data['Upper Band'] = data['SMA'] + (data['STD'] * 2)
                        data['Lower Band'] = data['SMA'] - (data['STD'] * 2)
                        # 4. データをプロット
                        plt.figure(figsize=(10, 6))
                        plt.plot(data['Close'], label='Close Price', color='blue')
                        plt.plot(data['SMA'], label='Simple Moving Average (20 days)', color='orange')
                        plt.plot(data['Upper Band'], label='Upper Bollinger Band', color='green')
                        plt.plot(data['Lower Band'], label='Lower Bollinger Band', color='red')
                        plt.fill_between(data.index, data['Lower Band'], data['Upper Band'], color='gray', alpha=0.1)
                        plt.title(f'Bollinger Bands Analysis for {kinds}')
                        plt.xlabel('Date')
                        plt.ylabel('Price')
                        plt.legend(loc='best')
                        st.pyplot(plt)
                    elif bunseki == '相対力指数（RSI）':
                        data = df.copy()

                        # 2. RSIを計算する関数
                        def calculate_RSI(data, period=14):
                            delta = data['Close'].diff(1)  # 終値の変化量
                            gain = np.where(delta > 0, delta, 0)  # 上昇分のみ
                            loss = np.where(delta < 0, -delta, 0)  # 下落分のみ

                            avg_gain = pd.Series(gain).rolling(window=period).mean()  # 平均の上昇
                            avg_loss = pd.Series(loss).rolling(window=period).mean()  # 平均の下落

                            rs = np.where(avg_loss == 0, 0, avg_gain / avg_loss)

                            rsi = 100 - (100 / (1 + rs))  # RSIの計算
                            return rsi
                        # RSIを計算（デフォルトは14日間）
                        data['RSI'] = calculate_RSI(data)
                        st.write(data['RSI'])
                        # 3. データをプロット
                        plt.figure(figsize=(10, 6))
                        # 終値のプロット
                        plt.subplot(2, 1, 1)
                        plt.plot(data['Close'], label='Close Price', color='blue')
                        plt.title(f'{kinds} Close Price')
                        plt.ylabel('Price')
                        plt.legend(loc='best')
                        # RSIのプロット
                        plt.subplot(2, 1, 2)
                        plt.plot(data['RSI'], label='RSI (14)', color='orange')
                        plt.axhline(70, color='red', linestyle='--')  # 買われすぎのライン
                        plt.axhline(30, color='green', linestyle='--')  # 売られすぎのライン
                        plt.title('RSI (Relative Strength Index)')
                        plt.ylabel('RSI Value')
                        plt.legend(loc='best')
                        plt.tight_layout()
                        st.pyplot(plt)
                except Exception as e:
                    st.write(f"データの取得中にエラーが発生しました: {str(e)}")
########
st.markdown(
    """
    <br><br><br>
    """
,unsafe_allow_html=True)
st.write('・csvの読み込みを行う際はあらかじめ、Excelなどを用いて、データの加工をしてもらう必要があります。')
st.write('・不具合等ございましたらお手数ですがTOPページ内のお問い合わせフォームにてお願いします。')

st.markdown(
    """
    <br><br><br>
    """
,unsafe_allow_html=True)
st.link_button("TOP","")
