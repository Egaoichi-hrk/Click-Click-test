import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import japanize_matplotlib
from PIL import Image
from statsmodels.tsa.arima.model import ARIMA
from streamlit_extras.stylable_container import stylable_container


#実行させるには（streamlit run C:\ウェブ企業\開発コード\WUM\pages\機械学習.py

image = Image.open("Click Click LOGO 1.jpg")
st.set_page_config(
    page_title = "機械学習",
    page_icon = image,
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
st.subheader('機械学習')
#######
st.markdown(
    """
    <br>
    """
    , unsafe_allow_html=True)

######
st.subheader("回帰分析")
with st.form(key='profile_form'):
    uploaded_file = st.file_uploader("csvファイルのアップロード", type="csv")
    submit1_btn = st.form_submit_button('実行')
    dataframe = None
    selected_data = None

    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file, comment="#")
        if submit1_btn and dataframe is not None:
            st.write(dataframe)
            numeric_columns = dataframe.columns.tolist()
            name = st.selectbox('分析', ('単回帰分析', '重回帰分析'))
            selected_columns = st.multiselect("計算に使用する列を選択してください:", numeric_columns)
            st.text("※重回帰を行う場合は、被説明変数（ｙ）を1つ目に選択、説明変数（ｘ）を二つ目以降に選択")

            if selected_columns:
                selected_data = dataframe[selected_columns]
                st.write(selected_data)
                numeric_df = selected_data.select_dtypes(include=[float, int])
                value = numeric_df
#######
                if name == '単回帰分析':
                    st.title("単回帰分析の結果")

                    # データを作成
                    X = selected_data.iloc[:, 0].values.reshape(-1, 1)
                    y = selected_data.iloc[:, 1].values

                    # 回帰モデルを作成
                    model = LinearRegression()
                    model.fit(X, y)

                    # 回帰係数と切片
                    slope = model.coef_[0]
                    intercept = model.intercept_

                    st.write(f"回帰係数: {slope}")
                    st.write(f"切片: {intercept}")

                    # 予測
                    y_pred = model.predict(X)

                    # 結果をプロット
                    plt.figure(figsize=(10, 6))
                    plt.scatter(X, y, color='blue', label='データポイント')
                    plt.plot(X, y_pred, color='red', label='回帰直線')
                    plt.xlabel('X')
                    plt.ylabel('y')
                    plt.legend()
                    plt.title("回帰分析のプロット")

                    st.pyplot(plt)
                elif name =='重回帰分析':

                    st.title("重回帰分析の結果")


                    st.text("1つ目の変数を被説明変数（ｙ）、二つ目以降の変数を説明変数（ｘ）")
                    # データを作成 (複数の説明変数)
                    y = selected_data.iloc[:, 0].values  # 1つ目の列を被説明変数（y）として使用
                    X = selected_data.iloc[:, 1:].values  # 2つ目以降の列を説明変数（X）として使用

                    # 回帰モデルを作成
                    model = LinearRegression()
                    model.fit(X, y)

                    # 回帰係数と切片
                    coefficients = model.coef_
                    intercept = model.intercept_

                    st.write(f"回帰係数: {coefficients}")
                    st.write(f"切片: {intercept}")

                    # 予測
                    y_pred = model.predict(X)
                    st.write(f"予測値: {y_pred}")

                    plt.figure(figsize=(10, 6))
                    plt.scatter(y, y_pred, color='blue')
                    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)  # 予測値と実測値が完全に一致する直線
                    plt.xlabel('実測値 (y)')
                    plt.ylabel('予測値 (y_pred)')
                    plt.title('実測値 vs. 予測値')
                    st.pyplot(plt)

st.subheader("その他分析")
with st.form(key='profile_form1'):
    uploaded_file = st.file_uploader("csvファイルのアップロード", type="csv")
    submit1_btn = st.form_submit_button('実行')
    dataframe = None
    selected_data = None

    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file, comment="#")
        if submit1_btn and dataframe is not None:
            st.write(dataframe)
            numeric_columns = dataframe.columns.tolist()
            name = st.selectbox('分析', ( 'ロジスティック回帰', '決定木分析','階層型クラスタリング', 'K-平均クラスタリング'))
            selected_columns = st.multiselect("計算に使用する列を選択してください:", numeric_columns)
            st.text("※被説明変数（ｙ）を1つ目に選択、特徴量（ｘ）を二つ目以降に選択")
            if selected_columns:
                selected_data = dataframe[selected_columns]
                st.write(selected_data)
                numeric_df = selected_data.select_dtypes(include=[float, int])
                value = numeric_df
                if name == 'ロジスティック回帰':
                    # 特徴量 2つ目以降の列を説明変数（X）として使用
                    X = selected_data.iloc[:, 1:].values
                    # ターゲット (0 または 1 の二値分類) 1つ目の列を被説明変数（y）として使用
                    y = selected_data.iloc[:, 0].values

                    # データをトレーニングセットとテストセットに分割
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    # ロジスティック回帰モデルのインスタンスを作成
                    model = LogisticRegression()

                    # モデルを学習させる
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # モデルの精度を計算
                    accuracy = accuracy_score(y_test, y_pred)
                    print(f'Accuracy: {accuracy:.2f}')

                    # 混同行列の表示
                    cm = confusion_matrix(y_test, y_pred)
                    print('Confusion Matrix:')
                    print(cm)

                    # 分類レポートの表示
                    report = classification_report(y_test, y_pred)
                    print('Classification Report:')
                    print(report)



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

