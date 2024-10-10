import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import scipy.stats as stats
import japanize_matplotlib

#実行させるには（streamlit run C:\ウェブ企業\開発コード\WUM\pages\統計分析1.py）

image = Image.open("Click Click LOGO 1.jpg")
st.set_page_config(
    page_title = "統計分析",
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
##########
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
st.subheader('統計分析１')
########
st.markdown(
    """
    <br>
    """
,unsafe_allow_html=True)

###########

with st.form(key='profile_form1'):

    st.subheader('標準化')
    mu = st.text_input('母平均 (μ)')
    sample_mean = st.text_input('標本平均')
    variance = st.text_input('分散')

    submit_btn = st.form_submit_button('入力した値で実行')

    if submit_btn:
        try:
            # 入力値を浮動小数点数に変換
            mu = float(mu)
            sample_mean = float(sample_mean)
            variance = float(variance)

            # 分散から標準偏差を計算
            std_dev = variance ** 0.5

            # 標準化の計算
            standardized_value = (sample_mean - mu) / std_dev

            # 結果を表示
            st.write(f"標準化された値: {standardized_value:.4f}")

        except ValueError:
            st.write("有効な数値を入力してください。")
######
st.markdown(
    """
     <br><br><br><br> <!-- スペースを追加 -->
    """,
     unsafe_allow_html=True
)


# csv読み込み時:
def standardize(data):
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std_dev = variance ** 0.5
    standardized_data = [(x - mean) / std_dev for x in data]
    return standardized_data





#######

with st.form(key='profile_form'):
    st.subheader('正規分布を用いた確率計算')
    uploaded_file = st.file_uploader("CSVファイルのアップロード", type="csv")
    submit1_btn = st.form_submit_button('実行')
    dataframe = None

    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file, comment="#")
        if submit1_btn and dataframe is not None:
            st.write(dataframe)
            numeric_columns = dataframe.select_dtypes(include=[float, int]).columns.tolist()
            selected_column = st.selectbox("計算に使用する列を一つ選択してください:", numeric_columns)
            if selected_column:
                selected_data = dataframe[selected_column]
                st.write("選択されたデータ:", selected_data)
                # ユーザーにどの計算を行うか選択してもらう
                war = st.selectbox('計算を選択:', ['a≦X≦bの確率', 'a≦X', 'a≧X'])

                # ユーザーから入力を取得
                a = st.text_input('a の値を入力:', value='0')
                b = st.text_input('b の値を入力:', value='0')  # デフォルト値を 0 にして、必ず値を持つようにする

                # 数値変換とバリデーション
                try:
                    a = float(a)
                    if war == 'a≦X≦bの確率':
                        b = float(b)
                except ValueError:
                    st.error("a と b は数値でなければなりません。")
                    st.stop()

                # 平均と標準偏差を計算
                mu = np.mean(selected_data)
                sigma = np.std(selected_data)
                x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)

                # 正規分布の確率密度関数 (PDF) を計算
                pdf = stats.norm.pdf(x, mu, sigma)

                # プロットを作成
                plt.figure(figsize=(8, 6))
                plt.plot(x, pdf, label=f'μ={mu}, σ={sigma}', color='blue')

                # タイトルとラベルを設定
                plt.title('正規分布', fontsize=20)
                plt.xlabel('X', fontsize=14)
                plt.ylabel('確率密度', fontsize=14)

                # グリッドを表示
                plt.grid(True)

                # 凡例を表示
                plt.legend()

                # プロットを表示
                st.pyplot(plt)

                # 確率計算
                distribution = stats.norm(mu, sigma)

                if war == 'a≦X':
                    # X <= a の確率を計算
                    probability_below_a = distribution.cdf(a)
                    # X >= a の確率を計算
                    probability_above_a = 1 - probability_below_a
                    # 結果を表示
                    st.write(f"{a} 以上の確率: {probability_above_a:.4f}")

                elif war == 'a≧X':
                    # X <= a の確率を計算
                    probability_below_a = distribution.cdf(a)
                    st.write(f"{a} 以下の確率: {probability_below_a:.4f}")

                elif war == 'a≦X≦bの確率':
                    # X <= b の確率を計算
                    probability_below_b = distribution.cdf(b)
                    # X < a の確率を計算
                    probability_below_a = distribution.cdf(a)
                    # X が a 以上 b 以下の確率を計算
                    probability_between_a_and_b = probability_below_b - probability_below_a
                    # 結果を表示
                    st.write(f"{a} 以上 {b} 以下の確率: {probability_between_a_and_b:.4f}")
#######

st.markdown(
    """
     <br><br><br><br> <!-- スペースを追加 -->
    """,
    unsafe_allow_html=True
)

#######

with st.form(key='profile_form2'):
    st.subheader('t検定')
    uploaded_file = st.file_uploader("CSVファイルのアップロード", type="csv")
    submit1_btn = st.form_submit_button('実行')
    dataframe = None

    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file, comment="#")
        if dataframe is not None:
            st.write(dataframe)
            numeric_columns = dataframe.select_dtypes(include=[float, int]).columns.tolist()
            selected_column = st.selectbox("計算に使用する列を一つ選択してください:", numeric_columns)

            if selected_column:
                selected_data = dataframe[selected_column]
                st.write("選択されたデータ:", selected_data)
                data_mean = np.mean(selected_data)
                data_var = np.var(selected_data)
                data_size = len(selected_data)
                st.write(f"平均: {data_mean}, 分散: {data_var}, サンプルサイズ: {data_size}")

                # 有意水準の選択
                per = st.selectbox('有意水準', ('片側1%', '片側5%', '両側1%', '両側5%'))
                alpha_dict = {'片側1%': (0.01, 'greater'), '片側5%': (0.05, 'greater'),
                              '両側1%': (0.01, 'two-sided'), '両側5%': (0.05, 'two-sided')}
                alpha, alternative = alpha_dict[per]

                # 検定の選択
                selection = st.selectbox('分析名', ('標本母平均検定', '対応差の検定', '母平均の差の検定'))

                # 標本母平均検定 (一標本t検定)
                if selection == '標本母平均検定':
                    data_mean1 = st.text_input('母平均 (数字を入力してください)', '')
                    if data_mean1:
                        try:
                            data_mean1 = float(data_mean1)
                            t_stat, p_value = stats.ttest_1samp(selected_data, popmean=data_mean1, alternative=alternative)
                            st.write(f"T値 = {t_stat}")
                            st.write(f'P値 = {p_value}')
                            if p_value < alpha:
                                st.write(f"帰無仮説を棄却します。(帰無仮説: μ = {data_mean1})")
                            else:
                                st.write(f"帰無仮説を棄却できません。(帰無仮説: μ = {data_mean1})")
                        except ValueError:
                            st.error("有効な数値を入力してください。")

                # 対応差の検定 または 母平均の差の検定
                elif selection in ['対応差の検定', '母平均の差の検定']:
                    selected_column1 = st.selectbox("追加で計算に使用する列を選択してください:", numeric_columns)
                    if selected_column1:
                        selected_data1 = dataframe[selected_column1]
                        st.write("選択されたデータ:", selected_data1)

                        try:
                            # 対応差の検定
                            if selection == '対応差の検定':
                                t_stat, p_value = stats.ttest_rel(selected_data, selected_data1, alternative=alternative)
                                st.write(f"T値 = {t_stat}")
                                st.write(f'P値 = {p_value}')

                            # 母平均の差の検定（等分散性を仮定しない）
                            elif selection == '母平均の差の検定':
                                t_stat, p_value = stats.ttest_ind(selected_data, selected_data1, equal_var=False, alternative=alternative)
                                st.write(f"T値 = {t_stat}")
                                st.write(f'P値 = {p_value}')

                            if p_value < alpha:
                                st.write("帰無仮説を棄却します。差がある。")
                            else:
                                st.write("帰無仮説を棄却できません。差はない。")
                        except ValueError:
                            st.error("有効な数値を入力してください。")


########
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
