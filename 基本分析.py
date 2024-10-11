import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from streamlit_extras.stylable_container import stylable_container



#######

st.header('Click Click')

######
st.subheader('基本分析')
########

with st.form(key='profile_form1'):
    value = st.text_input('値(スペースで区切って入力してください)')
    name = st.selectbox('分析', ('平均', '分散','標準偏差'))
    submit_btn = st.form_submit_button('入力した値で実行')
    if value:
        try:
            value_list = value.split()
            value = [float(number) for number in value_list]
        except ValueError:
            st.write("有効な数値を入力してください。")
    if  submit_btn:
        if len(value) == 0:
            st.write("値が入力されていません。")
        else:
            if name == ('平均'):

                value_mean = np.mean(value)
                st.text(f'平均＝{value_mean}')

            elif name == ('分散'):

                value_var = np.var(value)
                st.text(f'分散＝{value_var}')
            else:

                value_std = np.std(value)
                st.text(f'標準偏差＝{value_std}')



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
            selected_columns = st.multiselect("計算に使用する列を選択してください:", numeric_columns)
            name_plot = st.selectbox('dataframeを図として可視化', ('折れ線グラフ', '棒グラフ', '散布図', 'ヒストグラム'))
            name = st.selectbox('分析', ('平均', '分散', '共分散', '標準偏差','相関係数'))

            if selected_columns:
                selected_data = dataframe[selected_columns]
                st.write(selected_data)
                numeric_df = selected_data.select_dtypes(include=[float, int])
                value = numeric_df

                fig, ax = plt.subplots()

                if name_plot == '折れ線グラフ':
                    ax.plot(value)
                elif name_plot == '棒グラフ':
                    for column in value.columns:
                        ax.bar(value.index, value[column], label=column)
                    ax.legend()
                elif name_plot == '散布図':
                    if len(value.columns) >= 2:
                        ax.scatter(value[value.columns[0]], value[value.columns[1]])
                        ax.set_xlabel(value.columns[0])
                        ax.set_ylabel(value.columns[1])
                    else:
                        st.warning('散布図には少なくとも2つの列が必要です。')
                elif name_plot == 'ヒストグラム':
                    ax.hist(value, bins=20)

                # Displaying the figure in Streamlit
                st.pyplot(fig)


                if not value.empty:
                    if name == '平均':
                        value_mean = value.mean()
                        st.text(f'平均＝\n{value_mean}')
                    elif name == '分散':
                        value_var = value.var()
                        st.text(f'分散＝\n{value_var}')
                    elif name == '共分散':
                        value_cov = value.cov()
                        st.text(f'共分散＝\n{value_cov}')
                    elif name == '標準偏差':
                        value_std = value.std()
                        st.text(f'標準偏差＝\n{value_std}')
                    elif name == '相関係数':
                        value_corr = value.corr()
                        st.text(f'相関係数＝\n{value_corr}')
    cancel_btn = st.form_submit_button('キャンセル')

########

st.write('・csvの読み込みを行う際はあらかじめ、Excelなどを用いて、データの加工をしてもらう必要があります。')
st.write('・不具合等ございましたらお手数ですがTOPページ内のお問い合わせフォームにてお願いします。')







