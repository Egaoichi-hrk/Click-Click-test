import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import scipy.stats as stats
import japanize_matplotlib
#実行させるには（streamlit run C:\ウェブ企業\開発コード\WUM\pages\統計分析2.py

image = Image.open("Click Click LOGO 1.jpg")
st.set_page_config(
    page_title = "統計分析",
    page_icon = image,
)
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
st.subheader('統計分析２')
st.markdown(
    """
    <br>
    """
    , unsafe_allow_html=True)

######


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
