import streamlit as st

from streamlit_ace import st_ace, KEYBINDINGS, LANGUAGES, THEMES
#from streamlit_gallery.utils import readme

from streamlit_player import st_player, _SUPPORTED_EVENTS

import numpy as np
import pandas as pd
from PIL import Image
import requests
from pycaret.classification import *

import pandas_profiling
from streamlit_pandas_profiling import st_profile_report


@st.cache(allow_output_mutation=True)
def gen_profile_report(df, *report_args, **report_kwargs):
    return df.profile_report(*report_args, **report_kwargs)


with st.sidebar:
    image = Image.open('kunishou.png')
    st.image(image,width=120)
    #image = Image.open('space.jpg')
    #st.image(image,use_column_width=True)

    st.title("Kunishou's WEB Tools")

    st.write('WEB上で使用可能な機械学習・データ分析ツールを公開しています。Streamlitのメモリ上限の関係で一部実行できないものがあります。')
    st.write('')
    sideradio = st.radio('',('FAST PyCaret', 'Pandas-Profiling', 'Ace Editor', 'My Movies', 'About me'))

    st.markdown("")
    st.markdown("## Link")
    st.markdown("[Qiita](https://qiita.com/kunishou)")
    st.markdown("[GitHub](https://github.com/kunishou)")

#-------------------------------------------------------------------------------------------------

if sideradio == 'FAST PyCaret':
    #st.title('FAST PyCaret')
    image2 = Image.open('logo.png')
    st.image(image2,use_column_width=True)
    #url2 = 'https://pycaret.org/wp-content/uploads/2020/03/Divi93_43.png'
    #image2 = Image.open(requests.get(url2,stream=True).raw)
    #st.image(image2,width=400)
    st.write('')

    st.write('手軽にモデル性能を比較するためのAutoMLツールです。')
    st.write('現段階では分類タスクのみ実施可能。回帰タスクは作成中。')
    st.write('')

    task = st.selectbox('機械学習タスクを選択',('分類', '回帰'))
    st.write('')

    if task == '分類':

        uploaded_file = st.file_uploader('1. ファイルをアップロードする、もしくはタイタニック号の乗客データを使用する', type='csv')

        check = st.checkbox('タイタニック号の乗客データを使用', value=False)

        if uploaded_file is not None:
            st.write('')
            radio = st.radio('2.ファイルのエンコード形式を選択',('utf-8', 'cp932'))
            st.write('')

            if 'push1' not in st.session_state:
                st.session_state.push1 = False #push1がsession_stateに追加されていない場合，False

            button1 = st.button('3. ファイル読み込み')
            st.write('')

            if button1:
                st.session_state.push1 = True

        # メイン画面
        if (uploaded_file is not None and st.session_state.push1) or check:
            st.markdown('### 読み込みデータ表示')
            # アップロードファイルをメイン画面にデータ表示
            if uploaded_file is not None and st.session_state.push1:
                df1 = pd.read_csv(uploaded_file, encoding=radio)
            if check:
                df1 = pd.read_csv('titanic.csv')

            st.write(df1)
            st.write('')
            
            if check:
                target = st.selectbox('4. 目的変数を選択',('Survived',))
            else:
                target = st.selectbox('4. 目的変数を選択',tuple(df1.columns))

            df1_2 = df1.drop(target,axis=1)
            imbalance = st.checkbox('5. 目的変数にラベルの不均衡あり', value=False)
            ignore_features = st.multiselect('6. 使用しない特徴量を選択（未選択可）', list(df1_2.columns))
            st.write('')

            if 'push2' not in st.session_state:
                st.session_state.push2 = False

            button1_2 = st.button('7. モデル比較の実行')
            st.write('※完了までに時間がかかります')
            st.write('')

            if button1_2:
                st.session_state.push2 = True

            if target and st.session_state.push2:
                st.write('使用メモリ上限の関係でこの先実行不可')

    if task == '回帰':
        st.markdown('# Under Construction')

#-------------------------------------------------------------------------------------------------

if sideradio == 'Pandas-Profiling':
    
    #st.title('Pandas-Profiling')
    url3 = 'https://pandas-profiling.github.io/pandas-profiling/docs/assets/logo_header.png'
    image3 = Image.open(requests.get(url3,stream=True).raw).crop((100,200,1270,550))
    st.image(image3,use_column_width=True)
    
    st.write('')
    st.write('pandasデータフレームのプロファイリング結果をまとめて出力可能なEDAツールです。')

    uploaded_file2 = st.file_uploader('1. ファイルをアップロードする、もしくはタイタニック号の乗客データのDEMOを表示する', type='csv')

    check2 = st.checkbox('タイタニック号の乗客データのDEMOを表示', value=False)

    if uploaded_file2 is not None:
        st.write('')
        radio2 = st.radio('2.ファイルのエンコード形式を選択',('utf-8', 'cp932'))
        st.write('')

        button2 = st.button('3. ファイル読み込み')
        st.write('')

    # メイン画面
    if (uploaded_file2 is not None and button2) or check2:
        # アップロードファイルをメイン画面にデータ表示
        if uploaded_file2 is not None and button2:
            st.header('読み込みデータ表示')
            df2 = pd.read_csv(uploaded_file2, encoding=radio2)
            st.write(df2)
            st.write('')
            pr = df2.profile_report()
            st_profile_report(pr)
        if check2:
            df2 = pd.read_csv('titanic.csv')
            st.write(df2)
            pr = gen_profile_report(df2, explorative=True)
            with st.expander("DEMO REPORT", expanded=True):
                st_profile_report(pr)

#-------------------------------------------------------------------------------------------------

if sideradio == 'Ace Editor':
    #st.title('Ace Editor')
    url4 = 'https://ace.c9.io/doc/site/images/ace-logo.png'
    image4 = Image.open(requests.get(url4,stream=True).raw)
    st.image(image4,width=150)
    st.write('')
    st.write('ブラウザ上で動作するJavaScriptベースのコードエディタです')
    st.write('サイドバーから設定変更ができます。')    

    st.sidebar.title("⚙️ Parameters")

    with st.container():
        content = st_ace(
                placeholder=st.sidebar.text_input("Editor placeholder", value="Write your code here"),
                language=st.sidebar.selectbox("Language mode", options=LANGUAGES, index=121),
                theme=st.sidebar.selectbox("Theme", options=THEMES, index=35),
                keybinding=st.sidebar.selectbox("Keybinding mode", options=KEYBINDINGS, index=3),
                font_size=st.sidebar.slider("Font size", 5, 24, 14),
                tab_size=st.sidebar.slider("Tab size", 1, 8, 4),
                show_gutter=st.sidebar.checkbox("Show gutter", value=True),
                show_print_margin=st.sidebar.checkbox("Show print margin", value=False),
                wrap=st.sidebar.checkbox("Wrap enabled", value=False),
                auto_update=st.sidebar.checkbox("Auto update", value=False),
                readonly=st.sidebar.checkbox("Read-only", value=False),
                key="ace",
                )

    st.write(content)
        
#-------------------------------------------------------------------------------------------------

if sideradio == 'My Movies':
    with st.sidebar:
        st.title("⚙️ Parameters")

        options = {
            "events": st.multiselect("Events to listen", _SUPPORTED_EVENTS, ["onProgress"]),
            "progress_interval": 1000,
            "volume": st.slider("Volume", 0.0, 1.0, 1.0, .01),
            "playing": st.checkbox("Playing", False),
            "loop": st.checkbox("Loop", False),
            "controls": st.checkbox("Controls", True),
            "muted": st.checkbox("Muted", False),
        }

    st.markdown("### 🎬 Python初学者のためのPandas100本ノック  [(Qiita)](https://qiita.com/kunishou/items/bd5fad9a334f4f5be51c)")
    st.write('')
    st_player("https://youtu.be/apYJZbiM_D4")
    st.write('')
    st.markdown("### 🎬 夏なのでStreamlitを用いてWEBアプリケーションを作成してみた")
    st.write('')
    st_player("https://youtu.be/PEA1us0II4Q")

#-------------------------------------------------------------------------------------------------

if sideradio == 'About me':

    """
    # Kunishou
    通信業界で働いている社会人11年目。Python、機械学習が好き。何かを形にするのが好き。Kaggle Expertを目指しています。横浜出身。
    ## 資格
    統計検定2級/ Python3エンジニア認定基礎試験・データ分析試験/ ORACLE MASTER Bronze/ AWSクラウドプラクティショナー/ AI Quest2020修了/ 知的財産管理技能検定3級/ and more ...
    ## スキル
    * Python: Python/ PyTorch/ Django/ Flask/ Streamlit/ Tkinter
    * Other: SQL/ HTML/ CSS
    * Photoshop/ Illustrator/ SAI/ After Effects/ Photoshop Lightroom/ Fusion360
    ## 趣味
    機械学習、Pythonの勉強/ デジタルクリエイト/ 音楽/ カメラ
    ## 座右の銘
    バットを振らなきゃホームランは打てない

    """

st.sidebar.markdown("")
st.sidebar.markdown("## Link")
st.sidebar.markdown("[Qiita](https://qiita.com/kunishou)")
st.sidebar.markdown("[GitHub](https://github.com/kunishou)")
