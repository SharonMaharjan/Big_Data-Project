import streamlit as st
from PIL import Image
import shutil
import numpy as np
from fastai.vision.all import *
from fastai.data.external import *
#================================== SETUP =================================
model_path = "./resnet50best.pkl"
img_file = "temp.jpg"
# codeblock below is needed for Windows path #############
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
##########################################################

learner = load_learner('./resnet50best.pkl')

# Initialize session_state
session_state = st.session_state
if 'score_right' not in session_state:
    session_state.score_right = 0
if 'score_wrong' not in session_state:
    session_state.score_wrong = 0
if 'total_count' not in session_state:
    session_state.total_count = 0

def increase_score_right():
    session_state.score_right += 1
    session_state.total_count += 1

def increase_score_wrong():
    session_state.score_wrong += 1
    session_state.total_count += 1

def show_results():
    st.sidebar.markdown(f'''
        Tries: {session_state.total_count}\n
        Right: :green[{session_state.score_right}]\n
        Wrong: :red[{session_state.score_wrong}]\n
        Accuracy: {session_state.score_right / session_state.total_count * 100 if session_state.total_count > 0 else 0} %
    ''')

# Create an instance of the Scoreboard outside any function or method

score_placeholder = st.empty()

def predict(image):
    img = PILImage.create(image)
    pred = learner.predict(img)
    print(pred)
    return pred[0]


#==========================================================================
st.title("Transfer learning model for classification of tomato types:tomato::frame_with_picture:")
st.caption('''
Created by [Daan Michielsen](https://github.com/DaanMichielsen) and [Sharon Maharjan](https://github.com/SharonMaharjan)
           ''')
st.caption('''
[GitHub repository](https://github.com/SharonMaharjan/Big_Data-Project)
           ''')


st.subheader("Classes of tomatoes",divider="red")
celebrity, pineapple, pear = st.columns(3)
nothing, zebra, nothing, sweet, nothing = st.columns(5)

with celebrity:
    st.markdown('''**1. Celebrity**''')
    st.image("./streamlit_images/celebrity_tomato.png", use_column_width=True)
with pineapple:
    st.markdown('''**2. Pineapple**''')
    st.image("./streamlit_images/pineapple_tomato.png", use_column_width=True)
with pear:
    st.markdown('''**3. yellow pear**''')
    st.image("./streamlit_images/yellow_pear_tomato.png", use_column_width=True)
with zebra:
    st.markdown('''**4. Green zebra**''')
    st.image("./streamlit_images/green_zebra_tomato.png", width=225)
with sweet:
    st.markdown('''**5. Supersweet 100**''')
    st.image("./streamlit_images/super_sweet_100_tomato.png", width=225)

st.sidebar.header("Try it yourself:grinning:", divider="red")

uploaded_image = st.sidebar.file_uploader("Upload image:outbox_tray:", type=["jpg","png"], 
                                              accept_multiple_files=False)

# Add a selectbox for the user to choose the correct class
selected_class = st.sidebar.selectbox("Select the correct class:", ["Celebrity tomato", "Pineapple tomato", "Yellow pear tomato", "Green zebra", "Super sweet 100"])

if uploaded_image != None:
    st.sidebar.subheader("Input:frame_with_picture:", divider='red')
    st.sidebar.image(uploaded_image, use_column_width=True)
    if st.sidebar.button(":green[Classify image]", key="classify_image", disabled=False if selected_class != "" else True):
        result = predict(uploaded_image)
        st.sidebar.subheader("Output", divider='red')
        st.sidebar.title(f'''
            {":white_check_mark:" if selected_class.lower() == result.lower() else ":x:"}{" :green"if selected_class.lower() == result.lower() else " :red"}[{result}]
        ''')        
        if selected_class.lower() == result.lower():
            increase_score_right()
        else:
            increase_score_wrong()

st.sidebar.subheader("Score:trophy:", divider="red", help=":orange[The score changes based on your feedback after classifications]")
# Update the displayed scores
accuracy = round(session_state.score_right / session_state.total_count * 100, 2) if session_state.total_count > 0 else 0
st.sidebar.markdown(f'''
    Tries: {session_state.total_count}\n
    Right: :green[{session_state.score_right}]\n
    Wrong: :red[{session_state.score_wrong}]\n
    Accuracy: {accuracy} %
''')

# Update the displayed scores
score_placeholder.markdown('')

if st.sidebar.button("Reset score", key="reset_score", disabled=True if session_state.total_count == 0 else False):
    session_state.score_right = 0
    session_state.score_wrong = 0
    session_state.total_count = 0
    # Update the displayed scores
    score_placeholder.markdown('')