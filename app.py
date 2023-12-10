import streamlit as st
from PIL import Image
import shutil
import numpy as np
import pandas as pd
from fastai.vision.all import *
from fastai.data.external import *
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic import PredictionServiceClient # change this line
from google.cloud.aiplatform.gapic.schema import predict as schema_predict # use an alias 
from google.oauth2 import service_account
import google.auth
import google.auth.transport.requests
import base64
import requests
from roboflow import Roboflow
#================================== SETUP =================================
model_path = "./resnet50best.pkl"
img_file = "temp.jpg"
# codeblock below is needed for Windows path #############
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
##########################################################

st.set_page_config(page_title='Tomato Classifier',
                       layout = 'wide', 
                       initial_sidebar_state = 'auto',
                       page_icon = './streamlit_images/logo.png')

learner = load_learner('./resnet50best.pkl', pickle_module=pickle)

classes = ["Celebrity tomato", "Green zebra tomato", "Pineapple tomato", "Yellow pear tomato", "Super sweet 100 tomato"]

temp_file_path = './temp.jpg'

#===============================Roboflow model setup=============================



roboflow_version = 2
rf = Roboflow(api_key=roboflow_API_Key)
project = rf.workspace().project(roboflow_model_endpoint)
roboflow_model = project.version(roboflow_version).model


#===============================Streamlit session variables setup==================================

# Initialize session_state
session_state = st.session_state
if 'score_right' not in session_state:
    session_state.score_right = 0
if 'score_wrong' not in session_state:
    session_state.score_wrong = 0
if 'score_google_right' not in session_state:
    session_state.score_google_right = 0
if 'score_google_wrong' not in session_state:
    session_state.score_google_wrong = 0
if 'score_roboflow_right' not in session_state:
    session_state.score_roboflow_right = 0
if 'score_roboflow_wrong' not in session_state:
    session_state.score_roboflow_wrong = 0
if 'total_count' not in session_state:
    session_state.total_count = 0

def increase_total_count():
    session_state.total_count += 1
def increase_score_right():
    session_state.score_right += 1
def increase_score_wrong():
    session_state.score_wrong += 1
def increase_score_google_right():
    session_state.score_google_right += 1
def increase_score_google_wrong():
    session_state.score_google_wrong += 1
def increase_score_roboflow_right():
    session_state.score_roboflow_right += 1
def increase_score_roboflow_wrong():
    session_state.score_roboflow_wrong += 1

credentials = service_account.Credentials.from_service_account_file("./teak-sun-407318-bae05ef2960d.json")
# credentials = google.auth.service_account.Credentials.from_service_account_file(
#     './teak-sun-407318-bae05ef2960d.json',
#     scopes=['https://www.googleapis.com/auth/cloud-platform']
# )
# request = google.auth.transport.requests.Request()
# credentials.refresh(request)

def predict_image_classification_sample(
   
   
    filename: str = "temp.jpg",
    location: str = "europe-west4",
    
):
    # The AI Platform services require regional API endpoints
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options, credentials=credentials)
    with open(filename, "rb") as f:
        file_content = f.read()

    # The format of each instance should conform to the deployed model's prediction input schema.
    encoded_content = base64.b64encode(file_content).decode("utf-8")
    instance = schema_predict.instance.ImageClassificationPredictionInstance(
        content=encoded_content,
    ).to_value()
    instances = [instance]
    # See gs://google-cloud-aiplatform/schema/predict/params/image_classification_1.0.0.yaml for the format of the parameters.
    parameters = schema_predict.params.ImageClassificationPredictionParams(
        confidence_threshold=0.5,
        max_predictions=5,
    ).to_value()
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    # print("response")
    # print(" deployed_model_id:", response.deployed_model_id)
    # See gs://google-cloud-aiplatform/schema/predict/prediction/image_classification_1.0.0.yaml for the format of the predictions.
    predictions = response.predictions
    for prediction in predictions:
        return dict(prediction)['displayNames']

def predict_image_classification_sample_request(image_bytes):
    data = {
        "instances": [{
            "content": "image_bytes"
        }],
        "parameters": {
            "confidenceThreshold": 0.5,
            "maxPredictions": 5
        }
    }
    headers = {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json"
    }
    url = "https://europe-west4-aiplatform.googleapis.com/v1/projects/618340661486/locations/europe-west4/endpoints/1528941287166705664:predict"
    response = requests.post(url, data=data, headers=headers)
    result = response.json()
    return result


score_placeholder = st.empty()

def tensor_to_rounded_float_list(tensor, decimals):
    # Create an empty list to store the rounded float values
    rounded_float_list = []
    # Loop through each element of the tensor
    for value in tensor:
        # Round the value to the given number of decimal places using torch.round and convert it to a Python scalar value
        rounded_float_value = torch.round(value * 10**decimals) / (10**decimals)
        rounded_float_value = rounded_float_value.item()
        # Round the value to the given number of decimal places and append it to the list
        rounded_float_value = round(rounded_float_value, decimals)
        rounded_float_list.append(rounded_float_value)
    # Return the list of rounded float values
    return rounded_float_list

def convert_roboflow_result(response, classes):
    predictions = response.predictions

    # Find the prediction with the highest confidence
    top_prediction = max(predictions, key=lambda x: x.json_prediction["confidence"])

    # Access class and confidence from the top prediction
    top_class = top_prediction.json_prediction["top"]
    top_confidence = top_prediction.json_prediction["confidence"]

    # Access class and confidence for each prediction in the list
    class_confidence = {prediction["class"]: prediction["confidence"] for prediction in top_prediction.json_prediction["predictions"]}

    # Create a list of confidences for each class in 'classes'
    confidences = [class_confidence.get(class_name, 0) for class_name in classes]

    return top_class, confidences

def predict(image):
    img = PILImage.create(image)
    pred = learner.predict(img)
    return pred[0], tensor_to_rounded_float_list(pred[2], 4)

def update_progress_bars(progress_bars, confidences, classes, status):
    for i, progress_bar in enumerate(progress_bars):
        progress_bar.progress(confidences[i], text=f"{classes[i]}: {confidences[i]}")
    status.success("Done!", icon='✅')


#==========================================================================
st.title("Transfer learning model for classification of tomato types:tomato::frame_with_picture:")
st.caption('''
Created by [Daan Michielsen](https://github.com/DaanMichielsen) and [Sharon Maharjan](https://github.com/SharonMaharjan)
           ''')
st.caption('''
[GitHub repository](https://github.com/SharonMaharjan/Big_Data-Project)
           ''')


st.subheader("Classes of tomatoes",divider="red")
celebrity, zebra, pineapple, pear, sweet = st.columns(5)
# zebra, sweet = st.columns(2)

with celebrity:
    st.markdown('''**1. Celebrity**''')
    st.image("./streamlit_images/celebrity_tomato.png", use_column_width=True)
with zebra:
    st.markdown('''**2. Green zebra**''')
    st.image("./streamlit_images/green_zebra_tomato.png", use_column_width=True)
with pineapple:
    st.markdown('''**3. Pineapple**''')
    st.image("./streamlit_images/pineapple_tomato.png", use_column_width=True)
with pear:
    st.markdown('''**4. yellow pear**''')
    st.image("./streamlit_images/yellow_pear_tomato.png", use_column_width=True)
with sweet:
    st.markdown('''**5. Super sweet 100**''')
    st.image("./streamlit_images/super_sweet_100_tomato.png", use_column_width=True)

st.subheader("About the model", divider="red")
st.markdown('''
We used a pretrained ResNet50 model to classify the tomatoes by applying transfer learning on the model using our own data. The model was trained on a dataset of 5 classes of tomatoes. The dataset was created by scraping images from Google Images.
''')
model_name = Path(model_path).name
with open(model_path, 'rb') as f:
    model_content = f.read()
st.download_button(label="Download Model", key="download_model", data=model_content, file_name=model_name)
st.subheader("Results", divider="red")
status = st.info("Run predictions to show results")
custom_result, google_result, roboflow_result = st.columns(3)
with custom_result:
    st.subheader("Custom", divider="red")
    conf_celebrity = st.empty()
    conf_zebra = st.empty()
    conf_pineapple = st.empty()
    conf_yellow_pear = st.empty()
    conf_sweet = st.empty()
with google_result:
    st.subheader("Google", divider="rainbow")
    conf_google_celebrity = st.empty()
    conf_google_zebra = st.empty()
    conf_google_pineapple = st.empty()
    conf_google_yellow_pear = st.empty()
    conf_google_sweet = st.empty()
with roboflow_result:
    st.subheader("Roboflow", divider="violet")
    conf_roboflow_celebrity = st.empty()
    conf_roboflow_zebra = st.empty()
    conf_roboflow_pineapple = st.empty()
    conf_roboflow_yellow_pear = st.empty()
    conf_roboflow_sweet = st.empty()
sliders = [conf_celebrity, conf_zebra, conf_pineapple, conf_yellow_pear, conf_sweet]
sliders_google = [conf_google_celebrity, conf_google_zebra, conf_google_pineapple, conf_google_yellow_pear, conf_google_sweet]
sliders_roboflow = [conf_roboflow_celebrity, conf_roboflow_zebra, conf_roboflow_pineapple, conf_roboflow_yellow_pear, conf_roboflow_sweet]
st.sidebar.header("Try it yourself:grinning:", divider="red")

uploaded_image = st.sidebar.file_uploader("Upload image:outbox_tray:", type=["jpg","png"], 
                                              accept_multiple_files=False)
# image_data = uploaded_image.read ()
image = Image.open (uploaded_image)
image.save (temp_file_path)
# Add a selectbox for the user to choose the correct class
selected_class = st.sidebar.selectbox("Select the correct class:", classes, key="select_class", help=":orange[Select the class of the uploaded image]")
assigned_class_id = classes.index(selected_class)
payload = {"image": {"content": image}}
if uploaded_image != None:
    st.sidebar.subheader("Input:frame_with_picture:", divider='red')
    st.sidebar.image(uploaded_image, use_column_width=True)
    if st.sidebar.button(":green[Classify image]", key="classify_image", disabled=False if selected_class != "" else True):
        result, confs = predict(uploaded_image)
        update_progress_bars(sliders, confs, classes, status)
        response = predict_image_classification_sample()
        # st.write(predict_image_classification_sample_request(image))
        response_roboflow = roboflow_model.predict(temp_file_path)
        prediction_roboflow, confidences_roboflow = convert_roboflow_result(response_roboflow, classes)
        update_progress_bars(sliders_roboflow, confidences_roboflow, classes, status)
        clean_response = "".join(response).replace("_", " ")
        st.sidebar.subheader("Output", divider='red')
        st.sidebar.markdown("Our model:")
        st.sidebar.subheader(f'''
            {":white_check_mark:" if selected_class.lower() == result.lower() else ":x:"}{" :green"if selected_class.lower() == result.lower() else " :red"}[{result}]
        ''') 
        st.sidebar.markdown(":blue[G]:red[o]:orange[o]:blue[g]:green[l]:red[e] vertex model:")
        st.sidebar.subheader(f'''
            {":white_check_mark:" if selected_class.lower() == clean_response.lower() else ":x:"}{" :green"if selected_class.lower() == clean_response.lower() else " :red"}[{"".join(clean_response)}]
        ''')
        st.sidebar.markdown(":violet[Roboflow] model:")
        st.sidebar.subheader(f'''
            {":white_check_mark:" if selected_class.lower() == prediction_roboflow.lower() else ":x:"}{" :green"if selected_class.lower() == prediction_roboflow.lower() else " :red"}[{prediction_roboflow}]
        ''')
        increase_total_count()
        if selected_class.lower() == result.lower():
            increase_score_right()
        else:
            increase_score_wrong()
        if selected_class.lower() == clean_response.lower():
            increase_score_google_right()
        else:
            increase_score_google_wrong()
        if selected_class.lower() == prediction_roboflow.lower():
            increase_score_roboflow_right()
        else:
            increase_score_roboflow_wrong()

st.sidebar.subheader("Score:trophy:", divider="red", help=":orange[The score changes based on your feedback after classifications]")
# Update the displayed scores
accuracy = round(session_state.score_right / session_state.total_count * 100, 2) if session_state.total_count > 0 else 0
accuracy_google = round(session_state.score_google_right / session_state.total_count * 100, 2) if session_state.total_count > 0 else 0
accuracy_roboflow = round(session_state.score_roboflow_right / session_state.total_count * 100, 2) if session_state.total_count > 0 else 0

# Create a pandas dataframe from the score variables
score_md = f'''
Tries: {session_state.total_count}

|         | Custom | :blue[G]:red[o]:orange[o]:blue[g]:green[l]:red[e] | :violet[Roboflow] |
|---------|--------|--------|----------|
| :green[Right]   | :green[{session_state.score_right}/{session_state.total_count}] | :green[{session_state.score_google_right}/{session_state.total_count}] | :green[{session_state.score_roboflow_right}/{session_state.total_count}] |
| :red[Wrong]   | :red[{session_state.score_wrong}/{session_state.total_count}] | :red[{session_state.score_google_wrong}/{session_state.total_count}] | :red[{session_state.score_roboflow_wrong}/{session_state.total_count}] |
| Accuracy| {accuracy} % | {accuracy_google} % | {accuracy_roboflow} % |
'''

# Display the markdown string as a formatted text
st.sidebar.markdown(score_md)

# Update the displayed scores
score_placeholder.markdown('')

if st.sidebar.button("Reset score", key="reset_score", disabled=True if session_state.total_count == 0 else False):
    session_state.score_right = 0
    session_state.score_wrong = 0
    session_state.score_google_right = 0
    session_state.score_google_wrong = 0
    session_state.score_roboflow_right = 0
    session_state.score_roboflow_wrong = 0
    session_state.total_count = 0
    # Update the displayed scores
    score_placeholder.markdown('')

