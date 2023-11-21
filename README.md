Big Data Project (Assignment Part A - Image Classification)
1. Collecting/Selecting dataset
  - Collect or select a dataset of images that can be categorized in at least 5 different classes.
  - Make sure the images you want to classify are challenging enough (so no ‘car versus truck versus plane’-type of classifications).
  - The level of challenge and creativity concerning the dataset.
  - Assess the quality of your dataset, including an EDA. If it’s not up to par, take steps to remedy. Include this analysis in your final report.
  - Possible Extra:
      o Include images with more than one label and extend your model so you can do multi-label predictions
2. Modeling using fastai
  - Use Fastai's / Pytorch modelling techniques, including transfer learning, and some of the advanced options we’ve discussed in class, to build a state of the art deep learning model that is able to classify the images.
     o To be clear: fastai/pytorch has to be used. See Extra for Keras / Tensorflow models
  - Assess the quality of the learned model. Include this analysis in your final report.
  - Possible Extra:
      o Keras/TensorFlow models can only be used as a possible extra, to be able to compare the ease of use/training versus the fastai/Pytorch models. Make sure the Keras models also apply advanced learning options, and employ transfer learning.
  - Possible Extra:
      o Research and apply ROC and AUC metrics to assess the quality of your classification model
  - Possible Extra:
      o Benchmark your model’s performance to the performance of Google’s TeachableMachine. And/Or compare your results with plug-and-play solutions like Google’s AutoML Vision or Google’s Vision API (use your GCP student credits). Or, you can even give Google’s new Vertex AI platform a spin. Roboflow is another platform you can use for comparison.
3. Deploy as WebApp, via Streamlit
  - Make an interactive web application using Streamlit, where:
      o You can explore / evaluate the dataset (basics of part A of this assignment)
      o You can test a pre-saved model, by uploading a new image and have the application tell you the class the image belongs to.
    ▪ Warning: models can become very large. Research how to use Git LFS on Github. If you’re unable to get it working for Streamlit Cloud, use a local Streamlit install, and add screenshots to your report
  - Possible Extra:
      o You can train a new image classifier model, by selecting advanced fastai options, and save the trained model when done (basics of part B)
    ▪ This option is only possible for a local Streamlit install, not for the Streamlit Cloud. Add screenshots to your report to demonstrate this feature, when chosen.
      o Create an API endpoint around your model, by using FastAPI (don’t confuse it with fastai), or by using Flask.
    ▪ Make sure you can test your API (curl, Postman,…)
