# datacredit_ml_azure
### Data credit machine learning project

This project demonstrates the integration of a machine learning model into a web application using Flask. The machine learning model, a Random Forest Classifier, is trained to predict a label based on various features.

For this project we'll use this dataset: https://www.kaggle.com/datasets/rohitudageri/credit-card-details?select=Credit_card.csv

#### Files
1. 0_ML_notebook.ipynb: A Jupyter Notebook containing the code to train the Random Forest model and save it as 'random_forest_model.pkl'.

2. 1_ML_model.py: Python script to preprocess the input data, including label encoding of categorical features.

3. app.py: Flask web application that receives numerical input values for various features, uses the trained model to predict the label, and displays the result.

4. templates/index.html: HTML template for the web application, including a form to input feature values and display the prediction result.

##### Usage

1. Run the Jupyter Notebook train_model.ipynb to train the Random Forest model and save it as 'random_forest_model.pkl'.

2. Execute preprocess_data.py to preprocess the input data and save the processed data.

3. Run the Flask web application using app.py.

4. Access the web application in your browser and enter numerical values for the specified features. Submit the form to see the predicted label.

The Docker Image will be pull in: https://hub.docker.com/r/jeanpierec/credit-forest-predictor
 
 
 
 
*_(to Harry and Hermione) Mysterious thing, time. Powerful, and when meddled with, dangerous. Sirius Black is in the topmost cell of the dark tower. You know the laws, Miss Granger. You must not be seen. And you would do well, I feel, to return before this last chime. If not, the consequences are too ghastly to discuss. If you succeed tonight, more than one innocent life may be spared. Three turns, should do it, I think. Oh, and by the way. When in doubt, I find retracing my steps to be a wise place to begin. Good luck._*

*Harry Potter and the Prisoner of Azkaban (film) (2004)*