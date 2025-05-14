# tennis-predictor
Machine learning project predicting outcomes of ATP tennis matches with Streamlit dashboard interface. The dashboard is hosted publically on Streamlit community cloud:

https://tennispredictor.streamlit.app/

## File Overview
The Streamlit dashboard is run by *app.py*, which requires the following files:
- *app_services.py* - reads the data file and the machine learning model and makes the prediction.
- *gb_model.pkl* - the trained gradient boosting model
- *atp_tennis_engineered.csv* - the match data with engineered features
- *requirements.txt*

Other files are used to update the data and model:
- *download_and_clean_data.ipynb* - does what it says on the tin; outputs *atp_tennis.csv*
- *train.ipynb* - engineers features and trains the model. Outputs *atp_tennis_engineered.csv* and *gb_model.pkl*
- *predict.ipynb* - only used for experimenting with prediction code

There's also *tennis_data_column_explanations.txt*, downloaded from the source of the data.

## Improvements
There are **lots** of potential improvements that could be tried. Some of which are marked by "@TODO" comments in the code.
