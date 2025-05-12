import streamlit as st
import pandas as pd
from app_services import Predictor

predictor = Predictor()
model_info = predictor.get_model_info()

# set page config
st.set_page_config(
    page_title="ATP Tennis Predictor",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# dashboard title
st.title("ATP Tennis Predictor")

# sidebar
st.sidebar.title("Model Information")
st.sidebar.write(f"""
This application uses a {model_info['model_type']} model to predict ATP tennis results. It was last updated on {model_info['date_created'].date()}
and has an accuracy of **{100*model_info['accuracy_test']:.1f}%** on the test set.

The dataset used for training and testing contains all ATP tour singles matches since 2000: over 60,000 matches.

The feature columns used in the model are: {set([x.split(' ')[0] for x in model_info['features']])} for each player.

Created by Adam Vaughan.
""")

# main content

# create a form for user to input Player 1, Player 2 and Surface
with st.form(key="prediction_form"):
    st.subheader("Match Prediction")
    
    # input dropdowns for user to select player names and surface
    # @TODO: restrict to selection to active players (match in last year?)
    player_A = st.selectbox("Select Player 1", predictor.players, index=predictor.players.index("Sinner J."))
    player_B = st.selectbox("Select Player 2", predictor.players, index=predictor.players.index("Alcaraz C."))
    surface = st.selectbox("Select Surface", predictor.surfaces)
    
    # submit button
    submit_button = st.form_submit_button(label='Predict Match Outcome')

# if the button is clicked, make prediction
if submit_button:
    if player_A == player_B:
        st.error("Please select two different players.")
    else:
        # make prediction
        winner, probability = predictor.predict_winner(player_A, player_B, surface)
        
        # display the result
        st.success(f"The predicted winner is: **{winner}** with a probability of **{round(100*probability)}%**")

    # @TODO: add feature data to the output
    st.write("Feature data of this match:")
    # create a table
    feature_df = predictor.get_feature_data(player_A, player_B, surface)

    display_feature_df = pd.DataFrame(
        columns=[player_A, "Player", player_B],
        data=[
            [int(feature_df["Rank A"][0]), "Rank", int(feature_df["Rank B"][0])],
            [round(100*feature_df["Form A"][0]), "Form (win % in last 90 days)", round(100*feature_df["Form B"][0])],
            [round(100*feature_df["H2HWinProportion A"][0]), "H2H Win %", round(100*feature_df["H2HWinProportion B"][0])],
            [round(100*feature_df["SurfaceWinProportion A"][0]), "Surface Win %", round(100*feature_df["SurfaceWinProportion B"][0])],
        ],
    )

    st.dataframe(display_feature_df, hide_index=True)

    # @TODO: add previous H2H matches to the output