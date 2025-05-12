import pandas as pd
import pickle
import datetime as dt
import sklearn

class Predictor:
    def __init__(self):
        # load model
        with open('gb_model.pkl', 'rb') as file:
            self.model = pickle.load(file)

        # load data
        self.df = pd.read_csv('atp_tennis_engineered.csv', index_col=0)
        self.df['Date'] = pd.to_datetime(self.df['Date'])

        # create list of players who have played in the last year to be
        # used in the dropdowns
        df = self.df.copy()
        last_year_df = df.loc[df['Date'] >= pd.to_datetime(self.model.metadata['date_created']) - pd.Timedelta(days=366)]
        self.players = sorted(list(set(last_year_df['Winner']).union(set(last_year_df['Loser']))))

        self.surfaces = self.df['Surface'].unique()

    def get_model_info(self) -> dict:
        ''' Unpack the model metadata and return it as a dictionary '''

        if self.model.metadata['model'] == 'gb':
            model_type = 'Gradient Boosting'
        else:
            model_type = 'Unknown'

        return {
            "model_type": model_type,
            "date_created": pd.to_datetime(self.model.metadata['date_created']),
            "accuracy_test": self.model.metadata['model_metrics']['accuracy_test'],
            "features": self.model.metadata['features']
        }
    
    def get_feature_data(self, player_A, player_B, surface) -> pd.DataFrame:
        ''' Calculate the feature data for the given players and surface '''

        df = self.df.copy()
        # find all matches containing player A and player B as either Winner or Loser
        matches_A = df[(df['Winner'] == player_A) | (df['Loser'] == player_A)]
        matches_B = df[(df['Winner'] == player_B) | (df['Loser'] == player_B)]

        # find rank of players in the last match they played
        last_match_A = matches_A.sort_values('Date').iloc[-1]
        last_match_B = matches_B.sort_values('Date').iloc[-1]
        rank_A = last_match_A['WRank'] if last_match_A['Winner'] == player_A else last_match_A['LRank']
        rank_B = last_match_B['WRank'] if last_match_B['Winner'] == player_B else last_match_B['LRank']

        # find win proportion of each player in 90 days up to last match in dataset
        last_date = pd.to_datetime(df['Date'].max())
        form_start_date = (last_date - pd.DateOffset(days=90)).date()
        form_matches_A = matches_A[pd.to_datetime(matches_A['Date']).dt.date >= form_start_date]
        form_matches_B = matches_B[pd.to_datetime(matches_B['Date']).dt.date >= form_start_date]

        form_A = (form_matches_A['Winner'] == player_A).sum() / len(form_matches_A) if len(form_matches_A) > 0 else 0
        form_B = (form_matches_B['Winner'] == player_B).sum() / len(form_matches_B) if len(form_matches_B) > 0 else 0

        # find win proportion of each player on the given surface
        surface_matches_A = matches_A[matches_A['Surface'] == surface]
        surface_matches_B = matches_B[matches_B['Surface'] == surface]
        surfacewinproportion_A = (surface_matches_A['Winner'] == player_A).sum() / len(surface_matches_A) if len(surface_matches_A) > 0 else 0
        surfacewinproportion_B = (surface_matches_B['Winner'] == player_B).sum() / len(surface_matches_B) if len(surface_matches_B) > 0 else 0

        # find head-to-head win proportion of each player
        h2h_matches = df[((df['Winner'] == player_A) & (df['Loser'] == player_B)) | ((df['Winner'] == player_B) & (df['Loser'] == player_A))]
        h2h_A = (h2h_matches['Winner'] == player_A).sum() / len(h2h_matches) if len(h2h_matches) > 0 else 0.5
        h2h_B = 1 - h2h_A

        return pd.DataFrame(
            data=[[rank_A, rank_B, surfacewinproportion_A, surfacewinproportion_B, h2h_A, h2h_B, form_A, form_B]],
            columns=self.model.metadata['features']
        )
    
    def predict_winner(self, player_A, player_B, surface) -> str:
        ''' Perform prediction and return the winner '''

        feature_df = self.get_feature_data(player_A, player_B, surface)
        prediction = self.model.predict(feature_df)[0]
        prediction_proba = max(self.model.predict_proba(feature_df)[0])

        return (player_A if prediction == 1 else player_B, prediction_proba)

