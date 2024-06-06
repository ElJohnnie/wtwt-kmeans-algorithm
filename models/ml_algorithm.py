import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans

class MLEngine:
    def __init__(self, csv_path='../data/movies.csv'):
        self.df = pd.read_csv(csv_path)
        self._preprocess_data()
        self._train_model()
    
    def _preprocess_data(self):
        self.df['year'] = self.df['title'].str.extract(r'\((\d{4})\)').astype(int)
        self.genres = self.df['genres'].str.get_dummies(sep='|')
        self.df = self.df.join(self.genres)
        self.df['year'] = self.df['year'] - self.df['year'].min()

    def _train_model(self):
        features = self.genres.columns.tolist() + ['year']
        self.kmeans = KMeans(n_clusters=10, random_state=42)
        self.df['cluster'] = self.kmeans.fit_predict(self.df[features])

    def predict(self, mood, primary_genre, secondary_genre, epoch):
        mood_to_genre = {
            'Feliz': 'Comedy',
            'Triste': 'Drama',
            'Ansioso': 'Thriller',
            'Animado': 'Action',
            'Entediado': 'Adventure'
        }
        
        primary_cluster = self.df[(self.df[primary_genre] == 1) & (self.df['year'] >= epoch)].groupby('cluster').size().idxmax()
        secondary_cluster = self.df[(self.df[secondary_genre] == 1) & (self.df['year'] >= epoch)].groupby('cluster').size().idxmax()
        recommended_movies = self.df[(self.df['cluster'].isin([primary_cluster, secondary_cluster])) & (self.df[mood_to_genre[mood]] == 1)]
        
        return {"message": "Predição realizada com sucesso!", "data": recommended_movies[['title', 'genres']].to_dict(orient='records')}
