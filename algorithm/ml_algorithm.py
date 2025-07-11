import pandas as pd
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor

class MLEngine:
    def __init__(self, csv_path='data/v3.csv'):
        self.df = pd.read_csv(csv_path)
        self._preprocess_data()
        self._train_model()
        self.cache = {}

    def _preprocess_data(self):
        self.df['year'] = self.df['title'].str.extract(r'\((\d{4})\)')
        self.df['year'] = pd.to_numeric(self.df['year'], errors='coerce')
        self.df = self.df.dropna(subset=['year'])
        self.df['year'] = self.df['year'].astype(int)
        self.genres = self.df['genres'].str.get_dummies(sep='|')
        self.df = self.df.join(self.genres)

    def _train_model(self):
        features = self.genres.columns.tolist() + ['year']
        self.kmeans = KMeans(n_clusters=120, random_state=38, n_init=10, max_iter=300)
        self.df['cluster'] = self.kmeans.fit_predict(self.df[features])

    def predict(self, mood, primary_genre, secondary_genre, decade):
        mood_to_genre = {
            'happy': ['Comedy', 'Animation', 'Horror', 'Adventure', 'Action'],
            'sad': ['Fantasy', 'Thriller', 'Animation', 'Comedy'],
            'anxious': ['Horror', 'SciFi', 'Action', 'Comedy', 'Fantasy', 'Adventure'],
            'excited': ['Action', 'SciFi', 'Adventure', 'Horror', 'Thriller'],
            'bored': ['Comedy', 'Animation', 'Fantasy', 'Adventure', 'Horror']
        }

        start_year = decade
        end_year = start_year + 10

        filtered_df = self.df[(self.df['year'] >= start_year) & (self.df['year'] < end_year)]
        
        if filtered_df.empty:
            return {"message": "Não há dados disponíveis para a década especificada."}

        def get_clusters(genre):
            return filtered_df[filtered_df[genre] == 1].groupby('cluster').size()

        with ThreadPoolExecutor(max_workers=8) as executor:
            future_primary = executor.submit(get_clusters, primary_genre)
            future_secondary = executor.submit(get_clusters, secondary_genre)
            primary_clusters = future_primary.result()
            secondary_clusters = future_secondary.result()
        
        if primary_clusters.empty or secondary_clusters.empty:
            return {"message": "Não há dados disponíveis para os gêneros favoritos fornecidos na década especificada."}
        
        primary_cluster = primary_clusters.idxmax()
        secondary_cluster = secondary_clusters.idxmax()
        
        recommended_movies = self.df[
            (self.df['cluster'].isin([primary_cluster, secondary_cluster])) & 
            (self.df[mood_to_genre[mood]].sum(axis=1) > 0)
        ]
        
        recommended_movies = recommended_movies.sample(frac=1).reset_index(drop=True)
        
        result = {
            "message": "Predição realizada com sucesso!",
            "data": recommended_movies[['title', 'genres']].to_dict(orient='records')
        }

        return result