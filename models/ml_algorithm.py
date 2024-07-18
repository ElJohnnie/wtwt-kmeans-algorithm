import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans

class MLEngine:
    def __init__(self, csv_path='data/v1.csv'):
        self.df = pd.read_csv(csv_path)
        self._preprocess_data()
        self._train_model()
    
    def _preprocess_data(self):
        # Extrair ano dos títulos
        self.df['year'] = self.df['title'].str.extract(r'\((\d{4})\)')
        self.df['year'] = pd.to_numeric(self.df['year'], errors='coerce')
        self.df = self.df.dropna(subset=['year'])
        self.df['year'] = self.df['year'].astype(int)

        # One-hot encoding dos gêneros
        self.genres = self.df['genres'].str.get_dummies(sep='|')
        self.df = self.df.join(self.genres)

    def _train_model(self):
        # Treina o modelo KMeans usando os gêneros e o ano como características.
        features = self.genres.columns.tolist() + ['year']
        self.kmeans = KMeans(n_clusters=100)
        self.df['cluster'] = self.kmeans.fit_predict(self.df[features])

    def predict(self, mood, primary_genre, secondary_genre, decade):
        # Realiza a predição de filmes baseados no humor, gêneros primário e secundário, e década.
        # Mapeamento de humor para gênero
        mood_to_genre = {
            'happy': 'Comedy',
            'sad': 'Drama',
            'anxious': 'Thriller',
            'excited': 'Action',
            'bored': 'Adventure'
        }

        # Definir intervalo de anos
        start_year = decade
        end_year = start_year + 10

        # Filtrar dataframe pela década
        filtered_df = self.df[(self.df['year'] >= start_year) & (self.df['year'] <= end_year)]
        
        if filtered_df.empty:
            return {"message": "Não há dados disponíveis para a década especificada."}
        
        # Encontrar os clusters primário e secundário mais frequentes
        primary_cluster = filtered_df[(filtered_df[primary_genre] == 1)].groupby('cluster').size()
        secondary_cluster = filtered_df[(filtered_df[secondary_genre] == 1)].groupby('cluster').size()
        
        if primary_cluster.empty or secondary_cluster.empty:
            return {"message": "Não há dados disponíveis para os gêneros favoritos fornecidos na década especificada."}
        
        # Selecionar os clusters mais frequentes
        primary_cluster = primary_cluster.idxmax()
        secondary_cluster = secondary_cluster.idxmax()
        
        # Filtrar filmes recomendados
        recommended_movies = self.df[(self.df['cluster'].isin([primary_cluster, secondary_cluster])) & 
                                     (self.df[mood_to_genre[mood]] == 1)]
        
        # Retornar filmes recomendados
        return {
            "message": "Predição realizada com sucesso!",
            "data": recommended_movies[['title', 'genres']].to_dict(orient='records')
        }
