import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import numpy as np

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
        self.kmeans = KMeans(n_clusters=500, random_state=5)
        self.df['cluster'] = self.kmeans.fit_predict(self.df[features])

    def predict(self, mood, primary_genre, secondary_genre, decade, top_n_clusters=5):
        # Realiza a predição de filmes baseados no humor, gêneros primário e secundário, e década.
        # O parâmetro top_n_clusters define quantos clusters devem ser considerados.

        # Mapeamento de humor para gênero
        mood_to_genre = {
            'happy': ['Comedy', 'Animation', 'Horror', 'Adventure', 'Action'],
            'sad': ['Fantasy', 'Thriller', 'Animation', 'Comedy'],
            'anxious': ['Thriller', 'Horror', 'SciFi', 'Action'],
            'excited': ['Action', 'SciFi', 'Adventure', 'Horror', 'Thriller'],
            'bored': ['Comedy', 'Fantasy', 'Adventure', 'Horror']
        }

        # Definir intervalo de anos
        start_year = decade
        end_year = start_year + 10

        # Filtrar dataframe pela década
        filtered_df = self.df[(self.df['year'] >= start_year) & (self.df['year'] <= end_year)]
        
        if filtered_df.empty:
            return {"message": "Não há dados disponíveis para a década especificada."}
        
        # Encontrar os n clusters mais frequentes para o gênero primário
        primary_cluster_counts = filtered_df[filtered_df[primary_genre] == 1].groupby('cluster').size()
        primary_clusters = primary_cluster_counts.nlargest(top_n_clusters).index.tolist()
        
        # Encontrar os n clusters mais frequentes para o gênero secundário
        secondary_cluster_counts = filtered_df[filtered_df[secondary_genre] == 1].groupby('cluster').size()
        secondary_clusters = secondary_cluster_counts.nlargest(top_n_clusters).index.tolist()
        
        # Combinar os clusters primários e secundários
        selected_clusters = list(set(primary_clusters + secondary_clusters))
        
        if not selected_clusters:
            return {"message": "Não há dados disponíveis para os gêneros favoritos fornecidos na década especificada."}
        
        # Filtrar filmes recomendados com base nos clusters selecionados
        recommended_movies = self.df[
            (self.df['cluster'].isin(selected_clusters)) & 
            (self.df[mood_to_genre[mood]].sum(axis=1) > 0)
        ]
        
        if recommended_movies.empty:
            return {"message": "Nenhum filme encontrado para a combinação fornecida."}

        # Adicionar aleatoriedade à seleção dos filmes recomendados
        recommended_movies = recommended_movies.sample(frac=1, random_state=np.random.randint(0, 1000))

        # Retornar filmes recomendados
        return {
            "message": "Predição realizada com sucesso!",
            "data": recommended_movies[['title', 'genres']].to_dict(orient='records')
        }
