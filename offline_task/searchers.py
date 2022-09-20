import numpy as np
from sklearn.neighbors import NearestNeighbors, DistanceMetric
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine

embs_tree = NearestNeighbors(algorithm="kd_tree", leaf_size=30, n_neighbors=5)

class Searcher:
    def __init__(self, embeddings: np.ndarray, embedding_tree=embs_tree):
        # Эмбеддинги всех слов
        self.embeddings = embeddings
        self.embedding_tree = embedding_tree
        self.embedding_tree.fit(self.embeddings)

    def find_closest(self, toxic_embedding: np.ndarray, sentence_word_embeddings: np.ndarray) -> int:
        # Достаём n_neighbors ближайших слов (индексы ближайших векторов)
        query = self.embedding_tree.kneighbors(toxic_embedding)
        nontoxic_indexes = query[1][0]
        # Вектора нетоксичных слов
        nontoxic_embeddings = self.embeddings[nontoxic_indexes]
        # Создаем массив сходств предложеных слов с нашим словом
        nontoxic_distances = np.zeros((len(nontoxic_embeddings)))
        # Заполним массив nontoxic_distances
        for i, nontoxic_embedding in enumerate(nontoxic_embeddings):
            sim = self.calculate_similarity(nontoxic_embedding, toxic_embedding)
            print(sim)
            nontoxic_distances[i] = sim

        # Массив средней НЕПОХОЖЕСТИ нетоксичного вектора на все слова из предложения
        nontoxic_sentence_similariries = np.array([])

        for nontoxic_embedding in nontoxic_embeddings:
            # Массив непохожести нетоксичного слова на все остальные слова из предложения
            sentences_sims = np.zeros((sentence_word_embeddings.shape[0]))
            for ind, word_embedding in enumerate(sentence_word_embeddings):
                # Поскольку нам надо найти самые НЕПОХОЖИЕ слова на остальные предложения, то вычитаем схожесть из единицы
                similarity = 1 - self.calculate_similarity(nontoxic_embedding, word_embedding)
                sentences_sims[ind] = similarity
            # Считаем среднюю НЕСХОЖЕСТЬ
            nontoxic_sentence_similarity = sentences_sims.mean()
            nontoxic_sentence_similariries = np.append(nontoxic_sentence_similariries, nontoxic_sentence_similarity)

        # Считаем для каждого нетоксичного слова среднее между  схожестью с токсичным словом и несхожестью со словами из предлодения
        result_similarity = (nontoxic_distances + nontoxic_sentence_similariries) / 2
        # Выбираем слово с максимальной метрикой
        max_index = nontoxic_indexes[np.argmax(result_similarity)]
        return max_index

    def calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        print(vec1)
        sim = cosine(vec1, vec2)
        return sim