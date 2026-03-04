import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter
import random


class SimpleBagOfWords:
    def __init__(self):
        self.vocabulary = {}
        self.word_index = {}
        self.vocab_size = 0

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^а-яёa-z0-9_]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def fit(self, texts):
        all_words = []
        for text in texts:
            processed_text = self.preprocess_text(text)
            words = processed_text.split()
            all_words.extend(words)
        unique_words = sorted(set(all_words), key=lambda x: x)
        self.vocab_size = len(unique_words)
        self.vocabulary = {word: idx for idx, word in enumerate(unique_words)}
        self.word_index = {idx: word for idx, word in enumerate(unique_words)}
        return self

    def transform(self, texts):
        vectors = []
        for text in texts:
            processed_text = self.preprocess_text(text)
            words = processed_text.split()
            vector = [0] * self.vocab_size
            for word in words:
                if word in self.vocabulary:
                    vector[self.vocabulary[word]] += 1
            vectors.append(vector)
        return np.array(vectors, dtype=float)

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)

    def get_feature_names(self):
        return [self.word_index[i] for i in range(self.vocab_size)]


# Тестовые данные
test_texts = [
    "собака укусила человека",
    "человек укусил собаку",
    "стреляю из лука",
    "плачу от лука",
    "люблю смотреть фильмы",
    "фильмы смотреть люблю"
]

bow_model = SimpleBagOfWords()
vectors = bow_model.fit_transform(test_texts)

print("Словарь:", bow_model.get_feature_names())
print("Размер словаря:", bow_model.vocab_size)
print("\nВекторы BoW:")
for i, text in enumerate(test_texts):
    print(f"'{text}': {vectors[i].astype(int)}")

eq_order = np.array_equal(vectors[0], vectors[1])
print(f"\nВекторы для 'собака укусила человека' и 'человек укусил собаку' равны? {eq_order}")
print("Слова в обеих фразах про 'лук':",
      set(bow_model.preprocess_text(test_texts[2]).split()) & set(bow_model.preprocess_text(test_texts[3]).split())
      )


# Визуализация BoW
def visualize_bow_vectors(texts, vectors, title="Визуализация BoW-векторов"):
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)

    plt.figure(figsize=(10, 6))
    colors = ['red', 'red', 'blue', 'blue', 'green', 'green']

    for i, (text, color) in enumerate(zip(texts, colors)):
        x, y = vectors_2d[i, 0], vectors_2d[i, 1]
        plt.scatter(x, y, marker='o', s=100, edgecolor='k', facecolor=color)
        short_text = text if len(text) <= 20 else text[:18] + '...'
        plt.annotate(short_text, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=9)

    plt.xlabel('Компонента PCA 1')
    plt.ylabel('Компонента PCA 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Укусы'),
        Patch(facecolor='blue', label='Лук'),
        Patch(facecolor='green', label='Фильмы')
    ]
    plt.legend(handles=legend_elements)
    plt.show()


visualize_bow_vectors(test_texts, vectors, "BoW-векторы (PCA) — демонстрация ограничений")

# Word2Vec часть: пробуем загрузить модель, иначе синтетика
try:
    import gensim.downloader as api

    print("\nПробуем загрузить предобученную модель word2vec-ruscorpora-300...")
    word2vec_model = api.load('word2vec-ruscorpora-300')
    print("Модель загружена.")
    real_model = True
except Exception as e:
    print("\nНе удалось загрузить модель (или gensim недоступен). Будем использовать синтетические эмбеддинги.")
    word2vec_model = None
    real_model = False

word_groups = {
    'nouns_people': ['человек_NOUN', 'люди_NOUN', 'личность_NOUN', 'персона_NOUN'],
    'nouns_time': ['время_NOUN', 'год_NOUN', 'день_NOUN', 'месяц_NOUN'],
    'verbs_action': ['говорить_VERB', 'сказать_VERB', 'молчать_VERB', 'кричать_VERB'],
    'adjectives_size': ['большой_ADJ', 'маленький_ADJ', 'огромный_ADJ', 'крошечный_ADJ'],
    'adjectives_quality': ['хороший_ADJ', 'плохой_ADJ', 'отличный_ADJ', 'ужасный_ADJ']
}


def get_vectors_for_groups(model, groups, dim=50, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    all_words = []
    all_vectors = []
    labels = []
    if model is not None:
        for gname, words in groups.items():
            for w in words:
                if w in model:
                    vec = model[w]
                else:
                    vec = np.random.randn(model.vector_size)
                all_words.append(w)
                all_vectors.append(vec)
                labels.append(gname)
        return all_words, np.vstack(all_vectors), labels
    else:
        centers = {}
        for i, gname in enumerate(groups.keys()):
            centers[gname] = np.random.randn(dim) * 3
        for gname, words in groups.items():
            for w in words:
                vec = centers[gname] + np.random.randn(dim) * 0.5
                all_words.append(w)
                all_vectors.append(vec)
                labels.append(gname)
        return all_words, np.vstack(all_vectors), labels


words, all_vectors, labels = get_vectors_for_groups(word2vec_model, word_groups, dim=50)

perp = min(30, max(5, len(words) // 3))
tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto')
vectors_2d = tsne.fit_transform(all_vectors)

plt.figure(figsize=(12, 8))
group_to_color = {
    'nouns_people': 'red',
    'nouns_time': 'blue',
    'verbs_action': 'green',
    'adjectives_size': 'orange',
    'adjectives_quality': 'purple'
}
for i, w in enumerate(words):
    g = labels[i]
    x, y = vectors_2d[i, 0], vectors_2d[i, 1]
    plt.scatter(x, y, s=120, color=group_to_color[g], edgecolor='k', alpha=0.9)
    plt.annotate(w, (x, y), textcoords='offset points', xytext=(5, 5), fontsize=10, fontweight='bold')

plt.title("Семантическое пространство (t-SNE)")
plt.grid(alpha=0.2)
from matplotlib.patches import Patch

legend_elements = [Patch(facecolor=c, label=k) for k, c in group_to_color.items()]
plt.legend(handles=legend_elements)
plt.show()


# Демонстрация векторной арифметики (если модель загружена)
def demonstrate_vector_arithmetic_with_tags(model):
    if model is None:
        print("\nРеальная модель отсутствует — пропускаем демонстрацию точных most_similar.")
        return
    examples = [
        (['король_NOUN', 'женщина_NOUN'], ['мужчина_NOUN'], 'королева_NOUN'),
        (['отец_NOUN', 'дочь_NOUN'], ['сын_NOUN'], 'мать_NOUN'),
        (['большой_ADJ', 'хороший_ADJ'], ['маленький_ADJ'], 'плохой_ADJ'),
        (['бежать_VERB', 'стоять_VERB'], ['стоять_VERB'], 'бежать_VERB'),
        (['день_NOUN', 'ночь_NOUN'], ['ночь_NOUN'], 'день_NOUN')
    ]
    for pos, neg, expected in examples:
        try:
            res = model.most_similar(positive=pos, negative=neg, topn=5)
            print(f"\nПоложит: {pos}, отрицат: {neg} -> Ожидалось: {expected}. Топ-5:")
            for word, score in res:
                print(f"  {word}: {score:.4f}")
        except Exception as e:
            print(f"Ошибка при обработке {pos} - {neg}: {e}")


demonstrate_vector_arithmetic_with_tags(word2vec_model)


def find_similar_words_with_tags(model, word, topn=8):
    print(f"\nСлова, близкие к '{word}':")
    if model is None:
        print("  (модель отсутствует — невозможно найти реальные похожие слова)")
        return
    try:
        res = model.most_similar(positive=[word], topn=topn)
        for w, sc in res:
            print(f"  {w}: {sc:.4f}")
    except Exception as e:
        print(f" Ошибка при поиске похожих слов: {e}")


find_similar_words_with_tags(word2vec_model, 'человек_NOUN')
find_similar_words_with_tags(word2vec_model, 'говорить_VERB')
find_similar_words_with_tags(word2vec_model, 'большой_ADJ')
find_similar_words_with_tags(word2vec_model, 'хороший_ADJ')
