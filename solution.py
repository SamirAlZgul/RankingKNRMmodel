'''

будем обучать правильно рандировать документы, для этого надо написать все функции по формированию документов
'''
import pandas as pd
import numpy as np
from pandas import read_csv

import string
import nltk
from collections import Counter
from datasets import load_dataset
glove_vectors_path = "glove.6B/glove.6B.50d.txt"

class Solution:
    def __init__(self, glue_qqp_dir,
                 glove_vectors_path,
                 min_occurancies = 1,
                 random_seed = 0,
                 emb_rand_uni_bound = 0.2,
                 freeze_knrm_embeddings = True,
                 ):


        # путь до папки с трейном и тестом
        self.glue_qqp_dir = glue_qqp_dir
        # путь до файла с эмбеддингами
        self.glove_vectors_path = glove_vectors_path
        # надо взять отдельно трейн и тест
        self.glue_train_df = self._get_glue_df('train')
        self.glue_dev_df = self._get_glue_df('dev')
        # теперь из этого надо создать валидационный набор
        self.dev_pairs_for_ndcg = self._create_val_pairs(self.glue_dev_df)

        self.min_occurancies = min_occurancies
        self.all_tokens = self._get_all_tokens([self.glue_train_df,self.glue_dev_df], self.min_occurancies)

        self.random_seed = random_seed
        self.emb_rand_uni_bound = emb_rand_uni_bound
        self.freeze_embeddings = freeze_knrm_embeddings


    # напишем функцию, которая читает трейн и text-файлы c лейблами
    def _get_glue_df(self, partition_type):
        assert partition_type in ['train', 'dev']
        glue_df = read_csv(self.glue_qqp_dir+f'{partition_type}.tsv', sep='\t', dtype='object')
        glue_df = glue_df.dropna(axis = 0, how='any').reset_index(drop=True)
        glue_df_fin = pd.DataFrame({'id_left':glue_df['qid1'],
                                    'id_right':glue_df['qid2'],
                                    'text_left':glue_df['question1'],
                                    'text_right':glue_df['question2'],
                                    'label':glue_df['is_duplicate']})
        return glue_df_fin

    # напишем функцию, которая формирует набор, на ктором будем валидировать обученную модель
    def _create_val_pairs(self, inp_df, fill_top_to=15, min_group_size=1, seed = 0):
        # на вход приходит датасет inp_df
        inp_df['label'] = inp_df['label'].astype('int64')
        # выбираем айдишники и лейблы - мы всегда восстановим по айдишникам сами текста
        inp_df_select = inp_df[['id_left','id_right','label']]
        # смотрим сколько вообще сэмплов в каждой id-группе
        inp_df_group_sizes = inp_df_select.groupby('id_left').size()
        print("inp_df_group_sizes", inp_df_group_sizes)
        # теперь выбираем конкретные айдишники текстов с левой колонки - id_left где записей больше 2
        # это нужно, чтобы сделать каждую группу хотя бы минимально представительной
        glue_dev_leftids_to_use = list(inp_df_group_sizes[inp_df_group_sizes >= min_group_size].index)
        print("glue_dev_leftids_to_use", glue_dev_leftids_to_use)
        # теперь выбираем из изначального датасета только представительные номера, которые обозначили на предыдущем шаге
        groups = inp_df_select[inp_df_select.id_left.isin(glue_dev_leftids_to_use)].groupby('id_left')
        # получаем список всех ids (правый не фильтруем так как id - симметричные)
        all_ids = set(inp_df['id_left']).union(set(inp_df['id_right']))
        #print("all_ids", all_ids)
        np.random.seed(seed)
        out_pairs = []
        pad_sample = []
        print('groups',groups)
        for id_left, group in groups:
            # для каждого id_left выбираем все пары релевантных документов
            ones_ids = group[group.label>0].id_right.values
            # и все пары нерелеватных документов
            zeroes_ids = group[group.label==0].id_left.values
            # теперь смотрим их количество
            sum_len = len(ones_ids)+len(zeroes_ids)
            # теперь смотрим, сколько пар в каждую группу надо добавить для того ,чтобы избежать дисбаланса в группах
            # то есть чтобы везде было по 15 сэмплов
            # это нужно для стабильности обучения, бат-обработки и т д
            num_pad_items = max(0, fill_top_to-sum_len)
            if num_pad_items>0:
                # если к текущей группе нужно добавить сколько то семплов до 15
                # то выбираем элементы которые надо добавить
                # это вот все которые на текущем шаге выбраны - их нельзя боать
                cur_chosen = set(ones_ids).union(set(zeroes_ids)).union(id_left)
                # из остальных рандомно добираем пары ids
                pad_sample = list(np.random.choice(list(all_ids-cur_chosen), num_pad_items, replace=False))
            else:
                pad_sample=[]
            # теперь мы получили для конкретного id_left все 15 пар - надо семплировать
            for i in ones_ids:
                out_pairs.append([id_left,i,2])
            for i in zeroes_ids:
                out_pairs.append([id_left,i,1])
            for i in pad_sample:
                out_pairs.append([id_left,i,0])
        return out_pairs

    #напишем функцию, которая правильно фильтрует полученные датасеты: убираем знаки препинания, формирует нижний
    #регистр и пр.
    def _get_all_tokens(self, list_of_df, min_occurancies):
        tokens=[]
        # проходимся по каждому датасету
        for df in list_of_df:
            unique_texts = set(df[['text_left','text_right']].values.reshape(-1))
            # склеиваем все тексты в одну строку
            unique_texts = str(' '.join(unique_texts))
            # предобрабатывваем ве тексты и токенизируем их
            df_tokens = self._simple_preproc(unique_texts)
            tokens.extend(df_tokens)
        # далее из общего списка, надо убрать редкие слова, чтобы модель не зацикливалась на них
        count_filtered = self._filter_rare_words(Counter(tokens), min_occurancies)
        # поулчаем словарь с убранными редкими токенами, которые встречаются реже, чем min_occurancies
        return list(count_filtered.keys())

    # здесь функция которая для каждого текста убираем лишние пробелы и вызывает функцию удаления знаков пунктуации
    # и далее токенизирует слова
    def _simple_preproc(self, inp_str):
        base_str = inp_str.strip().lower()
        str_wo_punct = self._handle_punctuation(base_str)
        return nltk.word_tokenize(str_wo_punct)

    # функция, которая убирает все знаки пунктуации
    def _handle_punctuation(self, inp_str):
        inp_str = str(inp_str)
        for punct in string.punctuation:
            inp_str = inp_str.replace(punct,'')
        return inp_str

    # функция, которая убирает редкие слова проходясь по словарю
    def _filter_rare_words(self, vocab, min_occurancies):
        out_vocab = dict()
        for word, cnt in vocab.items():
            if cnt>min_occurancies:
                out_vocab[word] = cnt
        return out_vocab


    def _read_glove_embeddings(self,file_path):
        embedding_data = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                current_line = line.rstrip().split(' ')
                embedding_data[current_line[0]] = current_line[1:]
        return embedding_data


    def _create_glove_emb_from_file(self, file_path, inner_keys, random_seed, rand_uni_bound):
        glove_emb = self._read_glove_embeddings(file_path)
        inner_keys = ['PAD', 'OOV'] + inner_keys
        # коиличество строк в матрице = количество слов
        high_matrix = len(inner_keys)
        # размерность эмбеддинга
        len_matrix = len(list(glove_emb.values())[0])
        print(high_matrix)
        print(len_matrix)

        matrix = np.empty((high_matrix, len_matrix))
        vocab = dict()
        unk_words = []
        np.random.seed(random_seed)
        # проходимся по словам
        for idx, word in enumerate(inner_keys):
            vocab[word] = idx
            if word in glove_emb:
                matrix[idx] = glove_emb[word]
            else:
                unk_words.append(word)
                matrix[idx] = np.random.uniform(-rand_uni_bound, rand_uni_bound, size=len_matrix)
        return matrix, vocab, unk_words



    def _build_knrm_model(self):
        emb_matrix, vocab, unk_words = self._create_glove_emb_from_file(self.glove_vectors_path,
                                                                       self.all_tokens, self.random_seed,
                                                                       self.emb_rand_uni_bound)
        return emb_matrix, vocab, unk_words













#
# #TESTING...
# dataset = load_dataset("nyu-mll/glue", "qqp")
#
#
# def save_glue_format(split, df, filename):
#     # Переименовываем колонки в оригинальный GLUE формат
#     df_glue = pd.DataFrame({
#         'qid1': df['question1'].index if 'qid1' not in df.columns else df['qid1'],
#         'qid2': df['question2'].index if 'qid2' not in df.columns else df['qid2'],
#         'question1': df['question1'],
#         'question2': df['question2'],
#         'is_duplicate': df['label']
#     })
#     df_glue.to_csv(filename, sep='\t', index=False)
#
# # Сохраняем train и dev
# save_glue_format('train', dataset["train"].to_pandas(), './train.tsv')
# save_glue_format('dev', dataset["validation"].to_pandas(), './dev.tsv')

# try:
#     # Пробуем новый вариант
#     nltk.data.find('tokenizers/punkt_tab')
#     print("Ресурс punkt_tab уже загружен")
# except LookupError:
#     print("Загружаем punkt_tab...")
#     nltk.download('punkt_tab')
#
# # Создаем объект, но подменяем атрибуты после создания
# sol = Solution.__new__(Solution)  # создаем объект без вызова __init__
# sol.glue_qqp_dir = None
# sol.min_occurancies = 1
#
# # Теперь вручную создаем тестовые датафреймы
# test_df = pd.DataFrame({
#     'text_left': [
#         'Hello world!',           # слово Hello встречается
#         'Hello Python!',           # слово Hello встречается снова
#         'Python programming'       # слово Python встречается снова
#     ],
#     'text_right': [
#         'Hello world again!',      # Hello и world
#         'Python is great',
#         'Python'                 # Python
#     ]
# })

# Присваиваем тестовые датафреймы
# sol.glue_train_df = test_df
# sol.glue_dev_df = test_df.copy()
#
# # Теперь можно тестировать методы
# print("="*60)
# print("ТЕСТИРОВАНИЕ НА МАЛЕНЬКОМ ДАТАСЕТЕ")
# print("="*60)

# # Тестируем _get_all_tokens
# tokens_1 = sol._get_all_tokens([sol.glue_train_df], min_occurancies=2)
# print(f"\nmin_occurancies=1: {sorted(tokens_1)}")

#TESTING get GLOVE embeddings
#TESTING get GLOVE embeddings
# sol = Solution.__new__(Solution)  # создаем объект без вызова __init__
# sol.glue_qqp_dir = None
# sol.min_occurancies = 1
#
# # Создаем тестовые данные
# test_tokens = ['hello', 'world', 'python', 'programming']

sol = Solution(
    glue_qqp_dir="glue_qqp/",  # путь к папке с данными GLUE QQP
    glove_vectors_path="glove.6B/glove.6B.50d.txt",  # путь к GloVe эмбеддингам
    min_occurancies=1,
    random_seed=0,
    emb_rand_uni_bound=0.2,
    freeze_knrm_embeddings=True
)

print("sol", sol)


emb_matrix, vocab, unk_words = sol._build_knrm_model()

print(f"Форма матрицы эмбеддингов: {emb_matrix.shape}")
print(f"Размер словаря: {len(vocab)}")
print(f"Количество неизвестных слов: {len(unk_words)}")
if len(unk_words) > 0:
    print(f"Примеры неизвестных слов: {unk_words[:10]}")
print(f"Пример словаря (первые 5 слов): {list(vocab.items())[:5]}")
print(f"Тип данных матрицы: {emb_matrix.dtype}")
print(f"Несколько значений из матрицы: {emb_matrix[0, :5]}")

print("\n" + "="*50)
print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
print("="*50)