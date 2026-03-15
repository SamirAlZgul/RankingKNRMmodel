'''

будем обучать правильно рандировать документы, для этого надо написать все функции по формированию документов
'''
import pandas as pd
import numpy as np
import torch
from pandas import read_csv

import string
import nltk
from collections import Counter
from datasets import load_dataset
glove_vectors_path = "glove.6B/glove.6B.50d.txt"

# импортируем описанную модель KNRM
from main import KNRM

class Solution:
    def __init__(self, glue_qqp_dir,
                 glove_vectors_path,
                 min_occurancies = 1,
                 random_seed = 0,
                 emb_rand_uni_bound = 0.2,
                 freeze_knrm_embeddings = True,
                 knrm_kernel_num = 21,
                 knrm_out_mlp = [],
                 dataloader_bs = 1024,
                 train_lr = 0.001,
                 change_train_loader_ep = 10
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
        self.knrm_kernel_num = knrm_kernel_num
        self.knrm_out_mlp = knrm_out_mlp
        self.dataloader_bs = dataloader_bs
        self.train_lr = train_lr
        self.change_train_loader_ep = change_train_loader_ep
        # получаем модель инициализированную, словарь слов и словарь редких слов, для которых нет эмбеддингов
        self.model, self.vocab, self.unk_words = self._build_knrm_model()

        # получаем id-text датасеты с train и dev
        self.idx_to_text_mapping_train = self._get_idx_to_text_mapping(self.glue_train_df)
        self.idx_to_text_mapping_dev = self._get_idx_to_text_mapping(self.glue_dev_df)

        # создаем датасет из пар (запрос, документ) и соотоветствующая им оценка релевантности
        self.val_dataset = ValPairsDataset(self.dev_pairs_for_ndcg, self.idx_to_text_mapping_dev,
                                           vocab = self.vocab,
                                           oov_val = self.vocab['OOV'],
                                           preproc_func = self._simple_preproc)
        # теперь из валидационного датасета надо сделать даталоадер
        # одновременно с этим запрос и документ надо дополнить до требуемого размера или обрезать
        # за это ответчает функция collate_fn
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.dataloader_bs,\
                                                          num_workers=0, collate_fn = collate_fn, shuffle=False)



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
        #("inp_df_group_sizes", inp_df_group_sizes)
        # теперь выбираем конкретные айдишники текстов с левой колонки - id_left где записей больше 2
        # это нужно, чтобы сделать каждую группу хотя бы минимально представительной
        glue_dev_leftids_to_use = list(inp_df_group_sizes[inp_df_group_sizes >= min_group_size].index)
        #print("glue_dev_leftids_to_use", glue_dev_leftids_to_use)
        # теперь выбираем из изначального датасета только представительные номера, которые обозначили на предыдущем шаге
        groups = inp_df_select[inp_df_select.id_left.isin(glue_dev_leftids_to_use)].groupby('id_left')
        # получаем список всех ids (правый не фильтруем так как id - симметричные)
        all_ids = set(inp_df['id_left']).union(set(inp_df['id_right']))
        #print("all_ids", all_ids)
        np.random.seed(seed)
        out_pairs = []
        pad_sample = []
        #print('groups',groups)
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
        torch.manual_seed(self.random_seed)
        knrm = KNRM(emb_matrix, freeze_embeddings = self.freeze_embeddings, kernel_num = self.knrm_kernel_num,
                    out_layers = self.knrm_out_mlp)
        return knrm, vocab, unk_words


    # надо будет по индексам вынимать тексты
    # создаем словарь чтобы по id могли легко вынимать тексты
    # и тексты справа и тексты с айди слева все в одну строку
    def _get_idx_to_text_mapping(self, inp_df):
        left_dict = (inp_df[['id_left','text_left']].drop_duplicates().set_index('id_left')['text_left'].to_dict())
        right_dict = (inp_df[['id_right','text_right']].drop_duplicates().set_index('id_right')['text_right'].to_dict())
        left_dict.update(right_dict)
        return left_dict




# а этот класс для того чтобы трейн сделать
class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, index_pairs_or_triplets, idx_to_text_mapping, vocab, oov_val, preproc_func, max_len=30):
        self.index_pairs_or_triplets = index_pairs_or_triplets
        self.vocab = vocab
        # значение, если слова нет в словаре
        self.oov_val = oov_val
        self.preproc_func = preproc_func
        self.idx_to_text_mapping = idx_to_text_mapping
        # ограничиваем текст в запросе и документе
        # это ухудшает смыл, но улучшает производительность
        self.max_len = max_len

    def __len__(self):
        return len(self.index_pairs_or_triplets)

    def _tokenized_text_to_index(self, tokenized_text):
        # берем значение по слову i
        # если такого слова нет в словаре, то передаем значение self.oov_val - не в словаре
        res = [self.vocab.get(i, self.oov_val) for i in tokenized_text]
        return res

    def _convert_text_idx_to_token_idxs(self, idx):
        # берем запрос по кокнретному idx
        # предобрабатываем его, токенизируем
        # берем idx, используем функцию idx_to_text_mapping, чтобы поулчить текст
        # далее сам текст предобрабатываем, убираем символы пунктуации и пр.
        # внутри функции preproc_func есть разбитие на токены с помощью nltk
        tokenized_text = self.preproc_func(self.idx_to_text_mapping[idx])
        # далее надо понять какие индексы соотвесттвуют этим токенам в словаре
        idxs = self._tokenized_text_to_index(tokenized_text)
        return idxs

# этот класс копирует трейн класс частично, но используется для создания пар(pair, target)
class ValPairsDataset(RankingDataset):
    def __getitem__(self,idx):
        cur_row = self.index_pairs_or_triplets[idx]
        print(cur_row)
        left_idxs = self._convert_text_idx_to_token_idxs(cur_row[0])[:self.max_len]
        r1_idxs = self._convert_text_idx_to_token_idxs(cur_row[1])[:self.max_len]

        pair = {'query':left_idxs, 'document':r1_idxs}
        target = cur_row[2]
        print(pair, target)
        return (pair, target)



# эта функция отвечает за корректный размер батча для запроса и документа, чтобы было ровно
def collate_fn(batch_obj):
    max_len_q1 = -1
    max_len_d1 = -1
    max_len_q2 = -1
    max_len_d2 = -1
    is_triplets = False

    for elem in batch_obj:
        if len(elem)==3:
            left_elem, right_elem, label = elem
            is_triplets = True
        # если Pointwise - подход (+ на этапе инференса его можно использовать)
        else:
            left_elem, label = elem

        max_len_q1 = max(len(left_elem['query']), max_len_q1)
        max_len_d1 = max(len(left_elem['document']), max_len_d1)
        if len(elem)==3:
            max_len_q2 = max(len(right_elem['query']), max_len_q2)
            max_len_d2 = max(len(right_elem['document']), max_len_d2)
    q1s = []
    q2s = []
    d1s = []
    d2s = []
    labels = []

    for elem in batch_obj:
        if is_triplets:
            left_elem, right_elem,label = elem
            is_triplets = True
        else:
            left_elem, label = elem

        # считаем сколько надо добавить к каждой последовательности
        pad_len1 = max_len_q1 - len(left_elem['query'])
        pad_len2 = max_len_d1 - len(left_elem['document'])
        if is_triplets:
            pad_len3 = max_len_d2 - len(right_elem['document'])
            pad_len4 = max_len_q2 - len(right_elem['query'])

        q1s.append(left_elem['query']+[0]*pad_len1)
        d1s.append(left_elem['document']+[0]*pad_len2)
        if is_triplets:
            q2s.append(right_elem['query']+[0]*pad_len3)
            d2s.append(right_elem['document']+[0]*pad_len4)
        labels.append(label)

    # переводим в тензора
    q1s = torch.LongTensor(q1s)
    d1s = torch.LongTensor(d1s)
    if is_triplets:
        q2s = torch.LongTensor(q2s)
        d2s = torch.LongTensor(d2s)
    labels = torch.LongTensor(labels)

    res_left = {'query':q1s, 'document':d1s}
    if is_triplets:
        res_right = {'query':q2s, 'document':d2s}
        return res_left, res_right, labels
    else:
        return res_left, labels










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


knrm, vocab, unk_words = sol._build_knrm_model()

print(f"Модель KNRM форма: {knrm}")
print(f"Размер словаря: {len(vocab)}")
print(f"Количество неизвестных слов: {len(unk_words)}")
if len(unk_words) > 0:
    print(f"Примеры неизвестных слов: {unk_words[:10]}")
print(f"Пример словаря (первые 5 слов): {list(vocab.items())[:5]}")

print("\n" + "="*50)
print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
print("="*50)

# ============= НОВЫЙ КОД ДЛЯ ПРОВЕРКИ val_dataset =============
print("\n" + "=" * 70)
print("ПРОВЕРКА ValPairsDataset")
print("=" * 70)

# Проверяем, что val_dataset создался
print(f"\nТип val_dataset: {type(sol.val_dataset)}")
print(f"Размер val_dataset: {len(sol.val_dataset)}")

# Выводим первые 5 элементов из dev_pairs_for_ndcg (сырые данные)
print(f"\nПервые 5 пар из dev_pairs_for_ndcg:")
for i, pair in enumerate(sol.dev_pairs_for_ndcg[:5]):
    print(f"  {i + 1}. {pair}")

# Проверяем работу __getitem__ для первых нескольких элементов
print(f"\nПервые 5 элементов из val_dataset (после преобразования):")
for i in range(min(5, len(sol.val_dataset))):
    print(f"\n--- Элемент {i} ---")
    pair, target = sol.val_dataset[i]
    print(f"  Query (индексы): {pair['query']}")
    print(f"  Document (индексы): {pair['document']}")
    print(f"  Target: {target}")

    # Декодируем обратно в текст для проверки (опционально)
    # Создаем обратный словарь: индекс -> слово
    idx_to_word = {idx: word for word, idx in sol.vocab.items()}

    query_words = [idx_to_word.get(idx, '<UNK>') for idx in pair['query']]
    doc_words = [idx_to_word.get(idx, '<UNK>') for idx in pair['document']]

    print(f"  Query (слова): {query_words}")
    print(f"  Document (слова): {doc_words}")

# ============= ПРОВЕРКА collate_fn =============
print("\n" + "=" * 70)
print("ПРОВЕРКА collate_fn")
print("=" * 70)

# Берем первые 2 элемента из val_dataset вручную
print("\n1. Сначала проверим отдельные элементы val_dataset:")
sample_elements = []
for i in range(2):
    pair, target = sol.val_dataset[i]
    sample_elements.append((pair, target))
    print(f"\nЭлемент {i}:")
    print(f"  Query (длина {len(pair['query'])}): {pair['query']}")
    print(f"  Document (длина {len(pair['document'])}): {pair['document']}")
    print(f"  Target: {target}")

# Теперь применяем collate_fn к этим двум элементам
print("\n" + "-" * 50)
print("2. Применяем collate_fn к батчу из 2 элементов:")
batch = sample_elements  # это список из (pair, target)
result = collate_fn(batch)

print(f"\nТип результата: {type(result)}")
print(f"Длина результата: {len(result)}")

# Распаковываем результат
if len(result) == 2:  # pointwise режим
    left_batch, labels_batch = result
    print("\nРежим: POINTWISE")
    print(f"\nleft_batch keys: {left_batch.keys()}")
    print(f"query tensor shape: {left_batch['query'].shape}")
    print(f"document tensor shape: {left_batch['document'].shape}")
    print(f"labels tensor shape: {labels_batch.shape}")

    print(f"\nquery tensor (первые 2 строки):\n{left_batch['query'][:2]}")
    print(f"\ndocument tensor (первые 2 строки):\n{left_batch['document'][:2]}")
    print(f"\nlabels tensor: {labels_batch}")

else:  # triplet режим (хотя у нас val_dataset с 2 элементами)
    left_batch, right_batch, labels_batch = result
    print("\nРежим: TRIPLET")
    print(f"left_batch keys: {left_batch.keys()}")
    print(f"right_batch keys: {right_batch.keys()}")
    print(f"left query shape: {left_batch['query'].shape}")
    print(f"left document shape: {left_batch['document'].shape}")
    print(f"right query shape: {right_batch['query'].shape}")
    print(f"right document shape: {right_batch['document'].shape}")
    print(f"labels shape: {labels_batch.shape}")

# Проверяем, что все последовательности выровнены до одной длины
print("\n" + "-" * 50)
print("3. Проверка выравнивания (padding):")

if len(result) == 2:
    # Для pointwise: проверяем длины query и document
    query_lens = [len(sample_elements[i][0]['query']) for i in range(2)]
    doc_lens = [len(sample_elements[i][0]['document']) for i in range(2)]

    print(f"Исходные длины query: {query_lens}")
    print(f"Исходные длины document: {doc_lens}")
    print(f"После padding - query shape: {left_batch['query'].shape}")
    print(f"После padding - document shape: {left_batch['document'].shape}")

    # Проверяем, что паддинг нулями работает
    max_query_len = max(query_lens)
    max_doc_len = max(doc_lens)

    print(f"\nПроверка первой строки query (должна быть длина {max_query_len}):")
    print(f"  Было: {sample_elements[0][0]['query']}")
    print(f"  Стало: {left_batch['query'][0].tolist()}")
    print(f"  Добавлено паддингов: {left_batch['query'][0].tolist()[query_lens[0]:]}")

    print(f"\nПроверка второй строки query (должна быть длина {max_query_len}):")
    print(f"  Было: {sample_elements[1][0]['query']}")
    print(f"  Стало: {left_batch['query'][1].tolist()}")
    print(f"  Добавлено паддингов: {left_batch['query'][1].tolist()[query_lens[1]:]}")

# Проверяем через dataloader
print("\n" + "-" * 50)
print("4. Проверка через val_dataloader (первые 2 батча):")

# Берем первые 2 батча из dataloader
for batch_idx, batch_data in enumerate(sol.val_dataloader):
    if batch_idx >= 2:  # только первые 2 батча
        break

    print(f"\n--- Батч {batch_idx + 1} ---")

    if len(batch_data) == 2:
        left_batch, labels_batch = batch_data
        print(f"Тип: POINTWISE")
        print(f"Query shape: {left_batch['query'].shape}")
        print(f"Document shape: {left_batch['document'].shape}")
        print(f"Labels shape: {labels_batch.shape}")
        print(f"Labels: {labels_batch}")

        # Проверяем первые 2 элемента в батче
        print(f"\nПервые 2 query в батче (первые 10 токенов):")
        for i in range(min(2, left_batch['query'].shape[0])):
            print(f"  Query {i}: {left_batch['query'][i][:10].tolist()}...")
            print(f"  Doc {i}: {left_batch['document'][i][:10].tolist()}...")
    else:
        left_batch, right_batch, labels_batch = batch_data
        print(f"Тип: TRIPLET")
        print(f"Left query shape: {left_batch['query'].shape}")
        print(f"Right query shape: {right_batch['query'].shape}")
        print(f"Labels shape: {labels_batch.shape}")

print("\n" + "=" * 70)
print("ПРОВЕРКА ЗАВЕРШЕНА")
print("=" * 70)