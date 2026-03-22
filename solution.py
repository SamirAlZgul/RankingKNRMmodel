import pandas as pd
import numpy as np
import torch
import math
from pandas import read_csv
import string
import nltk
from collections import Counter
import sys
import random


# Set random seeds early
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Set seed before any operations
set_seeds(0)

glove_vectors_path = "glove.6B/glove.6B.50d.txt"


# Custom Sampler that doesn't use numpy
class CustomRandomSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, generator=None):
        self.data_source = data_source
        self.generator = generator

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        if self.generator is not None:
            # Use PyTorch generator to shuffle
            for i in range(len(indices) - 1, 0, -1):
                j = torch.randint(0, i + 1, (1,), generator=self.generator).item()
                indices[i], indices[j] = indices[j], indices[i]
        else:
            random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.data_source)


class Solution:
    def __init__(self, glue_qqp_dir,
                 glove_vectors_path,
                 min_occurancies=1,
                 random_seed=0,
                 emb_rand_uni_bound=0.2,
                 freeze_knrm_embeddings=True,
                 knrm_kernel_num=21,
                 knrm_out_mlp=[],
                 dataloader_bs=1024,
                 train_lr=0.001,
                 change_train_loader_ep=10
                 ):

        # Set random seeds for reproducibility
        set_seeds(random_seed)

        # путь до папки с трейном и тестом
        self.glue_qqp_dir = glue_qqp_dir
        # путь до файла с эмбеддингами
        self.glove_vectors_path = glove_vectors_path
        # надо взять отдельно трейн и тест
        self.glue_train_df = self._get_glue_df('train')
        self.glue_dev_df = self._get_glue_df('dev')

        # Debug data structure
        self.debug_data_structure()

        # Проверка, что данные не пустые
        if len(self.glue_train_df) == 0:
            raise ValueError("Training data is empty!")
        if len(self.glue_dev_df) == 0:
            raise ValueError("Development data is empty!")

        # теперь из этого надо создать валидационный набор
        self.dev_pairs_for_ndcg = self._create_val_pairs(self.glue_dev_df)

        self.min_occurancies = min_occurancies
        self.all_tokens = self._get_all_tokens([self.glue_train_df, self.glue_dev_df], self.min_occurancies)

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

        # создаем датасет из пар (запрос, документ) и соответствующая им оценка релевантности
        if len(self.dev_pairs_for_ndcg) > 0:
            self.val_dataset = ValPairsDataset(self.dev_pairs_for_ndcg, self.idx_to_text_mapping_dev,
                                               vocab=self.vocab,
                                               oov_val=self.vocab['OOV'],
                                               preproc_func=self._simple_preproc,
                                               max_len=30)
            # Use shuffle=False to avoid RandomSampler
            self.val_dataloader = torch.utils.data.DataLoader(
                self.val_dataset, batch_size=self.dataloader_bs,
                num_workers=0, collate_fn=collate_fn, shuffle=False
            )
        else:
            print("Warning: No validation pairs created!")
            self.val_dataloader = None

    def debug_data_structure(self):
        """Debug method to understand data structure"""
        print("\n=== Data Structure Debug ===")
        print(f"Train data shape: {self.glue_train_df.shape}")
        print(f"Train columns: {list(self.glue_train_df.columns)}")
        print(f"Label distribution in train:")
        print(self.glue_train_df['label'].value_counts())

        print(f"\nDev data shape: {self.glue_dev_df.shape}")
        print(f"Label distribution in dev:")
        print(self.glue_dev_df['label'].value_counts())

        # Check groups
        groups = self.glue_train_df.groupby('id_left')
        print(f"\nNumber of unique queries in train: {len(groups)}")

        # Count queries with both positive and negative docs
        queries_with_both = 0
        for query_id, group in groups:
            has_pos = (group.label == 1).any()
            has_neg = (group.label == 0).any()
            if has_pos and has_neg:
                queries_with_both += 1

        print(f"Queries with both positive and negative docs: {queries_with_both}")

        # Show sample
        print("\nSample data:")
        print(self.glue_train_df.head())
        print("=== End Debug ===\n")

    # напишем функцию, которая читает трейн и text-файлы c лейблами
    def _get_glue_df(self, partition_type):
        assert partition_type in ['train', 'dev']
        glue_df = read_csv(self.glue_qqp_dir + f'{partition_type}.tsv', sep='\t', dtype='object')
        glue_df = glue_df.dropna(axis=0, how='any').reset_index(drop=True)
        glue_df_fin = pd.DataFrame({'id_left': glue_df['qid1'],
                                    'id_right': glue_df['qid2'],
                                    'text_left': glue_df['question1'],
                                    'text_right': glue_df['question2'],
                                    'label': glue_df['is_duplicate']})
        # Convert label to numeric
        glue_df_fin['label'] = pd.to_numeric(glue_df_fin['label'], errors='coerce')
        glue_df_fin = glue_df_fin.dropna().reset_index(drop=True)
        return glue_df_fin

    def _create_val_pairs(self, inp_df, fill_top_to=15, min_group_size=1, seed=0):
        """Create validation pairs with proper structure"""
        inp_df['label'] = inp_df['label'].astype('int64')
        # выбираем айдишники и лейблы - мы всегда восстановим по айдишникам сами текста
        inp_df_select = inp_df[['id_left', 'id_right', 'label']]

        # смотрим сколько вообще сэмплов в каждой id-группе
        inp_df_group_sizes = inp_df_select.groupby('id_left').size()

        # выбираем конкретные айдишники текстов с левой колонки - id_left где записей больше 2
        glue_dev_leftids_to_use = list(inp_df_group_sizes[inp_df_group_sizes >= min_group_size].index)

        # получаем список всех ids (правый не фильтруем так как id - симметричные)
        all_ids = set(inp_df['id_left']).union(set(inp_df['id_right']))

        # Устанавливаем seed для PyTorch
        torch.manual_seed(seed)
        random.seed(seed)
        out_pairs = []

        for id_left in glue_dev_leftids_to_use:
            group = inp_df_select[inp_df_select.id_left == id_left]

            # для каждого id_left выбираем все пары релевантных документов
            ones_ids = group[group.label > 0].id_right.values
            # и все пары нерелевантных документов
            zeroes_ids = group[group.label == 0].id_right.values

            # теперь смотрим их количество
            sum_len = len(ones_ids) + len(zeroes_ids)

            # сколько пар нужно добавить для того, чтобы везде было по fill_top_to сэмплов
            num_pad_items = max(0, fill_top_to - sum_len)
            pad_sample = []

            if num_pad_items > 0:
                # все текущие выбранные элементы
                cur_chosen = set(ones_ids).union(set(zeroes_ids)).union({id_left})
                # из остальных рандомно добираем пары ids
                available_ids = list(all_ids - cur_chosen)

                if len(available_ids) >= num_pad_items:
                    # Выбираем без повторений используя Python random
                    import random as py_random
                    pad_sample = py_random.sample(available_ids, num_pad_items)
                else:
                    # Если недостаточно, используем с повторением
                    if len(available_ids) > 0:
                        import random as py_random
                        pad_sample = [py_random.choice(available_ids) for _ in range(num_pad_items)]
                    else:
                        # Если вообще нет доступных, используем случайные ID
                        import random as py_random
                        all_ids_list = list(all_ids)
                        pad_sample = [py_random.choice(all_ids_list) for _ in range(num_pad_items)]

            # добавляем все пары
            for i in ones_ids:
                out_pairs.append([id_left, i, 2])  # релевантные
            for i in zeroes_ids:
                out_pairs.append([id_left, i, 1])  # нерелевантные
            for i in pad_sample:
                out_pairs.append([id_left, i, 0])  # заполнители

        return out_pairs

    # напишем функцию, которая правильно фильтрует полученные датасеты
    def _get_all_tokens(self, list_of_df, min_occurancies):
        tokens = []
        for df in list_of_df:
            if len(df) > 0:
                # Get unique texts from both columns
                unique_texts = set(df[['text_left', 'text_right']].values.reshape(-1))
                # Convert all texts to strings and join them
                unique_texts_str = ' '.join(str(text) for text in unique_texts)
                # Tokenize
                df_tokens = self._simple_preproc(unique_texts_str)
                tokens.extend(df_tokens)
        count_filtered = self._filter_rare_words(Counter(tokens), min_occurancies)
        return list(count_filtered.keys())

    def _simple_preproc(self, inp_str):
        base_str = str(inp_str).strip().lower()
        str_wo_punct = self._handle_punctuation(base_str)
        return nltk.word_tokenize(str_wo_punct)

    def _handle_punctuation(self, inp_str):
        inp_str = str(inp_str)
        for punct in string.punctuation:
            inp_str = inp_str.replace(punct, '')
        return inp_str

    def _filter_rare_words(self, vocab, min_occurancies):
        out_vocab = dict()
        for word, cnt in vocab.items():
            if cnt > min_occurancies:
                out_vocab[word] = cnt
        return out_vocab

    def _read_glove_embeddings(self, file_path):
        embedding_data = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                current_line = line.rstrip().split(' ')
                embedding_data[current_line[0]] = current_line[1:]
        return embedding_data

    def _create_glove_emb_from_file(self, file_path, inner_keys, random_seed, rand_uni_bound):
        glove_emb = self._read_glove_embeddings(file_path)
        inner_keys = ['PAD', 'OOV'] + inner_keys
        high_matrix = len(inner_keys)

        # Get embedding dimension from the first vector
        if glove_emb:
            len_matrix = len(list(glove_emb.values())[0])
        else:
            len_matrix = 50  # Default for GloVe 50d

        matrix = np.empty((high_matrix, len_matrix))
        vocab = dict()
        unk_words = []
        np.random.seed(random_seed)

        for idx, word in enumerate(inner_keys):
            vocab[word] = idx
            if word in glove_emb:
                matrix[idx] = glove_emb[word]
            else:
                unk_words.append(word)
                matrix[idx] = np.random.uniform(-rand_uni_bound, rand_uni_bound, size=len_matrix)
        return matrix, vocab, unk_words

    def _build_knrm_model(self):
        # Import KNRM here to avoid circular imports
        import main
        KNRM = main.KNRM

        emb_matrix, vocab, unk_words = self._create_glove_emb_from_file(
            self.glove_vectors_path,
            self.all_tokens,
            self.random_seed,
            self.emb_rand_uni_bound
        )
        torch.manual_seed(self.random_seed)
        knrm = KNRM(emb_matrix,
                    freeze_embeddings=self.freeze_embeddings,
                    kernel_num=self.knrm_kernel_num,
                    out_layers=self.knrm_out_mlp)
        return knrm, vocab, unk_words

    def _get_idx_to_text_mapping(self, inp_df):
        left_dict = (inp_df[['id_left', 'text_left']].drop_duplicates().set_index('id_left')['text_left'].to_dict())
        right_dict = (
            inp_df[['id_right', 'text_right']].drop_duplicates().set_index('id_right')['text_right'].to_dict())
        left_dict.update(right_dict)
        return left_dict

    def train(self, n_epochs):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.train_lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        ndcgs = []

        for ep in range(n_epochs):
            print(f"\n{'=' * 50}")
            print(f"Epoch {ep}/{n_epochs}")
            print(f"{'=' * 50}")

            # Создаем новые триплеты для каждой эпохи
            sampled_train_triplets = self._sample_data_for_train_iter(
                self.glue_train_df,
                seed=ep,
                max_triplets=20000
            )

            if len(sampled_train_triplets) == 0:
                print(f"ОШИБКА: Нет данных для эпохи {ep}")
                continue

            print(f"Создано {len(sampled_train_triplets)} триплетов для обучения")

            train_dataset = TrainTripletsDataset(
                sampled_train_triplets,
                self.idx_to_text_mapping_train,
                vocab=self.vocab,
                oov_val=self.vocab['OOV'],
                preproc_func=self._simple_preproc,
                max_len=30
            )

            # Используем кастомный sampler вместо shuffle=True
            generator = torch.Generator()
            generator.manual_seed(ep)
            sampler = CustomRandomSampler(train_dataset, generator=generator)

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.dataloader_bs,
                sampler=sampler,
                num_workers=0,
                collate_fn=collate_fn
            )

            # Обучение
            self.model.train()
            total_loss = 0
            num_batches = 0

            for batch_idx, batch in enumerate(train_dataloader):
                try:
                    inp_1, inp_2, y = batch

                    preds_1 = self.model(inp_1)
                    preds_2 = self.model(inp_2)

                    loss = criterion(preds_1.squeeze() - preds_2.squeeze(), y.float())

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    total_loss += loss.item()
                    num_batches += 1

                    if batch_idx % 50 == 0:
                        print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")

                except Exception as e:
                    print(f"Ошибка в батче {batch_idx}: {e}")
                    continue

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {ep} - Средний Loss: {avg_loss:.4f}")

            # Валидация
            if self.val_dataloader:
                ndcg = self.valid(self.model, self.val_dataloader)
                ndcgs.append(ndcg)
                print(f"Epoch {ep} - NDCG: {ndcg:.4f}")

                if ndcg > 0.925:
                    print(f"Ранняя остановка на эпохе {ep} с NDCG {ndcg:.4f}")
                    break

        return ndcgs

    # Остальные методы без изменений...
    def _create_fallback_training_data(self, inp_df, seed, max_pairs=10000):
        """Fallback method to create training data when no triplets are available"""
        set_seeds(seed)
        inp_df['label'] = inp_df['label'].astype('int64')

        triplets = []

        # Get all positive pairs
        positive_pairs = inp_df[inp_df.label == 1][['id_left', 'id_right']].values
        # Get all negative pairs
        negative_pairs = inp_df[inp_df.label == 0][['id_left', 'id_right']].values

        print(f"Positive pairs available: {len(positive_pairs)}")
        print(f"Negative pairs available: {len(negative_pairs)}")

        # If we have both positive and negative pairs, create triplets
        if len(positive_pairs) > 0 and len(negative_pairs) > 0:
            # Create up to max_pairs triplets
            num_triplets = min(max_pairs, len(positive_pairs) * len(negative_pairs))

            for _ in range(num_triplets):
                # Random positive pair
                pos_idx = np.random.randint(0, len(positive_pairs))
                pos_query, pos_doc = positive_pairs[pos_idx]

                # Random negative pair
                neg_idx = np.random.randint(0, len(negative_pairs))
                neg_query, neg_doc = negative_pairs[neg_idx]

                # Use positive query as the query
                triplets.append([pos_query, pos_doc, neg_doc, 1])

        # If still no triplets, create from pairs with same query
        if len(triplets) == 0:
            # Group by query
            groups = inp_df.groupby('id_left')

            for query_id, group in groups:
                docs_with_labels = group[['id_right', 'label']].values

                if len(docs_with_labels) >= 2:
                    # Try to find at least one positive and one negative
                    pos_docs = [doc for doc, label in docs_with_labels if label == 1]
                    neg_docs = [doc for doc, label in docs_with_labels if label == 0]

                    if len(pos_docs) > 0 and len(neg_docs) > 0:
                        for pos_doc in pos_docs[:5]:  # Limit to 5 per query
                            for neg_doc in neg_docs[:5]:
                                triplets.append([query_id, pos_doc, neg_doc, 1])
                                if len(triplets) >= max_pairs:
                                    break
                            if len(triplets) >= max_pairs:
                                break
                    if len(triplets) >= max_pairs:
                        break

        print(f"Created {len(triplets)} fallback training triplets")
        return triplets

    def _sample_data_for_train_iter(self, inp_df, seed, max_triplets=10000):
        """Улучшенная выборка обучающих данных с учетом структуры датасета"""
        set_seeds(seed)
        inp_df['label'] = inp_df['label'].astype('int64')

        triplets = []

        # Получаем все положительные и отрицательные пары
        positive_pairs = inp_df[inp_df.label == 1][['id_left', 'id_right']].values
        negative_pairs = inp_df[inp_df.label == 0][['id_left', 'id_right']].values

        print(f"Доступно положительных пар: {len(positive_pairs)}")
        print(f"Доступно отрицательных пар: {len(negative_pairs)}")

        # Создаем триплеты, комбинируя положительные и отрицательные пары
        if len(positive_pairs) > 0 and len(negative_pairs) > 0:
            # Ограничиваем количество триплетов
            num_triplets = min(max_triplets, len(positive_pairs) * len(negative_pairs))

            # Используем torch для генерации случайных индексов
            for _ in range(num_triplets):
                # Выбираем случайную положительную пару
                pos_idx = torch.randint(0, len(positive_pairs), (1,)).item()
                pos_query, pos_doc = positive_pairs[pos_idx]

                # Выбираем случайную отрицательную пару
                neg_idx = torch.randint(0, len(negative_pairs), (1,)).item()
                neg_query, neg_doc = negative_pairs[neg_idx]

                # Используем запрос из положительной пары
                triplets.append([int(pos_query), int(pos_doc), int(neg_doc), 1])

                # Также иногда используем запрос из отрицательной пары для разнообразия
                if torch.rand(1).item() < 0.3:
                    triplets.append([int(neg_query), int(pos_doc), int(neg_doc), 1])

        # Если всё еще нет триплетов, пробуем другой подход
        if len(triplets) == 0:
            print("Пробуем альтернативный метод создания триплетов...")

            # Группируем по обоим id (левый и правый)
            all_queries = {}

            # Создаем словарь: запрос -> список (документ, релевантность)
            for _, row in inp_df.iterrows():
                # Для левого id
                if row['id_left'] not in all_queries:
                    all_queries[row['id_left']] = []
                all_queries[row['id_left']].append((row['id_right'], row['label']))

                # Для правого id тоже может быть запросом
                if row['id_right'] not in all_queries:
                    all_queries[row['id_right']] = []
                all_queries[row['id_right']].append((row['id_left'], row['label']))

            # Создаем триплеты из запросов, у которых есть и положительные, и отрицательные документы
            for query_id, docs_with_labels in all_queries.items():
                pos_docs = [doc for doc, label in docs_with_labels if label == 1]
                neg_docs = [doc for doc, label in docs_with_labels if label == 0]

                if len(pos_docs) > 0 and len(neg_docs) > 0:
                    # Создаем до 20 триплетов на запрос
                    max_per_query = min(20, len(pos_docs) * len(neg_docs))
                    for _ in range(max_per_query):
                        pos_idx = torch.randint(0, len(pos_docs), (1,)).item()
                        neg_idx = torch.randint(0, len(neg_docs), (1,)).item()
                        pos_doc = pos_docs[pos_idx]
                        neg_doc = neg_docs[neg_idx]
                        triplets.append([int(query_id), int(pos_doc), int(neg_doc), 1])

                        if len(triplets) >= max_triplets:
                            break

                if len(triplets) >= max_triplets:
                    break

        # Если всё еще нет, создаем искусственные пары
        if len(triplets) == 0:
            print("Создаем искусственные пары...")
            # Берем все положительные пары как релевантные
            # и случайные документы из других запросов как нерелевантные
            all_docs = set(inp_df['id_left'].values) | set(inp_df['id_right'].values)
            all_docs_list = list(all_docs)

            for query_id, pos_doc in positive_pairs[:200]:  # Ограничиваем количество
                # Ищем нерелевантный документ
                candidate_docs = list(all_docs - {pos_doc})
                if candidate_docs:
                    neg_idx = torch.randint(0, len(candidate_docs), (1,)).item()
                    neg_doc = candidate_docs[neg_idx]
                    triplets.append([int(query_id), int(pos_doc), int(neg_doc), 1])

                    if len(triplets) >= max_triplets:
                        break

        print(f"Создано {len(triplets)} тренировочных триплетов")
        return triplets

    def valid(self, model, val_dataloader):
        if val_dataloader is None:
            return 0.0

        labels_and_groups = val_dataloader.dataset.index_pairs_or_triplets
        labels_and_groups = pd.DataFrame(labels_and_groups, columns=['id_left', 'id_right', 'rel'])

        all_preds = []
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                pair, _ = batch
                # Исправление здесь - передаем словарь в модель
                preds = model(pair)  # pair уже является словарем с ключами 'query' и 'document'
                # Конвертируем в список Python вместо numpy
                preds_list = preds.squeeze().tolist()
                if isinstance(preds_list, list):
                    all_preds.extend(preds_list)
                else:
                    all_preds.append(preds_list)

        if not all_preds:
            return 0.0

        labels_and_groups['preds'] = all_preds

        ndcgs = []
        for cur_id in labels_and_groups['id_left'].unique():
            cur_df = labels_and_groups[labels_and_groups['id_left'] == cur_id]
            if len(cur_df) > 1:
                ndcg = self._ndcg_k(cur_df.rel.values, cur_df.preds.values, ndcg_top_k=10)
                if not np.isnan(ndcg):
                    ndcgs.append(ndcg)

        return np.mean(ndcgs) if ndcgs else 0.0

    def _ndcg_k(self, ys_true, ys_pred, ndcg_top_k=10):
        """
        Вычисляет NDCG@k без использования numpy (только Python списки)
        """

        def dcg(ys_true, ys_pred):
            # Создаем список пар (значение, индекс)
            pairs = list(enumerate(ys_pred))
            # Сортируем по значению предсказания (убывание)
            pairs.sort(key=lambda x: x[1], reverse=True)
            # Берем топ-k индексов
            top_k_indices = [idx for idx, _ in pairs[:ndcg_top_k]]
            # Сортируем истинные значения по этим индексам
            ys_true_sorted = [ys_true[i] for i in top_k_indices]
            ret = 0
            for i, l in enumerate(ys_true_sorted, 1):
                ret += (2 ** l - 1) / (math.log2(i + 1))
            return ret

        # Конвертируем в списки, если это numpy массивы
        if hasattr(ys_true, 'tolist'):
            ys_true = ys_true.tolist()
        if hasattr(ys_pred, 'tolist'):
            ys_pred = ys_pred.tolist()

        ideal_dcg = dcg(ys_true, ys_true)
        if ideal_dcg == 0:
            return 0.0
        pred_dcg = dcg(ys_true, ys_pred)
        return pred_dcg / ideal_dcg


class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, index_pairs_or_triplets, idx_to_text_mapping, vocab, oov_val, preproc_func, max_len=30):
        self.index_pairs_or_triplets = index_pairs_or_triplets
        self.vocab = vocab
        self.oov_val = oov_val
        self.preproc_func = preproc_func
        self.idx_to_text_mapping = idx_to_text_mapping
        self.max_len = max_len

    def __len__(self):
        return len(self.index_pairs_or_triplets)

    def _tokenized_text_to_index(self, tokenized_text):
        res = [self.vocab.get(i, self.oov_val) for i in tokenized_text]
        return res

    def _convert_text_idx_to_token_idxs(self, idx):
        if idx not in self.idx_to_text_mapping:
            return []
        tokenized_text = self.preproc_func(self.idx_to_text_mapping[idx])
        idxs = self._tokenized_text_to_index(tokenized_text)
        return idxs[:self.max_len]

    def __getitem__(self, idx):
        pass


class TrainTripletsDataset(RankingDataset):
    def __getitem__(self, idx):
        cur_row = self.index_pairs_or_triplets[idx]
        left_idxs = self._convert_text_idx_to_token_idxs(cur_row[0])
        r1_idxs = self._convert_text_idx_to_token_idxs(cur_row[1])
        r2_idxs = self._convert_text_idx_to_token_idxs(cur_row[2])

        pair_1 = {'query': left_idxs, 'document': r1_idxs}
        pair_2 = {'query': left_idxs, 'document': r2_idxs}
        target = cur_row[3]
        return (pair_1, pair_2, target)


class ValPairsDataset(RankingDataset):
    def __getitem__(self, idx):
        cur_row = self.index_pairs_or_triplets[idx]
        left_idxs = self._convert_text_idx_to_token_idxs(cur_row[0])
        r1_idxs = self._convert_text_idx_to_token_idxs(cur_row[1])

        pair = {'query': left_idxs, 'document': r1_idxs}
        target = cur_row[2]
        return (pair, target)


def collate_fn(batch_obj):
    max_len_q = -1
    max_len_d = -1
    is_triplets = False

    # First pass to find max lengths
    for elem in batch_obj:
        if len(elem) == 3:
            left_elem, right_elem, label = elem
            is_triplets = True
            max_len_q = max(len(left_elem['query']), max_len_q)
            max_len_d = max(len(left_elem['document']), max_len_d)
            max_len_q = max(len(right_elem['query']), max_len_q)
            max_len_d = max(len(right_elem['document']), max_len_d)
        else:
            left_elem, label = elem
            max_len_q = max(len(left_elem['query']), max_len_q)
            max_len_d = max(len(left_elem['document']), max_len_d)

    # Second pass to pad
    q1s = []
    d1s = []
    q2s = []
    d2s = []
    labels = []

    for elem in batch_obj:
        if is_triplets:
            left_elem, right_elem, label = elem
            pad_len_q1 = max_len_q - len(left_elem['query'])
            pad_len_d1 = max_len_d - len(left_elem['document'])
            pad_len_q2 = max_len_q - len(right_elem['query'])
            pad_len_d2 = max_len_d - len(right_elem['document'])

            q1s.append(left_elem['query'] + [0] * pad_len_q1)
            d1s.append(left_elem['document'] + [0] * pad_len_d1)
            q2s.append(right_elem['query'] + [0] * pad_len_q2)
            d2s.append(right_elem['document'] + [0] * pad_len_d2)
            labels.append(label)
        else:
            left_elem, label = elem
            pad_len_q = max_len_q - len(left_elem['query'])
            pad_len_d = max_len_d - len(left_elem['document'])

            q1s.append(left_elem['query'] + [0] * pad_len_q)
            d1s.append(left_elem['document'] + [0] * pad_len_d)
            labels.append(label)

    # Convert to tensors
    q1s = torch.LongTensor(q1s)
    d1s = torch.LongTensor(d1s)
    labels = torch.FloatTensor(labels)

    if is_triplets:
        q2s = torch.LongTensor(q2s)
        d2s = torch.LongTensor(d2s)
        return ({'query': q1s, 'document': d1s}, {'query': q2s, 'document': d2s}, labels)
    else:
        return ({'query': q1s, 'document': d1s}, labels)


# Main execution
if __name__ == "__main__":
    try:
        sol = Solution(
            glue_qqp_dir="glue_qqp/",
            glove_vectors_path="glove.6B/glove.6B.50d.txt",
            min_occurancies=1,
            random_seed=42,
            emb_rand_uni_bound=0.2,
            freeze_knrm_embeddings=True
        )

        ndcgs = sol.train(12)
        print(f"\n{'=' * 50}")
        print(f"Training completed!")
        print(f"Final NDCG scores: {ndcgs}")
        if ndcgs:
            print(f"Best NDCG: {max(ndcgs):.4f}")
        print(f"{'=' * 50}")

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback

        traceback.print_exc()