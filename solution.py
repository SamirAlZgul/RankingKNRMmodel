'''

будем обучать правильно рандировать документы, для этого надо написать все функции по формированию документов
'''
import pandas as pd
from pandas import read_csv
from datasets import load_dataset

class Solution:
    def __init__(self, glue_qqp_dir):


        # путь до папки с трейном и тестом
        self.glue_qqp_dir = glue_qqp_dir
        # путь до файла с эмбеддингами
        #self.glove_vectors_path = glove_vectors_path
        # надо взять отдельно трейн и тест
        self.glue_train_df = self._get_glue_df('train')
        self.glue_dev_df = self._get_glue_df('dev')


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



# sol = Solution("./")
# dat = sol.glue_train_df
# print(dat.head(4))