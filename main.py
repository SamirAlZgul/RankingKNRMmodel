import torch
import torch.nn.functional as F

class GaussianKernel(torch.nn.Module):
    def __init__(self, mu=1., sigma=1.):
        super().__init__()
        self.mu=mu
        self.sigma=sigma
    def forward(self,x):
        return torch.exp(-0.5*torch.pow(x-self.mu,2)/(self.sigma**2))

class KNRM(torch.nn.Module):
    """
        KNRM (Kernel-based Neural Ranking Model) - нейросетевая модель для ранжирования документов
        Использует ядра гаусса для моделирования совпадений на разных уровнях точности
    """
    def __init__(self, kernel_num=21, sigma=0.1, exact_sigma=0.001):
        """
            Args:
            kernel_num: количество ядер (по умолчанию 21, как в оригинальной статье)
            sigma: стандартное отклонение для ядер Гаусса
        """
        super(KNRM, self).__init__()
        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma

        self.kernels = self._get_kernels_layers()


    def _get_kernels_layers(self):
        """
            Создает слои с ядрами Гаусса для KNRM модели.
            Каждое ядро отвечает за определенный уровень совпадения термов.
            # активируем 21 степень похожести
        """
        # ModuleList позволяет корректно зарегистрировать все ядра как параметры модели
        kernels = torch.nn.ModuleList()
        for i in range(self.kernel_num):
            mu = 1/(self.kernel_num-1) + 2*i/(self.kernel_num-1) - 1.0
            sigma = self.sigma
            if mu>1:
                mu=1
                sigma=self.exact_sigma
            kernels.append(GaussianKernel(mu, sigma))
        return kernels

    def _get_matching_matrix(self, query, doc):
        """
        показывает как каждое слово запроса соотносится с каждым словом документа
        на вход тут BWD - это матрица (Batch, Words, Dimension) -
        (размер батча (количество запросов), количество слов в запросе, размер эмбеддинга)
        BRD - это матрица (Batch, Words in Document, Dimension) -
        (размер батча (количество документов), количество слов в док, размер эмбеддинга)
        BWR - это матрица (Batch, Words, количество слов в док)
        matching_matrix[b, i, j] =
        косинусное сходство между i-тым словом запроса и j-тым словом документа в b-том примере
        (используем построение матрицы в нотации Эйнштейна)
        """
        embed_query = self.embedding(query)
        embed_doc = self.embedding(doc)
        matching_matrix = torch.einsum(
            'BWD, BRD->BWR',
            F.normalize(embed_query, p=2, dim=-1),
            F.normalize(embed_doc, p=2, dim=-1)
        )

    def _apply_kernels(self, matching_matrix):
        KM = []
        for kernel in self.kernels:
            # берем Гауссово ядро exp(-0.5 * (x - μ)²/σ²) и применяем к каждому элементу матрицы матчинга
            # получаем насколько далеки полученные знаачения от центра текущего распределения (ядра)
            # далее дял каждого запроса суммируем все похожести по строкам
            # далее по столбцам суммируем по всем документам
            # далее это все обрабатываем log1p чтобы сгладить большие значения
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1).sum(dim=-1)) # (batch_size, ) # по количеству запросов
            KM.append(K) # список из тензора размера batch_size,
        kernels_out = torch.stack(KM, dim=1) # (batch_size, self.kernels)
        return kernels_out




