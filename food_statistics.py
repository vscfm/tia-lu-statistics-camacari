class Statistics:
    """
    Uma classe para realizar cálculos estatísticos em um conjunto de dados.

    Atributos
    ----------
    dataset : dict[str, list]
        O conjunto de dados, estruturado como um dicionário onde as chaves
        são os nomes das colunas e os valores são listas com os dados.
    """
    def __init__(self, dataset):
        """
        Inicializa o objeto Statistics.

        Parâmetros
        ----------
        dataset : dict[str, list]
            O conjunto de dados, onde as chaves representam os nomes das
            colunas e os valores são as listas de dados correspondentes.
        """
        self.dataset = dataset

    def mean(self, column):
        values = self.dataset[column]
        return sum(values) / len(values)

    def median(self, column):
        values = sorted(self.dataset[column])
        n = len(values)
        mid = n // 2
        if n % 2 == 0:
            return (values[mid - 1] + values[mid]) / 2
        return values[mid]

    def mode(self, column):
        values = self.dataset[column]
        freq = {}
        for v in values:
            freq[v] = freq.get(v, 0) + 1
        max_freq = max(freq.values())
        return [k for k, v in freq.items() if v == max_freq]

    def stdev(self, column):
        values = self.dataset[column]
        mean_val = self.mean(column)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def variance(self, column):
        values = self.dataset[column]
        mean_val = self.mean(column)
        return sum((x - mean_val) ** 2 for x in values) / len(values)

    def covariance(self, column_a, column_b):
        x = self.dataset[column_a]
        y = self.dataset[column_b]
        if len(x) != len(y):
            raise ValueError("As colunas devem ter o mesmo tamanho.")
        mean_x = self.mean(column_a)
        mean_y = self.mean(column_b)
        return sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / len(x)

    def itemset(self, column):
        return set(self.dataset[column])

    def absolute_frequency(self, column):
        values = self.dataset[column]
        freq = {}
        for v in values:
            freq[v] = freq.get(v, 0) + 1
        return freq

    def relative_frequency(self, column):
        abs_freq = self.absolute_frequency(column)
        total = sum(abs_freq.values())
        return {k: v / total for k, v in abs_freq.items()}

    def cumulative_frequency(self, column, frequency_method='absolute'):
        if frequency_method == 'absolute':
            freq = self.absolute_frequency(column)
        elif frequency_method == 'relative':
            freq = self.relative_frequency(column)
        else:
            raise ValueError("Método inválido. Use 'absolute' ou 'relative'.")
        
        acumulado = {}
        soma = 0
        for k in sorted(freq.keys()):
            soma += freq[k]
            acumulado[k] = soma
        return acumulado

    def conditional_probability(self, column, value1, value2):
        values = self.dataset[column]
        total_b = 0
        count_ba = 0
        for i in range(len(values) - 1):
            if values[i] == value2:
                total_b += 1
                if values[i + 1] == value1:
                    count_ba += 1
        return count_ba / total_b if total_b > 0 else 0.0
