import string
import math
from collections import defaultdict

class MultinomialNaiveBayes: 
    
    #variaveis para realizar o calculo de probabilidade
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.vocabularies = set()
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.total_words_per_class = defaultdict(int)
        self.class_counts = defaultdict(int)
        self.total_documents = 0

    #separação e amazenamento das palavras 
    def _tokenize(self, text):
        text = text.lower()
        translator = str.maketrans("", "", string.punctuation)
        text = text.translate(translator) 
        return text.split()

    
    # Treina o modelo contando as palavras e as classes.
    def fit(self, X, y):
        
        self.total_documents = len(X)
        
        for text, label in zip(X, y):
            self.class_counts[label] += 1
            words = self._tokenize(text)
            
            for word in words:
                self.vocabularies.add(word)
                self.word_counts[label][word] += 1
                self.total_words_per_class[label] += 1

    # Classifica novos textos.
    def predict(self, X):
        
        predictions = []
        num_v = len(self.vocabularies) # Tamanho do vocabulário para a Suavização de Laplace
        
        for text in X:
            words = self._tokenize(text)
            best_class = None
            max_prob = float('-inf') # Começa com a menor probabilidade possível
            
            # Calcula a probabilidade para cada classe existente no treino
            for label in self.class_counts:
                # 1. Probabilidade a Priori: P(Classe) = docs_da_classe / total_docs
                # Usamos log() para evitar que multiplicações de números muito pequenos virem zero (Underflow)
                prob_classe = math.log(self.class_counts[label] / self.total_documents)
                
                # 2. Probabilidade Condicional das palavras: P(Palavra | Classe)
                prob_palavras = 0
                for word in words:
                    if word in self.vocabularies: # Ignora palavras que nunca viu no treino
                        # Suavização de Laplace aplicada aqui: (contagem + alpha) / (total_palavras + alpha * V)
                        count = self.word_counts[label][word]
                        prob_w_dado_c = (count + self.alpha) / (self.total_words_per_class[label] + self.alpha * num_v)
                        prob_palavras += math.log(prob_w_dado_c)
                
                # Probabilidade total para esta classe = P(Classe) * P(Palavra1|Classe) * ...
                # Como usamos log, a multiplicação vira SOMA:
                prob_total = prob_classe + prob_palavras
                
                if prob_total > max_prob:
                    max_prob = prob_total
                    best_class = label
                    
            predictions.append(best_class)
        return predictions
    

if __name__ == "__main__":

    # dados de treino 
    X_treino = [
    # positivos
    " gameplay incrível e muito divertido",
    " gráficos lindos e história envolvente",
    " controles responsivos e jogabilidade fluida",
    " valeu cada centavo recomendo muito",
    " trilha sonora épica e mundo aberto enorme",

    # negativos
    " cheio de bugs e trava o tempo todo",
    " história fraca e personagens sem profundidade",
    " muito curto e decepcionante para o preço",
    " otimização péssima cai para menos de 30fps",
    " controles horríveis e câmera impossível",
    "jogo muito ruim",

    # neutros
    " gameplay mediano nada que surpreenda",
    " gráficos ok mas historia deixa a desejar",
    " divertido por um tempo mas enjoa rápido",
    " tem pontos bons e ruins no geral razoável",
    ]
        

    # As respostas corretas para o treino (labels)
    y_treino = [
        #Positivos
        "positivo", "positivo", "positivo", "positivo", "positivo",

        #negativo
        "negativo", "negativo", "negativo", "negativo", "negativo", "negativo",

        #NEUTRO
        "neutro",   "neutro",   "neutro",   "neutro",
    ]

    # Instanciar o classificador
    modelo = MultinomialNaiveBayes(alpha=1.0)

    # Treinar o modelo com os dados históricos
    print("Treinando o modelo Naive Bayes...")
    modelo.fit(X_treino, y_treino)
    print("Treinamento concluído!\n")

    # --- FASE DE TESTE ---
    print("=== FASE DE CLASSIFICAÇÃO ===")
    print("Digite frases para classificar. Digite ""sair"" para encerrar.\n")

    while True:
        frase = input("Frase para classificar: ").strip()
        if frase.lower() == "sair":
            break

        resultado = modelo.predict([frase])[0]
        print(f"→ Classificação: [{resultado.upper()}]\n")

    print("Encerrando.")