import string
import math
from collections import defaultdict
import unicodedata

class MultinomialNaiveBayes: 
    
    #iniciação das variaveis
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
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
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
                #  Probabilidade a Priori: P(Classe) = docs_da_classe / total_docs
                # Usamos log() para evitar que multiplicações de números muito pequenos virem zero (Underflow)
                prob_classe = math.log(self.class_counts[label] / self.total_documents)
                
                # Probabilidade Condicional das palavras: P(Palavra | Classe)
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
    "o jogo é muito bom",
    "gameplay incrível e muito divertido",
    "gráficos lindos e história envolvente",
    "controles responsivos e jogabilidade fluida",
    "valeu cada centavo recomendo muito",
    "trilha sonora épica e mundo aberto enorme",
    "campanha emocionante e cheia de momentos marcantes",
    "multiplayer muito divertido com amigos",
    "excelente otimização roda liso no pc",
    "missões variadas e interessantes",
    "personagens carismáticos e bem desenvolvidos",
    "um dos melhores jogos que já joguei",
    "combate satisfatório e viciante",
    "mapa enorme cheio de coisas para fazer",
    "visual impressionante e muito bonito",
    "jogo extremamente divertido do começo ao fim",
    "inteligência artificial dos inimigos é ótima",
    "efeitos sonoros muito realistas",
    "experiência fantástica recomendo para todos",
    "jogo inovador e criativo",
    "história emocionante e bem escrita",
    "progressão muito recompensadora",
    "ótimo sistema de combate",
    "interface simples e fácil de usar",
    "animações muito bem feitas",
    "vale muito a pena comprar",
    "exepriencia incrivel",

    # negativos
    "cheio de bugs e trava o tempo todo",
    "história fraca e personagens sem profundidade",
    "muito curto e decepcionante para o preço",
    "otimização péssima cai para menos de 30fps",
    "controles horríveis e câmera impossível",
    "jogo muito ruim",
    "servidores vivem caindo",
    "muitas microtransações irritantes",
    "gameplay repetitivo e cansativo",
    "gráficos ultrapassados para a geração atual",
    "combate travado e sem graça",
    "missões muito repetitivas",
    "trilha sonora esquecível",
    "ia dos inimigos é muito burra",
    "vários crashes durante a partida",
    "jogo mal acabado e cheio de problemas",
    "não vale o preço cobrado",
    "campanha extremamente curta",
    "interface confusa e mal organizada",
    "pior experiência que tive em um jogo",
    "animações estranhas e mal feitas",
    "história sem sentido e mal escrita",
    "mundo vazio e sem vida",
    "multiplayer desbalanceado e injusto",
    "falta conteúdo no jogo",

    # neutros
    "gameplay mediano nada que surpreenda",
    "gráficos ok mas historia deixa a desejar",
    "divertido por um tempo mas enjoa rápido",
    "tem pontos bons e ruins no geral razoável",
    "o jogo é muito seguro entretanto fica aquém da verdadeira grandeza",
    "campanha razoável mas poderia ser melhor",
    "não é ruim mas também não impressiona",
    "gráficos medianos e gameplay aceitável",
    "jogo simples cumpre o que promete",
    "experiência comum sem grandes novidades",
    "alguns momentos são divertidos outros nem tanto",
    "vale a pena apenas em promoção",
    "o desempenho é estável mas os gráficos são simples",
    "história interessante porém mal desenvolvida",
    "combate divertido mas repetitivo",
    "multiplayer funciona bem mas falta conteúdo",
    "jogo ok para passar o tempo",
    "tem potencial mas precisa melhorar",
    "começa muito bem mas perde ritmo depois",
    "a experiência geral é apenas mediana",
    "poderia ser melhor"
    ]
        

    # As respostas corretas para o treino (labels)
    y_treino = [
        #Positivos
        "positivo", "positivo", "positivo","positivo","positivo","positivo","positivo",
        "positivo","positivo","positivo","positivo","positivo",
        "positivo","positivo","positivo","positivo","positivo",
        "positivo","positivo","positivo","positivo","positivo",
        "positivo","positivo","positivo","positivo","positivo",


        #negativo
        "negativo","negativo","negativo","negativo","negativo",
        "negativo","negativo","negativo","negativo","negativo",
        "negativo","negativo","negativo","negativo","negativo",
        "negativo","negativo","negativo","negativo","negativo",
        "negativo","negativo","negativo","negativo","negativo",

        #NEUTRO
        "neutro","neutro","neutro","neutro","neutro",
        "neutro","neutro","neutro","neutro","neutro",
        "neutro","neutro","neutro","neutro","neutro",
        "neutro","neutro","neutro","neutro","neutro", "neutro"
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