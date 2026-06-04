# Classificador de Sentimentos com Naive Bayes Multinomial

## 📖 Sobre o Projeto

Este projeto foi desenvolvido como trabalho da disciplina de Inteligência Artificial com o objetivo de demonstrar uma aplicação prática do algoritmo **Naive Bayes Multinomial** na área de **Processamento de Linguagem Natural (PLN)**.

O sistema é capaz de analisar frases relacionadas a jogos eletrônicos e classificá-las automaticamente em três categorias de sentimento:

* ✅ Positivo
* ❌ Negativo
* ➖ Neutro

A implementação foi realizada do zero em Python, sem utilizar bibliotecas de Machine Learning como Scikit-Learn, permitindo compreender detalhadamente o funcionamento interno do algoritmo.

---

## 🎯 Objetivos

* Demonstrar o funcionamento do algoritmo Naive Bayes Multinomial.
* Aplicar conceitos de classificação de textos.
* Utilizar probabilidades condicionais para prever sentimentos.
* Implementar técnicas de pré-processamento textual.
* Compreender o impacto da Suavização de Laplace na classificação.

---

## 🧠 Conceitos Utilizados

### Naive Bayes

O Naive Bayes é um algoritmo probabilístico baseado no Teorema de Bayes. Ele assume que as características analisadas (neste caso, as palavras de uma frase) são independentes entre si.

A classificação é realizada calculando a probabilidade de uma frase pertencer a cada classe disponível e escolhendo a classe com maior probabilidade.

### Naive Bayes Multinomial

A variante Multinomial é amplamente utilizada para classificação de textos, pois considera a frequência das palavras presentes nos documentos.

### Suavização de Laplace

Foi utilizada a Suavização de Laplace (α = 1.0) para evitar que palavras não observadas durante o treinamento resultem em probabilidades iguais a zero.

---

## ⚙️ Funcionalidades

* Pré-processamento de texto:

  * Conversão para letras minúsculas;
  * Remoção de acentos;
  * Remoção de pontuação;
  * Tokenização das palavras.

* Treinamento do modelo utilizando frases rotuladas.

* Classificação de novas frases digitadas pelo usuário.

* Cálculo das probabilidades utilizando logaritmos para evitar problemas de underflow numérico.

---

## 🛠️ Tecnologias Utilizadas

* Python 3
* Bibliotecas padrão:

  * math
  * string
  * collections
  * unicodedata

Nenhuma biblioteca externa de Machine Learning foi utilizada.

---

## 📂 Estrutura do Algoritmo

### Treinamento

Durante o treinamento o programa:

1. Recebe frases previamente classificadas.
2. Conta a quantidade de documentos por classe.
3. Conta a frequência de cada palavra em cada classe.
4. Armazena o vocabulário completo.

### Predição

Ao receber uma nova frase:

1. O texto é pré-processado.
2. São calculadas as probabilidades de cada classe.
3. É aplicada a Suavização de Laplace.
4. A classe com maior probabilidade é selecionada como resultado.

---

## ▶️ Como Executar

Clone o repositório:

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
```

Acesse a pasta do projeto:

```bash
cd seu-repositorio
```

Execute o programa:

```bash
python main.py
```

---

## 💻 Exemplo de Uso

Entrada:

```text
Frase para classificar: gameplay incrível e gráficos lindos
```

Saída:

```text
→ Classificação: [POSITIVO]
```

Entrada:

```text
Frase para classificar: jogo cheio de bugs e problemas
```

Saída:

```text
→ Classificação: [NEGATIVO]
```

---

Entrada:

```text
Frase para classificar: historia ruim, mas boa jogabilidade
```

Saída:

```text
→ Classificação: [NEUTRO]
```

---

## 📚 Aplicação Prática

A análise de sentimentos é amplamente utilizada em diversas áreas, como:

* Avaliação de produtos;
* Análise de comentários em redes sociais;
* Classificação de avaliações de clientes;
* Monitoramento de reputação de marcas;
* Sistemas de recomendação.

Este projeto demonstra como um algoritmo relativamente simples pode ser utilizado para resolver problemas reais de classificação de texto.

---

## 👨‍🎓 Trabalho Acadêmico

Projeto desenvolvido para a disciplina de Inteligência Artificial com o objetivo de estudar e demonstrar a aplicação prática do algoritmo Naive Bayes Multinomial na classificação automática de sentimentos em textos.
