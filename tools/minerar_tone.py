from collections import defaultdict, Counter
import re


def carregar_exemplos(caminho_log: str) -> dict:
    """
    Lê o arquivo de logs do LLM e organiza as mensagens por tom identificado.
    Args:
        caminho_log (str): Caminho para o arquivo de log gerado pelo classificador de tom LLM.
    Returns:
        dict: Dicionário {tom: [mensagens]}.
    """
    resultados = defaultdict(list)
    with open(caminho_log, encoding="utf-8") as f:
        for linha in f:
            if linha.startswith("TOM:"):
                partes = linha.strip().split("|")
                if len(partes) == 2:
                    tom = partes[0].replace("TOM:", "").strip()
                    msg = partes[1].replace("MSG:", "").strip()
                    resultados[tom].append(msg)
    return resultados


def sugerir_termos_por_tom(dados_por_tom: dict, stopwords_set: set, min_freq: int = 2, max_termos: int = 30):
    """
    Para cada tom, imprime termos/gírias mais frequentes prontos para colar no classificador.
    Args:
        dados_por_tom (dict): Dicionário {tom: [mensagens]}.
        stopwords_set (set): Palavras a serem ignoradas.
        min_freq (int): Frequência mínima para sugerir o termo.
        max_termos (int): Número máximo de sugestões por tom.
    """
    for tom, frases in dados_por_tom.items():
        palavras = []
        for frase in frases:
            palavras += re.findall(r'\w+', frase.lower())
        freq = Counter(w for w in palavras if w not in stopwords_set)
        mais_comuns = [w for w, c in freq.most_common(max_termos) if c >= min_freq]

        print(f"\n# ======= {tom} ======= #")
        if tom == "IRRITADO E CONCISO":
            print("irritado_girias = (")
        elif tom == "INFORMAL E DESCONTRAÍDO":
            print("informal_girias = (")
        elif tom == "FORMAL E POLIDO":
            print("formal_frases = (")
        else:
            print("objetivo_termos = (")

        for palavra in mais_comuns:
            print(f'    "{palavra}",')
        print(")\n")
    print("# Copie e cole os termos acima na função detect_tone_local.")


# --- Configuração ---

# Caminho do log do LLM
CAMINHO_LOG = "logs/tone_llm_cases.txt"

# Stopwords básicas para português (pode expandir se quiser)
STOPWORDS = set("""
de a o e que do da em um uma é para com os no na por mas se não foi vai eu você ele ela nós eles elas já tá né pra lá aqui isso aquilo então
""".split())

# Execução
if __name__ == "__main__":
    dados_log = carregar_exemplos(CAMINHO_LOG)
    sugerir_termos_por_tom(dados_log, STOPWORDS)
