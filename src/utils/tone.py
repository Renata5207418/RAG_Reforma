import re
import unicodedata
import logging
from langchain_openai import ChatOpenAI
from src.config import settings

log = logging.getLogger(__name__)


def detect_tone_local(msg: str) -> str:
    """
    Detecta o tom da mensagem com base em padrões, gírias e palavras-chave conhecidas.
    Args:
        msg (str): Texto da pergunta do usuário.
    Returns:
        str:Um dos tons ('irritado e conciso', 'informal e descontraído', 'formal e polido', 'objetivo').
    """
    txt = unicodedata.normalize("NFKD", msg.lower())
    # Irritado
    if re.search(r"!{2,}|[😡🤬🤯]", msg) or any(w in txt for w in (
        "absurdo", "ridiculo", "ridículo", "palhaçada", "roubo", "nojento", "vergonha", "mentira",
        "mentiroso", "sacanagem", "falta de respeito", "não aguento", "ladrão", "indignado"
    )):
        return "irritado e conciso"
    # Informal
    if any(g in txt for g in (
        "mano", "mona", "bixa", "amigo", "amiga", "véi", "velho", "cara", "porra", "tipo", "mó", "tô", "sai fora", "aff", "vish", "top",
        "massa", "zika", "bora", "falae", "tmj", "parça", "kkk", "rs", "meu", "oxe", "véa", "véio", "kk", "rsrs"
    )):
        return "informal e descontraído"
    # Formal/polido
    if any(w in txt for w in (
        "por favor", "gentileza", "poderia", "agradeço", "cordialmente", "atenciosamente", "fico no aguardo",
        "seria possível", "obrigado", "obrigada", "grato", "grata", "gostaria", "aprecio", "saudações"
    )):
        return "formal e polido"
    return "objetivo"


# Função de detecção por LLM (OpenAI)
def detect_tone_llm(msg: str) -> str:
    """
    Usa LLM (OpenAI) para classificar o tom da mensagem quando o método local não reconhece.
    Args:
        msg (str): Texto da pergunta do usuário.
    Returns:
        str:Um dos tons ('irritado e conciso', 'informal e descontraído', 'formal e polido', 'objetivo').
    """
    llm = ChatOpenAI(
        model=settings.default_model,  # use um modelo leve e barato aqui
        openai_api_key=settings.api_key,
        temperature=0,
        max_tokens=15,
        top_p=1.0,
    )
    prompt = (
        "Classifique o tom da mensagem a seguir como uma das opções abaixo, respondendo APENAS com o nome do tom:\n"
        "- irritado e conciso\n"
        "- informal e descontraído\n"
        "- formal e polido\n"
        "- objetivo\n\n"
        "Mensagem: " + msg.strip()
    )
    try:
        result = llm.invoke(prompt).content.strip().lower()
        # Padroniza saída
        if "irritado" in result:
            return "irritado e conciso"
        if "informal" in result:
            return "informal e descontraído"
        if "formal" in result:
            return "formal e polido"
        return "objetivo"
    except Exception as exc:
        log.warning(f"LLM detect_tone_llm falhou: {exc}")
        return "objetivo"


# Função principal híbrida
def detect_tone(msg: str) -> str:
    """
    Detecta o tom híbrido: tenta localmente, depois via LLM se necessário.
    Loga exemplos classificados só pela LLM.
    Args:
        msg (str): Texto da pergunta do usuário.
    Returns:
        str: Tom detectado.
    """
    tone = detect_tone_local(msg)
    if tone == "objetivo":
        tone_llm = detect_tone_llm(msg)
        if tone_llm != "objetivo":
            with open("logs/tone_llm_cases.txt", "a", encoding="utf-8") as f:
                f.write(f"TOM: {tone_llm.upper()} | MSG: {msg.strip()}\n")
            log.info(f"Tone LLM detectou '{tone_llm}' para: {msg.strip()}")
        return tone_llm
    return tone
