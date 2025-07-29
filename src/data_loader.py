from typing import List, Dict


def load_test_docs() -> List[Dict[str, str]]:
    """
    Retorna uma lista de dicionários com 'id' e 'text'
    para usarmos no proof‑of‑concept de RAG.
    """
    return [
        {
            "id": "1",
            "text": "A Reforma Tributária do Brasil foi promulgada em 2023 e terá início em fases a partir de 2026, unificando diversos tributos sobre o consumo."
        },
        {
            "id": "2",
            "text": "A principal mudança da reforma é a substituição de tributos como PIS, Cofins, IPI, ICMS e ISS por dois novos impostos: o Imposto sobre Bens e Serviços (IBS) e a Contribuição sobre Bens e Serviços (CBS)."
        },
        {
            "id": "3",
            "text": "O novo modelo prevê um sistema dual, onde a CBS será de competência federal e o IBS será gerido por estados e municípios, com arrecadação centralizada por um comitê gestor nacional."
        },
        {
            "id": "4",
            "text": "A transição do sistema atual para o novo modelo ocorrerá entre 2026 e 2033, com a convivência de ambos os regimes por um período determinado."
        },
        {
            "id": "5",
            "text": "A alíquota padrão dos novos tributos ainda será definida por lei complementar, mas estudos indicam que poderá ficar em torno de 25% somando CBS e IBS."
        },
        {
            "id": "6",
            "text": "A reforma também prevê um sistema de cashback tributário para famílias de baixa renda, devolvendo parte dos tributos pagos no consumo."
        }
    ]
