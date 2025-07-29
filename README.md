# RAG‑Reforma‑Tributária

Este projeto é um *proof‑of‑concept* de pipeline RAG (Retrieval‑Augmented Generation) usando **LangChain** e **OpenAI**, 
com **Qdrant** como base vetorial. O objetivo é responder perguntas sobre a Reforma Tributária brasileira,
usando como fonte documentos oficiais (Word, PDF, planilhas), permitindo atualização fácil do contexto sem alterar o pipeline.

---

## 📋 Estrutura do Projeto

```text
LangChain [RAG-ReformaTributária]/
├── .venv/                         # Ambiente virtual Python
├── logs/
│   └── rag.log                    # Logs de execução
├── qdrant_data/                   # Volume local de dados do Qdrant
├── src/
│   ├── __init__.py
│   ├── config.py                  # Carrega .env e settings
│   ├── data_loader.py             # Loader dos docs (Word/Excel/CSV)
│   ├── qa_chain.py                # Pipeline RAG (Retriever + LLM)
│   ├── qa_safe.py                 # Fallback seguro do QA
│   ├── utils/
│   │   └── tone.py                # Detector de tom da pergunta
│   └── vector_store.py            # QdrantVectorStore (inicialização + dedupe)
├── tests/                         # Testes de regressão RAG
│   ├── data/
│   │   └── gold.jsonl             # Perguntas + termos‑chave esperados
│   ├── utils.py                   # Helpers: load_gold, answer_matches
│   ├── test_rag_eval.py           # Pytest principal
│   └── calibrate.py               # Script para afinar k / score_threshold
├── tools/                         # Scripts utilitários (ex: mineração de tom)
│   └── minerar_tone.py
├── .env                           # Variáveis de ambiente
├── .gitignore                     # Ignorar arquivos sensíveis/temporários
├── docker-compose.yml             # Compose para subir Qdrant facilmente
├── main.py                        # Ponto de entrada do sistema
├── README.md                      # Esta documentação
└── requirements.txt               # Dependências Python
```

---

## ⚙️ Configuração Inicial

1. **Clone o repositório** e entre na pasta:

   ```bash
   git clone <repo-url>
   cd LangChain
   ```

2. **Crie e ative o ambiente virtual:**

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate        # Windows PowerShell
   # ou
   source .venv/bin/activate       # bash/macOS/Linux
   ```

3. **Instale as dependências:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure as chaves no `.env`:**

   ```dotenv
   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   QDRANT_API_KEY=sua-chave-supersecreta
   ```

---

## 🚀 Subindo o Qdrant

Você precisa de um Qdrant acessível em `localhost:6333` (ou Qdrant Cloud).

### **A. Docker Compose (Recomendado)**

**Suba o Qdrant com Compose**:

```bash
docker-compose up -d
```

* Para parar depois:

  ```bash
  docker-compose stop
  ```
* Para remover (sem apagar volume!):

  ```bash
  docker-compose down
  ```

Acesse o painel em [http://localhost:6333/dashboard](http://localhost:6333/dashboard)
Quando solicitado, use a mesma API key do `.env`.

### **B. Docker manual (alternativo)**

```bash
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -e QDRANT__SERVICE__API_KEY=sua-chave-supersecreta \
  -v ${PWD}/qdrant_data:/qdrant/storage \
  qdrant/qdrant
```

### **C. Qdrant Cloud**

* Cadastre-se em [https://cloud.qdrant.io](https://cloud.qdrant.io) e crie um cluster.
* Configure a URL e APIKey em `src/vector_store.py`:

```python
_client = QdrantClient(
    url="https://<YOUR_ID>.cloud.qdrant.io",
    api_key="<YOUR_API_KEY>",
)
```

---

## 📄 Atualizando o Contexto com o Word

Para usar o Word como fonte:

* Salve o documento oficial da Reforma Tributária (.docx) na pasta desejada.
* **Adapte o `data_loader.py`** para carregar e dividir o conteúdo:

```python
from docx import Document as DocxDocument
from langchain.schema import Document

def load_word_docs(path="docs/reforma.docx") -> list:
    docx = DocxDocument(path)
    docs = []
    for i, para in enumerate(docx.paragraphs):
        if para.text.strip():
            docs.append(Document(page_content=para.text.strip(), metadata={"id": f"word-{i+1}"}))
    return docs
```

* Em `main.py`, troque a linha:

  ```python
  raw_docs = load_word_docs("docs/reforma.docx")
  ```
* **Rodando o main.py** vai atualizar o banco vetorial automaticamente.

---

## ▶️ Executando a Aplicação

Com o Qdrant rodando e o venv ativo:

```bash
python main.py
```

Você verá:

```
INFO ... Qdrant: xx chunks adicionados.
Pipeline RAG pronto! (digite 'sair' ou Ctrl+C para finalizar)
Pergunta>
```

Faça perguntas em português — o sistema responde **apenas** com base no documento carregado (não alucina).

---

## 🧪 Testes de Regressão

1. Instale o pytest:

   ```bash
   pip install pytest
   ```

2. Configure PYTHONPATH (caso não esteja em modo editable):

   ```bash
   # PowerShell
   $env:PYTHONPATH = $PWD
   # bash/macOS/Linux
   export PYTHONPATH="$PWD"
   ```

3. Execute:

   ```bash
   pytest -q
   ```

4. Para calibrar parâmetros (k, threshold):

   ```bash
   python tests/calibrate.py
   ```

---

## 💬 Tom da Resposta

O bot detecta automaticamente o tom da pergunta (formal, informal, irritado...) e adapta a resposta.
Exemplo:

* **Pergunta formal:**
  *"Poderia informar quem gere o IBS?"*
  → Resposta polida

* **Pergunta informal:**
  *"E aí, quem manda no IBS?"*
  → Resposta descontraída

---

## 🛠 Próximos Passos

1. **Melhorar chunking:** usar splitters avançados para seções longas do Word/PDF.
2. **Filtros avançados:** buscar só trechos de certas leis/artigos.
3. **API ou interface web:** expor como endpoint Flask/FastAPI ou chat web.
4. **CI/CD:** automação de testes com Docker+Qdrant (GitHub Actions).
5. **Logs e mineração de tom:** use o script `tools/minerar_tone.py` para turbinar o classificador local.

---

## 📖 Referências

* [Qdrant Quick-Start (Docker)](https://qdrant.tech/documentation/quick-start/)
* [LangChain QdrantVectorStore](https://python.langchain.com/docs/integrations/vectorstores/qdrant)
* [OpenAI Embeddings & Chat](https://platform.openai.com/docs/)

---

## 👥 Contribua

Sugestões, issues ou PRs são bem-vindos!
Adicione exemplos de uso, prompts ou sugestões para facilitar a adoção do RAG na sua empresa.

---

## 🤝 Licença

MIT © Renata Boppré Scharf

---

**Pronto para uso, treinamento ou evolução!**
Se tiver qualquer dúvida, só abrir um issue ou perguntar.

---