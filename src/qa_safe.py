from langchain.chains import RetrievalQA


class SafeRetrievalQA(RetrievalQA):
    """RetrievalQA que devolve fallback padronizado quando não há contexto."""
    # ↓ Mantém a factory de conveniência da classe‑mãe
    @classmethod
    def from_chain_type(cls, *args, **kwargs):  # type: ignore[override]
        qa = super().from_chain_type(*args, **kwargs)
        qa.__class__ = cls
        return qa

    def _call(self, inputs: dict, run_manager=None):        # noqa: N802
        tone = inputs.pop("tone", "objetivo")
        question = inputs[self.input_key]

        docs = self._get_docs(question, run_manager=run_manager)
        chain_inputs = {
            "input_documents": docs,
            "question": question,
            "tone": tone,
        }
        invoke_result = self.combine_documents_chain.invoke(
            chain_inputs,
            callbacks=run_manager.get_child() if run_manager else None,
        )
        answer = invoke_result["output_text"]

        out = {"result": answer}
        if self.return_source_documents:
            out["source_documents"] = docs
        if not docs:
            out["result"] = "Desculpe, não sei essa informação."
        return out
