import torch


class QueryRefiner:

    def __init__(self, sample_docs: list[str], llm: torch.nn.Module):
        self.sample_docs = sample_docs
        self.llm = llm
    
    def refine(self, original_query: str) -> str:
        context = "\n".join(self.sample_docs)
        prompt = (
            f"You are a helpful assistant that filters questions based on the following documents.\n\n"
            f"Documents:\n{context}\n\n"
            f"Original question:\n\"{original_query}\"\n\n"
            f"Please extract the sub-question that can be answered using the above documents.\nAnswer:"
        )
        response = self.llm.generate(prompt)
        return response