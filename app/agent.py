import torch

from app.conversation import ConversationHistory
from app.retriever import Retriever
from app.prompt_builder import PromptBuilder
from app.query_refine import QueryRefiner


class WebAgent:

    def __init__(
        self,
        retriever: Retriever,
        llm: torch.nn.Module,
        prompt_builder: PromptBuilder,
    ):
        self.history = ConversationHistory()
        self.retriever = retriever
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.refiner = QueryRefiner(retriever.doc_chunks, llm)

    def chat(self, user_input: str, embed_query_fn: callable) -> str:
        refined_query = self.refiner.refine(user_input)
        print(f"( 👉 Refined query: {refined_query})")
        query_vec = embed_query_fn(refined_query)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        docs = self.retriever.retrieve(query_vec)
        prompt = self.prompt_builder.build_prompt(self.history.get_last_n_turns(), user_input, docs)
        response = self.llm.generate(prompt)
        self.history.add_turn(user_input, response)
        return response