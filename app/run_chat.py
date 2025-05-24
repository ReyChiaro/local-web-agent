from sentence_transformers import SentenceTransformer

from app.agent import WebAgent
from app.retriever import Retriever
from app.llm_wrapper import LLMWrapper
from app.prompt_builder import PromptBuilder
from config import (
    VECTOR_DB_PATH,
    DOC_STORE_PATH,
    TOP_K_DOCS,
    LLM_NAME,
    MODEL_PATH,
    CLI_PATH,
    CHAT_LOG_PATH,
    EMBEDDING_MODEL_NAME,
)


def run_chat():
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    retriever = Retriever(
        index_path=str(VECTOR_DB_PATH),
        doc_path=DOC_STORE_PATH,
        top_k=TOP_K_DOCS,
    )
    llm = LLMWrapper(
        llm_name=LLM_NAME,
        model_path=MODEL_PATH,
        cli_path=CLI_PATH,
    )
    prompt_builder = PromptBuilder()
    agent = WebAgent(
        retriever=retriever,
        llm=llm,
        prompt_builder=prompt_builder,
    )

    while True:
        user_input = input("\n ✅ (You can type `exit` or `quit` to end the conversation)\n 🧑 : ")
        if user_input.lower() in {"exit", "quit"}:
            break
        response = agent.chat(user_input, embed_model.encode)
        print(f"\n 😃 : {response}")
        with open(CHAT_LOG_PATH, "a") as f:
            f.write(f"User: {user_input}\nAgent: {response}\n\n")

# if __name__ == "__main__":
#     run_chat()
