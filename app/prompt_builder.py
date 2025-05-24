class PromptBuilder:

    def __init__(self, system_prompt="You are a helpful assistant."):
        self.system_prompt = system_prompt
    
    def build_prompt(self, history: list[tuple[str]], query: str, docs: list[str]):
        doc_section = "\n\n".join(f"[Document] {doc}" for doc in docs)
        chat_section = ""
        for user, assistant in history:
            chat_section += f"\nUser: {user}\nAssistant: {assistant}"
        prompt = (
            f"{self.system_prompt}\n\n"
            f"{doc_section}\n\n"
            f"{chat_section}\n"
            f"User: {query}\n"
            f"Assistant:"
        )
        return prompt