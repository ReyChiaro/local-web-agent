class ConversationHistory:

    def __init__(self):
        self.history: list[tuple[str]] = []
    
    def add_turn(self, user_msg: str, assistant_msg: str):
        self.history.append((user_msg, assistant_msg))
    
    def get_last_n_turns(self, n:int=3) -> list[tuple[str]]:
        return self.history[-n:]
    