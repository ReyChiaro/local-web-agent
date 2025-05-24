from config import LLM_NAME, MODEL_PATH, CLI_PATH
from typing import Optional


class LLMWrapper:

    def __init__(
        self,
        llm_name: str=LLM_NAME,
        model_path: Optional[str]=MODEL_PATH,
        cli_path: Optional[str]=CLI_PATH,
    ):
        self.llm_name = llm_name
        if self.llm_name == "bitnet":
            from models.BitNet import Bitnet
            self.model = Bitnet(model_path, cli_path)
        else:
            # TODO: Other models & tokenizers initialization with provided API
            pass
    
    def generate(self, prompt: str) -> str:
        if self.llm_name == "bitnet":
            outputs = self.model.generate(prompt)
        else:
            # TODO: Other models generation
            pass
        return outputs
        

