import subprocess
import torch.nn as nn

from config import (
    LLM_NAME,
    MAX_TOKENS,
    TEMPERATURE,
    CTX_SIZE, 
    N_THREADS,
)

class Bitnet(nn.Module):

    def __init__(
        self,
        model_path: str,
        cli_path: str,
    ):
        self.model_path = model_path
        self.cli_path = cli_path

    def generate(
        self,
        prompt: str,
        max_tokens: int=MAX_TOKENS,
        temperature: float=TEMPERATURE,
    ) -> str:
        command = [
            self.cli_path,
            "-m", self.model_path,
            "-n", str(max_tokens),
            "-t", str(N_THREADS),
            "-p", prompt,
            "-ngl", '0',
            "-c", str(CTX_SIZE),
            "--temp", str(temperature),
            "-b", "1",
            # "-cnv",
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True, shell=False)
        return result.stdout.strip()
    
