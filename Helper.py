from typing import Optional, Final
import enum

class Helper:
    COLOR_ERROR: Final = 31
    COLOR_SUCCESS: Final = 32
    COLOR_WARNING: Final = 33
    COLOR_INFO: Final = 36
    
    @staticmethod
    def log(message: str, color: Optional[int] = None) -> str :
        if (color is None): return f"{message}\n"
        return f"\033[{color}m{message} \033[0m\n"
    

class Synapses(enum.Enum):
    input_to_hidden = 1
    hidden_to_output = 2
