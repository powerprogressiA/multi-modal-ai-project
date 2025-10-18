# Minimal simulated sensor used by tests/demo.
# Adjust to match your real implementation.
from typing import List
import random

class SimulatedSensor:
    def __init__(self, length: int = 256):
        self.length = length

    def get_frame(self) -> List[float]:
        # return a simple simulated signal (list of floats)
        return [random.random() for _ in range(self.length)]
