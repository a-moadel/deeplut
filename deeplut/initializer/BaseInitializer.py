from typing import Optional
from typing import Dict, Optional

class BaseInitializer:
    device: Optional[str]

    def __init__(self, table_count, k, kk, device: Optional[str]) -> None:
        self.table_count = table_count
        self.k = k
        self.kk = kk
        self.device = device
        
