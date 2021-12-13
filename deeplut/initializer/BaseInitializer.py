from typing import Optional
from typing import Dict, Optional

class BaseInitializer:
    device: Optional[str]

    def __init__(self, table_count, k, kk, weight_lookup_table: Dict, device: Optional[str]) -> None:
        self.table_count = table_count
        self.k = k
        self.kk = kk
        self.weight_lookup_table = weight_lookup_table
        self.device = device
        
