import numpy as np
import enum as Enum
from collections import deque
from environment.env_var import Box,OneD

class SupplyChainEnv:
    def __init__(self):
        #consider number of different products
        self.product_types_num=2
        #consider number of distribution warehouses
        self.distribution_warehouse=2
        #final time step
        self.T=30
        