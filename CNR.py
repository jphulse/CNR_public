import numpy as np
import pandas as pd
from typing import List
from typing import Optional




def CNR(graphs : List[np.ndarray], p: Optional[float] =.5) -> np.ndarray:
    if len(graphs) < 1:
        raise ValueError("graphs cannot be an empty list for CNR")
    if p == None or p < 0 or p > 1:
        raise ValueError("p value must be between 0 and 1 inclusive for CNR")
    dim = graphs[0].shape[0]
    consensus_graph = np.zeros(dim, dim)
    for matrix in graphs:
        for i in range(dim):
            for j in range(dim):
                consensus_graph[i, j] += matrix[i, j]
    for i in range(dim):
        for j in range(dim):
            consensus_graph[i, j] = 1 if consensus_graph[i, j] / len(list) > p else 0

    return consensus_graph 