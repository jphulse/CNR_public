import os 
import re 
import csv
import sys
import random
# pip install portalocker
from typing import List

# Data formattingimports
import pandas as pd
import numpy as np

from pandas import DataFrame
from numpy import ndarray
# pip install causallearn
# causal learning imports
# Causallearn is a novel package for causal discovery associated with a 2024
# paper which is already being cited within other new literature, already having 47 citations 
# in the field make this a new representation of the latest and greatest in causal discovery
# and the 30 years of work that have gone into improving these methods
from causallearn.search.ConstraintBased import PC
from causallearn.search.ConstraintBased import FCI
from causallearn.search.ScoreBased import GES 
from causallearn.search.FCMBased import lingam
# Not currently used, but provides utilities to draw the graphs and other fun stuff to play with, so I left it in
from causallearn.utils.GraphUtils import GraphUtils

# data processing or modification imports
from scipy import stats
import statsmodels.api as sm

# arbitrary, mainly a formality for additional data added by users for testing or refutal, will remove columns with no variance
def fix_invariance(df : DataFrame):

    return df.loc[:, df.var() != 0]

# fixes normal data, currently only fixes invariance, could be modified to remove non-numeric cols in the future, for now that must be done manually
def fix_normal_data(df :DataFrame):
    df = fix_invariance(df)
    return df

# Takes a 2D edgelist and turns them into an adjacency matrix for comparisons
# used for the graphs output by the pc algorithm
def turnEdgesToMatrix(edges, dim):
    matrix = np.zeros((dim, dim))
    for i, j in edges:
        matrix[i, j] = 1
    
    return matrix

# Turns an edgelist  from the fci part of causallearn into an adjacency matrix
# IMPORTANT only works on data labeled X1, X2, X3 ... XN, intended for use with fci output
def turnEdgeListToMatrix(list, dim):
    edges = []
    for edge in list:
        e1 = edge.get_numerical_endpoint1()
        e2 = edge.get_numerical_endpoint2()
        if e1 == -1:
            edges.append((int(edge.get_node1().get_name()[1:]) - 1, int(edge.get_node2().get_name()[1:]) - 1))
        elif (e1 == 2 and e2 != 1) or (e1 == 1 and e2 == 1):
            edges.append((int(edge.get_node1().get_name()[1:]) - 1, int(edge.get_node2().get_name()[1:]) - 1))
    
    return turnEdgesToMatrix(edges, dim)

# GES adjacency matrix process is slightly different from the others and done here
# this is done this way currently, although edges where there is a -1 are less
# certain, treated as undirected the same way they are handles in PC by converting them to 
# 1 on both sides, violates the DAG for the sake of consistency and fairness
def getGESAdjacencyMatrix(graph, dim):
    for i in range(dim):
        for j in range(dim):
            if graph[i, j] == -1:
                graph[i, j] = 1
    
    return graph

# Adjusts the LiNGAM matrix to fit our model for evaluation, there is potential to add an additional parameter to require a certain minimal threshold effect
# to be considered an edge, this would likely improve stability at the cost of edgecount but may not be helpful in accuracy
# potential area for hyperparameter optimization for stability
def modifyLinGamAdjacencyMatrix(mat, dim):
    for i in range(dim):
        for j in range(dim):
            if mat[i, j] != 0:
                mat[i, j] = 1
    return mat

# Calculates the Jaccard Index of two adjacency matrix causal graphs
# of size dim, the equation for this is the edges in g1 AND g2 over the edges
# in g1 OR g2, and this will be a value between 0 (least similar) and 1 (most similar)
def calculateJaccardIndex(g1, g2, dim):
    bothCount = 0
    eitherCount = 0
    for i in range(dim):
        for j in range(dim):
            if g1[i, j] == 1 or g2[i, j] == 1:
                eitherCount += 1
                if g1[i, j] == g2[i, j]:
                    bothCount += 1
    return bothCount / eitherCount if eitherCount > 0 else 0

# performs the pc algorithm and returns the appropriately formatted matrix                
def performPC(grid, dim, alph=.01):
    print(grid)
    print(grid.shape[0], grid.shape[1] )
    pc_graph = PC.pc(grid, show_progress=False, alpha=alph)
    return turnEdgesToMatrix(pc_graph.find_adj(), dim)

# performs the FCI alg and returns the formatted matrix
def performFCI(grid, dim, alph =.01):
    _, fci_list = FCI.fci(grid, show_progress=False, alpha=alph)
    return turnEdgeListToMatrix(fci_list, dim)

# performs the GES alg and returns the formatted matrix
def performGES(grid, dim):
    ges_graph = GES.ges(grid)
    return getGESAdjacencyMatrix(ges_graph['G'].graph, dim)

# performs the LiNGAM alg and returns the formatted matrix
def performLiNGAM(grid, dim):
    lin_model = lingam.DirectLiNGAM()
    lin_model.fit(grid)
    return modifyLinGamAdjacencyMatrix(lin_model.adjacency_matrix_, dim)




# Makes the four kinds of graphs for the REAl data given the 
# grid representation of the dataframe associated with the sample
def makeGraphs(grid):
    dim = grid.shape[1]
    pc_matrix = performPC(grid, dim)
    fci_matrix = performFCI(grid, dim)
    ges_matrix = performGES(grid, dim)
    lin_matrix = performLiNGAM(grid, dim)
    return pc_matrix, fci_matrix, ges_matrix, lin_matrix