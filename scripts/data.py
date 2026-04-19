import torch
import numpy as np
import networkx as nx
from src.utils.modules import Core, IOPair

def generateSamples(num_samples):
    """
    Generates a specified number of samples, each containing a 64x2 array of directional data.

    Args:
        num_samples (int): The number of samples to generate.

    Returns:
        torch.Tensor: A tensor containing the generated samples.
    """
    data = []
    for i in range(num_samples):
        dir = np.zeros(shape=[65,2])
        for i in range(0,65):
            if i%4==1:
                if i<32:
                    dir[i,1]=0
                    dir[i,0]=i//4
                else:
                    dir[i,1]=4
                    dir[i,0]=(i-32)//4
            elif i%4==2:
                if i<32:
                    dir[i,1]=1
                    dir[i,0]=i//4
                else:
                    dir[i,1]=5
                    dir[i,0]=(i-32)//4
            elif i%4==3:
                if i<32:
                    dir[i,1]=2
                    dir[i,0]=i//4
                else:
                    dir[i,1]=6
                    dir[i,0]=(i-32)//4
            else:
                if i<=32:
                    dir[i,1]=3
                    dir[i,0]=i//4 - 1
                else:
                    dir[i,1]=7
                    dir[i,0]=(i-32)//4 - 1
        dir = dir[1:]
        data.append(dir)
    data = torch.from_numpy(np.array(data).astype('float32'))
    return data

def create_data(batch_size, num_cores, cols):
    """
    Generates a batch of data representing core positions in a grid.

    Args:
        batch_size (int): The number of data samples to generate.
        num_cores (int): The total number of cores.
        cols (int): The number of columns in the grid.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, num_cores, 2) containing the positions of the cores.
    """
    data=[]
    dir = np.zeros(shape=[num_cores,2])
    for i in range(num_cores):
        dir[i,0] = i//cols
        dir[i,1] = i%cols
    for i in range(batch_size):
        data.append(dir)
    data = torch.from_numpy(np.array(data).astype('float32'))
    return data

def prep_data(num_cores, test=True):
    """
    Prepares data for a given number of cores.
    Parameters:
    num_cores (int): The number of cores to prepare data for.
    test (bool): If True, use benchmark data files. Default is True.
    Returns:
    tuple: A tuple containing:
        - data (various): The prepared data.
        - cores (list): A list of lists, each containing Core objects.
        - io (list): A list of lists, each containing I/O pairs.
        - dims (list): A list containing the dimensions of the graph.
        - graph (networkx.Graph): The graph representing the core connections.
    """

    file = open(f"data_{num_cores}cores.txt", 'r')
    if test:
        file = open(f"bm_{num_cores}cores.txt", 'r')
    lines = file.readlines()
    batch_size = len(lines)//num_cores
    cores = [[] for _ in range(batch_size)]
    for i,line in enumerate(lines):
        data = line.split()
        cores[i//num_cores].append(Core(int(data[0]), data[1], int(data[2]), int(data[3]), int(data[4])))


    data = None
    graph = None
    dims = None
    if num_cores == 7:
        io = [[2,7], [3,6]]
        graph = nx.Graph()
        for i in range(1,8):
            graph.add_node(i)
        edges=[]
        first1=[1, 2, 3, 4, 1, 2, 5, 6]
        second1=[4, 5, 6, 7, 2, 3, 4, 5]
        for i in range(len(first1)):
            edge=[first1[i], second1[i]]
            edges.append(edge)
        graph.add_edges_from(edges)
        dims = [3,3]
        data = create_data(batch_size, 7, 3)
    elif num_cores == 8:
        io = [[1,2], [3,7]]
        graph=nx.Graph()
        for i in range(1,8):
          graph.add_node(i)
        edges=[]
        first1 = [1,1,2,2,3,3,4,5,6,7]
        second1 = [2,5,3,6,4,7,8,6,7,8]
        for i in range(len(first1)):
            edge=[first1[i], second1[i]]
            edges.append(edge)
        graph.add_edges_from(edges)
        dims = [2,4]
        data = create_data(batch_size,8,4)
    elif num_cores == 10:
        io = [[1,7], [2,6], [3,10]]
        graph = nx.Graph()
        for i in range(1,11):
            graph.add_node(i)
        edges=[]
        first1=[1 ,2 ,3 ,4, 1, 2, 3, 5, 6, 5, 6, 7, 9]
        second1=[5, 6, 7, 8, 2, 3, 4, 9, 10, 6, 7, 8, 10]
        for i in range(len(first1)):
            edge=[first1[i], second1[i]]
            edges.append(edge)
        graph.add_edges_from(edges)
        dims = [3,4]
        data = create_data(batch_size, 10, 4)
    elif num_cores == 14:
        io = [[3,6], [4,10]]
        graph = nx.Graph()
        for i in range(1,15):
            graph.add_node(i)
        edges=[]
        first1=[1 ,2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]
        second1=[8, 9, 10, 11, 12, 13, 14, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14]
        for i in range(len(first1)):
            edge=[first1[i], second1[i]]
            edges.append(edge)
        graph.add_edges_from(edges)
        dims = [2,7]
        data = create_data(batch_size, 14, 7)
    elif num_cores == 16:
        io = [[2,4], [5,9], [13,14], [12,16]]
        graph = nx.Graph()
        for i in range(1,17):
            graph.add_node(i)
        edges=[]
        first1=[1,1,2,2,3,3,4,5,5,6,6,7,7,8,9,9,10,10,11,11,12,13,14,15]
        second1=[2,5,3,6,4,7,8,6,9,7,10,8,11,12,13,10,14,11,15,12,16,14,15,16]
        for i in range(len(first1)):
            edge=[first1[i], second1[i]]
            edges.append(edge)
        graph.add_edges_from(edges)
        dims = [4,4]
        data = create_data(batch_size, 16, 4)
    elif num_cores == 28:
        io = [[5,12], [8,16], [9,17], [13,20]]
        graph = nx.Graph()
        for i in range(1,29):
            graph.add_node(i)
        edges=[]
        first1=[1, 2, 3, 1, 2, 3, 4, 5, 6, 7, 5, 6, 7, 8, 9, 10, 11, 9, 10, 11, 12, 13, 14, 15, 13, 14, 15, 16, 17, 18, 19, 17, 18, 19, 20, 21, 22, 23, 21, 22, 23, 24, 25, 26, 27]
        second1=[2, 3, 4, 5, 6, 7, 8, 6, 7, 8, 9, 10, 11, 12, 10, 11, 12, 13, 14, 15, 16, 14, 15, 16, 17, 18, 19, 20, 18, 19, 20, 21, 22, 23, 24, 22, 23, 24, 25, 26, 27, 28, 26, 27, 28]
        for i in range(len(first1)):
            edge=[first1[i], second1[i]]
            edges.append(edge)
        graph.add_edges_from(edges)
        dims = [7,4]
        data = create_data(batch_size, 28, 4)
    elif num_cores == 32:
        io = [[1,4], [9,12], [17,20], [25,28]]
        graph = nx.Graph()
        for i in range(1,33):
            graph.add_node(i)
        edges=[]
        first1=[1, 2, 3, 1, 2, 3, 4, 5, 6, 7, 5, 6, 7, 8, 9, 10, 11, 9, 10, 11, 12, 13, 14, 15, 13, 14, 15, 16, 17, 18, 19, 17, 18, 19, 20, 21, 22, 23, 21, 22, 23, 24, 25, 26, 27, 25, 26, 27, 28, 29, 30, 31]
        second1=[2, 3, 4, 5, 6, 7, 8, 6, 7, 8, 9, 10, 11, 12, 10, 11, 12, 13, 14, 15, 16, 14, 15, 16, 17, 18, 19, 20, 18, 19, 20, 21, 22, 23, 24, 22, 23, 24, 25, 26, 27, 28, 26, 27, 28, 29, 30, 31, 32, 30, 31, 32]
        for i in range(len(first1)):
            edge=[first1[i], second1[i]]
            edges.append(edge)
        graph.add_edges_from(edges)
        dims = [8,4]
        data = create_data(batch_size, 32, 4)
    elif num_cores == 64:
        io = [[13,17], [33,40], [44,52], [61,63]]
        graph = nx.Graph()
        for i in range(1,65):
            graph.add_node(i)
        edges=[]
        first1=[1, 2, 3, 1, 2, 3, 4, 5, 6, 7, 5, 6, 7, 8, 9, 10, 11, 9, 10, 11, 12, 13, 14, 15, 13, 14, 15, 16, 17, 18, 19, 17, 18, 19, 20, 21, 22, 23, 21, 22, 23, 24, 25, 26, 27, 25, 26, 27, 28, 29, 30, 31]
        second1=[2, 3, 4, 5, 6, 7, 8, 6, 7, 8, 9, 10, 11, 12, 10, 11, 12, 13, 14, 15, 16, 14, 15, 16, 17, 18, 19, 20, 18, 19, 20, 21, 22, 23, 24, 22, 23, 24, 25, 26, 27, 28, 26, 27, 28, 29, 30, 31, 32, 30, 31, 32]
        for i in range(len(first1)):
            edge=[first1[i], second1[i]]
            edges.append(edge)
        graph.add_edges_from(edges)

        edges=[]
        first1=[1, 2, 3, 1, 2, 3, 4, 5, 6, 7, 5, 6, 7, 8, 9, 10, 11, 9, 10, 11, 12, 13, 14, 15, 13, 14, 15, 16, 17, 18]
        second1=[2, 3, 4, 5, 6, 7, 8, 6, 7, 8, 9, 10, 11, 12, 10, 11, 12, 13, 14, 15, 16, 14, 15, 16, 17, 18, 19, 19, 18, 19]
        for i in range(len(first1)):
            if first1[i]<17 and second1[i]<17:
                edge=[first1[i]+32, second1[i]+32]
                edges.append(edge)
        graph.add_edges_from(edges)

        edges=[]
        first1=[1 ,2 ,3 ,1 ,2, 3, 4, 5, 6, 7, 5, 6, 7, 8, 9, 10, 11, 9, 10, 11, 12, 13, 14, 15, 13, 14, 15, 16, 17, 18, 19, 17, 18, 19, 20, 21, 22, 23, 21, 22, 23, 24, 25, 26, 27 ]
        second1=[2, 3, 4, 5, 6, 7, 8, 6, 7, 8, 9, 10, 11, 12, 10, 11, 12, 13,14, 15, 16, 14, 15, 16, 17, 18, 19, 20, 18, 19, 20, 21, 22, 23, 24, 22, 23, 24, 25, 26, 27, 28, 26, 27, 28]

        for i in range(len(first1)):
            if first1[i]<17 and second1[i]<17:
                edge=[first1[i]+48, second1[i]+48]
                edges.append(edge)
        graph.add_edges_from(edges)

        for i in range(4,33,4):
            graph.add_edge(i, i+29)

        for i in range(45,49):
            graph.add_edge(i,i+4)
        dims = [8,8]
        data = generateSamples(batch_size)
        #data = create_data(batch_size, 40, 5)
    elif num_cores == 40:
        io = [[1, 35], [4,12], [30,32], [37, 39]]
        graph = nx.Graph()
        for i in range(1,41):
            graph.add_node(i)
        edges=[]
        first1=[1, 2, 3, 1, 2, 3, 4, 5, 6, 7, 5, 6, 7, 8, 9, 10, 11, 9, 10, 11, 12, 13, 14, 15, 13, 14, 15, 16, 17, 18, 19, 17, 18, 19, 20, 21, 22, 23, 21, 22, 23, 24, 25, 26, 27, 25, 26, 27, 28, 29, 30, 31]
        second1=[2, 3, 4, 5, 6, 7, 8, 6, 7, 8, 9, 10, 11, 12, 10, 11, 12, 13, 14, 15, 16, 14, 15, 16, 17, 18, 19, 20, 18, 19, 20, 21, 22, 23, 24, 22, 23, 24, 25, 26, 27, 28, 26, 27, 28, 29, 30, 31, 32, 30, 31, 32]
        for i in range(len(first1)):
            edge=[first1[i], second1[i]]
            edges.append(edge)
        graph.add_edges_from(edges)

        first1 = [1, 2, 3, 4, 5, 6, 7]
        second1 = [2, 3, 4, 5, 6, 7, 8]

        for i in range(len(first1)):
            edge = [second1[i]+32, first1[i]+32]
            edges.append(edge)
        graph.add_edges_from(edges)

        for i in range(0, 8):
            edge  = [4*i+1, 33+i]
            edges.append(edge)
        graph.add_edges_from(edges)
        dims = [8,5]
        #data = generateSamples(batch_size)
        data = create_data(batch_size, 40, 5)
    elif num_cores == 48:
        io = [[3,5], [21,30], [32,34], [38, 40]]
        graph = nx.Graph()
        for i in range(1,49):
            graph.add_node(i)
        edges=[]
        first1=[1, 2, 3, 1, 2, 3, 4, 5, 6, 7, 5, 6, 7, 8, 9, 10, 11, 9, 10, 11, 12, 13, 14, 15, 13, 14, 15, 16, 17, 18, 19, 17, 18, 19, 20, 21, 22, 23, 21, 22, 23, 24, 25, 26, 27, 25, 26, 27, 28, 29, 30, 31]
        second1=[2, 3, 4, 5, 6, 7, 8, 6, 7, 8, 9, 10, 11, 12, 10, 11, 12, 13, 14, 15, 16, 14, 15, 16, 17, 18, 19, 20, 18, 19, 20, 21, 22, 23, 24, 22, 23, 24, 25, 26, 27, 28, 26, 27, 28, 29, 30, 31, 32, 30, 31, 32]
        for i in range(len(first1)):
            edge=[first1[i], second1[i]]
            edges.append(edge)
        graph.add_edges_from(edges)


        edges=[]
        first1= [1, 2, 3, 4, 5 ,6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 2, 3, 4, 5, 6, 7, 8]
        second1=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,16,15,14,13,12,11,10,9]
        for i in range(len(first1)):
            edge=[first1[i]+32, second1[i]+32]
            edges.append(edge)
        graph.add_edges_from(edges)

        for i in range(1, 9):
            edge = [4*i, i+40]
            edges.append(edge)
            graph.add_edges_from(edges)
        dims = [8,6]
        #data = generateSamples(batch_size)
        data = create_data(batch_size, 48, 6)
    elif num_cores == 56:
        edges = []
        io = [[3,5], [21,30], [32,34], [38, 40]]
        graph = nx.Graph()
        for i in range(1, 57):
            graph.add_node(i)
        first1=[1, 2, 3, 1, 2, 3, 4, 5, 6, 7, 5, 6, 7, 8, 9, 10, 11, 9, 10, 11, 12, 13, 14, 15, 13, 14, 15, 16, 17, 18, 19, 17, 18, 19, 20, 21, 22, 23, 21, 22, 23, 24, 25, 26, 27, 25, 26, 27, 28, 29, 30, 31]
        second1=[2, 3, 4, 5, 6, 7, 8, 6, 7, 8, 9, 10, 11, 12, 10, 11, 12, 13, 14, 15, 16, 14, 15, 16, 17, 18, 19, 20, 18, 19, 20, 21, 22, 23, 24, 22, 23, 24, 25, 26, 27, 28, 26, 27, 28, 29, 30, 31, 32, 30, 31, 32]
        for i in range(len(first1)):
            edge=[first1[i], second1[i]]
            edges.append(edge)
        graph.add_edges_from(edges)
   
        edges=[]
        first1= [1, 2, 3, 4, 5 ,6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 2, 3, 4, 5, 6, 7, 8]
        second1=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,16,15,14,13,12,11,10,9]
        for i in range(len(first1)):
            edge=[first1[i]+32, second1[i]+32]
            edges.append(edge)
        graph.add_edges_from(edges)
        
        edges=[]       
        first1 = [1, 2, 3, 4, 5, 6, 7]
        second1 = [2, 3, 4, 5, 6, 7, 8]

        for i in range(len(first1)):
            edge = [second1[i]+48, first1[i]+48]
            edges.append(edge)
        graph.add_edges_from(edges)

        for i in range(1, 9):
            edge = [4*i, i+40]
            edges.append(edge)
        graph.add_edges_from(edges)

        for i in range(33, 41):
            edge = [i, i+16]
            edges.append(edge)
        graph.add_edges_from(edges)
        dims = [8,7]
        #data = generateSamples(batch_size)
        data = create_data(batch_size, 56, 7)

    return data, cores, io, dims, graph