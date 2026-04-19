import torch
import numpy as np
import networkx as nx
from queue import PriorityQueue

def hop_counts(G, source, target):
    """
    Calculate the shortest path length (hop count) between two nodes in a graph.

    Parameters:
    G (networkx.Graph): The graph in which to find the shortest path.
    source (node): The starting node for the path.
    target (node): The ending node for the path.

    Returns:
    int: The number of edges in the shortest path from source to target.

    Raises:
    NetworkXNoPath: If there is no path between source and target.
    NodeNotFound: If either source or target is not in the graph.
    """
    return nx.shortest_path_length(G, source, target)

def check_path_conflict(dir, core1, src1, sink1, core2, src2, sink2):
    """
    Checks if there is a path conflict between two sets of source, core, and sink coordinates.

    Args:
        dir (list): A list where elements are tuples of (y, x) coordinates.
        core1 (int): Identifier for the core node in the first path.
        src1 (int): Identifier for the source node in the first path.
        sink1 (int): Identifier for the sink node in the first path.
        core2 (int): Identifier for the core node in the second path.
        src2 (int): Identifier for the source node in the second path.
        sink2 (int): Identifier for the sink node in the second path.

    Returns:
        bool: True if there is a conflict between the two paths, False otherwise.
    """

    ys1 = dir[src1][0].item()
    xs1 = dir[src1][1].item()
    yc1 = dir[core1][0].item()
    xc1 = dir[core1][1].item()
    yk1 = dir[sink1][0].item()
    xk1 = dir[sink1][1].item()
    ys2 = dir[src2][0].item() 
    xs2 = dir[src2][1].item()
    yc2 = dir[core2][0].item()
    xc2 = dir[core2][1].item()
    yk2 = dir[sink2][0].item()
    xk2 = dir[sink2][1].item()

  # print(xc1, yc1)
  # print(xs1, ys1)
  # print(xk1, yk1)
  # print(xc2, yc2)
  # print(xs2, ys2)
  # print(xk2, yk2)

    if (xs1-xc1)*(xs2-xc2) > 0: 
        if not((xc1 <= min(xc2,xs2) and xs1 <= min(xc2,xs2)) or (xc1 >= max(xc2,xs2) and xs1 >= max(xc2,xs2))) and ys1==ys2:
            # print('F')
            return True
    if (xs1-xc1)*(xc2-xk2) > 0:
        if not((xc1 <= min(xc2,xk2) and xs1 <= min(xc2,xk2)) or (xc1 >= max(xc2,xk2) and xs1 >= max(xc2,xk2))) and ys1==yc2:
            # print('G')
            return True
    if (xc1-xk1)*(xs2-xc2) > 0:
        if not((xc1 <= min(xc2,xs2) and xk1 <= min(xc2,xs2)) or (xc1 >= max(xc2,xs2) and xk1 >= max(xc2,xs2))) and yc1==ys2:
            # print('H')
            return True
    if (xc1-xk1)*(xc2-xk2) > 0:
        if not((xc1 <= min(xc2,xk2) and xk1 <= min(xc2,xk2)) or (xc1 >= max(xc2,xk2) and xk1 >= max(xc2,xk2))) and yc1==yc2:
            # print('I')
            return True
    if (ys1-yc1)*(ys2-yc2) > 0:
        if not((yc1 <= min(yc2,ys2) and ys1 <= min(yc2,ys2)) or (yc1 >= max(yc2,ys2) and ys1 >= max(yc2,ys2))) and xc1==xc2: 
            # print('J')
            return True
    if (ys1-yc1)*(yc2-yk2) > 0:
        if not((yc1 <= min(yc2,yk2) and ys1 <= min(yc2,yk2)) or (yc1 >= max(yc2,yk2) and ys1 >= max(yc2,yk2))) and xc1==xk2:
            # print('K')
            return True
    if (yc1-yk1)*(ys2-yc2) > 0:
        if not((yc1 <= min(yc2,ys2) and yk1 <= min(yc2,ys2)) or (yc1 >= max(yc2,ys2) and yk1 >= max(yc2,ys2))) and xk1==xc2:
            # print('L')
            return True
    if (yc1-yk1)*(yc2-yk2) > 0:
        if not((yc1 <= min(yc2,yk2) and yk1 <= min(yc2,yk2)) or (yc1 >= max(yc2,yk2) and yk1 >= max(yc2,yk2))) and xk1==xk2:
            # print('M')
            return True

    return False

# (up, left, down, right) = (0, 1, 2, 3)
def objFunct(mappings, dir, cores, io_array, dims, graphs):
    """
    Objective function to evaluate core allocation efficiency.
    """
    # Step 1: Calculate the number of samples per core mapping
    num_samples = len(mappings) // len(cores) + 1

    # Step 2: Initialize lists to hold penalties and allocation orders
    penalty = []  # Stores penalties for each mapping
    orders = []   # Stores the order of core allocations for each mapping

    # Step 3: Iterate through each mapping in mappings
    for k, maps in enumerate(mappings):
        # Step 4: Prepare io_data to group cores by their associated IO paths
        io_data = [[] for _ in range(len(io_array))]
        for i, m in enumerate(maps):
            io_data[m].append(i)

        # Step 5: Initialize various arrays to track timings and allocation status
        start_time = [0] * len(maps)   # Step 8
        global_time = [0] * len(io_array)  # Step 7
        allocated = [False] * len(maps)  # Tracks whether a core has been allocated
        in_passive = [False] * len(maps)  # Tracks if a core is in the passive queue
        test_times = [[] for _ in range(len(io_array))]  # Step 9

        # Step 10: Calculate test times for each core in io_data
        for j in range(len(io_data)):
            for c in range(len(io_data[j])):
                core = io_data[j][c] + 1
                io = io_array[int(maps[core - 1])]
                src = io[0] - 1
                sink = io[1] - 1
                hsc = hop_counts(graphs, src + 1, core)  # Hop count source to core
                hck = hop_counts(graphs, core, sink + 1)  # Hop count core to sink
                l = cores[k // num_samples][j].scan  # Core's scanning capability
                p = cores[k // num_samples][j].patterns  # Core's pattern property
                time = (max(hsc, hck) + l) * p + (min(hsc, hck) + l - 1)  # Step 11
                test_times[j].append(time)

        # Step 13: Initialize allocation and passive priority queue
        alloc = []  # Stores allocated cores and their end times
        pass_arr = PriorityQueue()  # Passive queue for unallocated cores

        # Step 14: Main loop for allocation while there are unallocated cores
        while False in allocated:
            temp_passive = []  # Temporary list for resolving conflicts in the passive queue

            # Step 15: Process cores in the passive queue
            while not pass_arr.empty():
                strt, core = pass_arr.get()
                if allocated[core]:
                    continue

                is_conflict = False  # Flag to detect path conflicts
                io_idx = maps[core]
                core_in_io_indx = io_data[io_idx].index(core)

                for a, all in enumerate(alloc):
                    if start_time[core] < all[1]:
                        # Step 16: Check for path conflicts with already allocated cores
                        core1 = core
                        src1 = io_array[io_idx][0] - 1
                        sink1 = io_array[io_idx][1] - 1
                        core2 = all[0]
                        src2 = io_array[maps[all[0]]][0] - 1
                        sink2 = io_array[maps[all[0]]][1] - 1
                        is_conflict = check_path_conflict(dir, core1, src1, sink1, core2, src2, sink2)
                        if is_conflict:
                            temp_passive.append(core)
                            start_time[core] = max(start_time[core], global_time[maps[all[0]]])
                            break

                # Step 16 continued: Allocate core if no conflicts
                if not is_conflict:
                    allocated[core] = True
                    alloc.append((core, start_time[core] + test_times[io_idx][core_in_io_indx]))
                    global_time[io_idx] = start_time[core] + test_times[io_idx][core_in_io_indx]

                    # Update start times for other cores in the same io_data
                    for c in io_data[io_idx]:
                        if not allocated[c]:
                            start_time[c] = max(start_time[c], global_time[io_idx])

                    # Resolve conflicts for cores in temp_passive
                    for conf in temp_passive:
                        start_time[conf] = max(start_time[conf], global_time[io_idx])

                    break

            # Step 17: Process remaining unallocated cores in io_data
            for idx, io in enumerate(io_data):
                temp_conflict = []
                for indx, c in enumerate(io):
                    if allocated[c] or in_passive[c]:
                        continue

                    is_conflict = False
                    for a, all in enumerate(alloc):
                        if start_time[c] < all[1]:
                            # Step 19: Check conflicts with allocated cores
                            core1 = c
                            src1 = io_array[idx][0] - 1
                            sink1 = io_array[idx][1] - 1
                            core2 = all[0]
                            src2 = io_array[maps[all[0]]][0] - 1
                            sink2 = io_array[maps[all[0]]][1] - 1
                            is_conflict = check_path_conflict(dir, core1, src1, sink1, core2, src2, sink2)
                            if is_conflict:
                                temp_conflict.append(c)
                                start_time[c] = max(start_time[c], global_time[maps[all[0]]])
                                break

                    # Step 20: Allocate core if no conflicts
                    if not is_conflict:
                        allocated[c] = True
                        alloc.append((c, start_time[c] + test_times[idx][indx]))
                        global_time[idx] = start_time[c] + test_times[idx][indx]

                        # Update start times for other cores and resolve conflicts
                        for core in io:
                            if not allocated[core]:
                                start_time[core] = max(start_time[core], global_time[idx])
                        for conf in temp_conflict:
                            start_time[conf] = max(start_time[conf], global_time[idx])
                            pass_arr.put((start_time[conf], conf))
                            in_passive[conf] = True
                        temp_conflict = []
                        break

                    for c in temp_conflict:
                        pass_arr.put((start_time[c], c))
                        in_passive[c] = True

            # Step 15 continued: Refill the passive queue
            temp = []
            while not pass_arr.empty():
                start, core = pass_arr.get()
                temp.append(core)
            for c in temp:
                pass_arr.put((start_time[c], c))

        # Step 23: Append penalty and allocation order
        penalty.append(np.max(global_time) * -1)  # Negate max global time as penalty
        orders.append([item[0] for item in alloc])  # Append allocation order

    # Step 25: Return penalties and allocation orders
    return torch.from_numpy(np.array(penalty)), orders
