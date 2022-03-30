"""
   name: cs444_csp_test.py
purpose: Load test files and then call backtracking and min-conflict methods
   date: Feb 28, 2019

   Completed by Patrick Muradaz
"""

from cs444_PA2 import *

import sys
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description='read in adj matrix')

    parser.add_argument('--inputFile', action='store',
                        dest='input_file', default="", required=True,
                        help='csv file of nodeId, neighborID')

    parser.add_argument('--colorCount', action='store',
                        dest='color_count', default="", required=True,
                        help='starting value for k (number of colors)', type=int)

    return parser.parse_args()


def mod_adj_list(adj_list, u, v):
    this_node_list = adj_list.get(u, None)
    if this_node_list is None:
        this_node_list = []
        adj_list[u] = this_node_list

    if v not in this_node_list:
        this_node_list.append(v)



"""
Read in adjacency matrix and return a dictionary. 
keys in the dictionary are the nodeIDs of the graph
values are the list of neighb
"""
def parse_adj_file(parms):
    print('processing file:', parms.input_file)

    file = open(parms.input_file, 'r')

    adj_list = {}
    first_line = True
    for line in file:
        if not first_line:
            line_items = line.split(',')
            u = line_items[0].rstrip()
            v = line_items[1].strip()
            if u != v: # no self edges
                mod_adj_list(adj_list, u, v)  # treat edge as undirected
                mod_adj_list(adj_list, v, u)
        else:
            first_line = False

    return adj_list


def main():
    parms = parse_args()

    adj_list = parse_adj_file(parms)

    # largest degree node
    max_degree = max([len(items) for items in adj_list.values()])
    edge_count = sum([len(items) for items in adj_list.values()])

    print('Graph stats: nodes:', len(adj_list.keys()), ' edges:', edge_count,
          ' maxDegree is:', max_degree)

    # color_count is the initial value to start k (the number of colors)
    # your code should vary k to find the minimum value that can successfully
    # color the input graph

    color_list = list(range(parms.color_count))

    map_coloring_csp = MapColoringCSP(color_list, adj_list)

    # set the python recursion limit to be the depth of a solution path
    # with a little padding (for other functions)

    # MIN CONFLICTS

    print("")
    print('starting min conflicts')
    start_time = time.time()
    max_attempts = 1000
    min_conflicts_solution = min_conflicts(map_coloring_csp, max_attempts)
    end_time = time.time()
    print(min_conflicts_solution)
    print('min_conflicts solution time: {0:.2f}'.format(end_time - start_time))

    # BACKTRACKING

    sys.setrecursionlimit(len(adj_list.keys()) + 100)
    print("")
    print('starting backtracking search')
    start_time = time.time()
    solution_backtrack = backtracking_search(map_coloring_csp)
    end_time = time.time()
    print(solution_backtrack)
    print('backtracking solution time: {0:.2f}'.format(end_time - start_time))

    # BACKTRACKING WITH MRV

    print("")
    print('starting backtracking w/ MRV search')
    start_time = time.time()
    solution_backtrack = backtracking_search(map_coloring_csp)
    end_time = time.time()
    print(solution_backtrack)
    print('backtracking w/ MRV solution time: {0:.2f}'.format(end_time - start_time))


if __name__ == '__main__':
    main()
