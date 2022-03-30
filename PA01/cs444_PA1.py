#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
  Search
  ------
  This python file utilizes some class originally developed
  by Kevin Molloy, Peter Norvig, and Russell Stewarts teams.
  
  This file has been completed for PA1 by Patrick Muradaz
"""


from utils import PriorityQueue, shuffled

from collections import defaultdict, deque
import math
import random
import sys
from datetime import datetime
from datetime import timedelta
import bisect
from operator import itemgetter


infinity = float('inf')

# ______________________________________________________________________________


class Problem(object):

    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError
# ______________________________________________________________________________


class Node:

    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.  Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node.  Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action,
                    problem.path_cost(self.path_cost, self.state,
                                      action, next_state))
        return next_node
    
    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)

# ______________________________________________________________________________


# ______________________________________________________________________________
# Uninformed Search algorithms


def breadth_first_tree_search(problem):
    """Search the shallowest nodes in the search tree first.
        Search through the successors of a problem to find a goal.
        The argument frontier should be an empty queue.
        Repeats infinitely in case of loops. [Figure 3.7]"""

    frontier = deque([Node(problem.initial)])  # FIFO queue
    explored_nodes = 1
    while frontier:
        explored_nodes += 1
        node = frontier.popleft()
        if problem.goal_test(node.state):
            return node, explored_nodes
        frontier.extend(node.expand(problem))
    return None, explored_nodes


def depth_first_tree_search(problem):
    """Search the deepest nodes in the search tree first.
        Search through the successors of a problem to find a goal.
        The argument frontier should be an empty queue.
        Repeats infinitely in case of loops. [Figure 3.7]"""

    frontier = [Node(problem.initial)]  # Stack

    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        frontier.extend(node.expand(problem))
    return None


def best_first_graph_search(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search."""
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    explored_nodes =  0
    while frontier:
        node = frontier.pop()
        explored_nodes += 1
        if problem.goal_test(node.state):
            return node,explored_nodes
        explored.add(tuple(node.state))
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
    return None, explored_nodes


def uniform_cost_search(problem):
    """[Figure 3.14]"""
    return best_first_graph_search(problem, lambda node: node.path_cost)


def depth_limited_search(problem, limit=50):
    """[Figure 3.17]"""
    def recursive_dls(node, problem, limit):
        if problem.goal_test(node.state):
            return node
        elif limit == 0:
            return 'cutoff'
        else:
            cutoff_occurred = False
            for child in node.expand(problem):
                result = recursive_dls(child, problem, limit - 1)
                if result == 'cutoff':
                    cutoff_occurred = True
                elif result is not None:
                    return result
            return 'cutoff' if cutoff_occurred else None

    # Body of depth_limited_search:
    return recursive_dls(Node(problem.initial), problem, limit)


def iterative_deepening_search(problem):
    """[Figure 3.18]"""
    for depth in range(sys.maxsize):
        result = depth_limited_search(problem, depth)
        if result != 'cutoff':
            return result


# ______________________________________________________________________________
# Informed (Heuristic) Search


greedy_best_first_graph_search = best_first_graph_search
# Greedy best-first search is accomplished by specifying f(n) = h(n).


def astar_search(problem, h=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""

    return best_first_graph_search(problem, lambda n: n.path_cost + h(n))



class EightPuzzle(Problem):

    """ The problem of sliding tiles numbered from 1 to 8 on a 3x3 board,
    where one of the squares is a blank. A state is represented as a tuple with 9
    elements.  Element 0 contans the number in location 0 in the 3x3 grid (see picture below)

    |-----|-----|-----|
    |  0  |  1  |  2  |
    |  3  |  4  |  5  |
    |  6  |  7  |  8  |
    --------------------


    0 represents the empty square
    So, the board below is encoded as (2,4,3,1,5,6,7,8,0)

    |-----|-----|-----|
    |  2  |  4  |  3  |
    |  1  |  5  |  6  |
    |  7  |  8  |     |
    --------------------
    """
 
    def __init__(self, initial, goal=(1, 2, 3, 4, 5, 6, 7, 8, 0)):
        """ Define goal state and initialize a problem """

        self.goal = goal
        Problem.__init__(self, initial, goal)
    
    def find_blank_square(self, state):
        """Return the index of the blank square in a given state"""

        return state.index(0)
    
    def actions(self, state):
        """ Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment
        You may need to edit this list, for example, since if you are in the top
        left corner of the board, you can not move left or up.  """
        
        possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']

        blank = self.find_blank_square(state)
        
        if (blank == 0 or blank == 3 or blank == 6):
            possible_actions.remove('LEFT')
        if (blank == 2 or blank == 5 or blank == 8):
            possible_actions.remove('RIGHT')
        if (blank == 0 or blank == 1 or blank == 2):
            possible_actions.remove('UP')
        if (blank == 6 or blank == 7 or blank == 8):
            possible_actions.remove('DOWN')

        return possible_actions

    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """
        
        blank = self.find_blank_square(state)
        new_state = list(state)
        
        if (action == 'UP'):
            swap_item_location = blank - 3
            swap_item = state[swap_item_location]
            new_state[swap_item_location] = 0
            new_state[blank] = swap_item
        if (action == 'DOWN'):
            swap_item_location = blank + 3
            swap_item = state[swap_item_location]
            new_state[swap_item_location] = 0
            new_state[blank] = swap_item
        if (action == 'LEFT'):
            swap_item_location = blank - 1
            swap_item = state[swap_item_location]
            new_state[swap_item_location] = 0
            new_state[blank] = swap_item
        if (action == 'RIGHT'):
            swap_item_location = blank + 1
            swap_item = state[swap_item_location]
            new_state[swap_item_location] = 0
            new_state[blank] = swap_item

        return tuple(new_state)

    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """

        return state == self.goal

    def check_solvability(self, state):
        """ Checks if the given state is solvable """

        inversion = 0
        for i in range(len(state)):
            for j in range(i, len(state)):
                if state[i] > state[j] != 0:
                    inversion += 1
        
        return inversion % 2 == 0
    
    def h(self, node):
        """ Return the heuristic value for a given state.
        You should write a heuristic function that is as follows:
            h(n) = number of misplaced tiles
        """
        
        state = node.state
        h_value = 0
        
        for tile in state:
            if (tile != 0):
                correct_placement = tile - 1
                if (state.index(tile) != correct_placement):
                    h_value = h_value + 1
        
        return h_value;



    def h2(self, node):
        """ Return the heuristic value for a given state.
        You should write a heuristic function that is as follows:
        h2(n) = manhatten distance to move tiles into place
        """
        
        state = node.state
        h_value = 0
        
        for tile in state:
            if (tile != 0):
                current_index = state.index(tile)
                correct_index = tile - 1
                current_grid_place = (current_index%3, current_index/3)
                correct_grid_place = (correct_index%3, correct_index/3)
                
                h_value = h_value + abs(current_grid_place[0] - correct_grid_place[0]) + abs(current_grid_place[1] - correct_grid_place[1])
        
        return h_value;
    
    
    

# ______________________________________________________________________________
#main

""" Test the program by randomly generating 10 problems 
    and solving them with A* h, A* h2, and IDS (when appropriate).
    Print the output to the screen so the user can see the initial
    state, solution sequence, and run time.
    """

puzzles = []
i = 0

while i < 10:
    state = shuffled((1, 2, 3, 4, 5, 6, 7, 8, 0))
    puzzle = EightPuzzle(state)
    
    if (puzzle.check_solvability(state)):
        puzzles.append(puzzle)
        i = i + 1
        
for problem in puzzles: 
    print("A*, h1:")
    print(problem.initial)
    start = datetime.now()
    node, count = astar_search(problem, problem.h)
    stop = datetime.now()
    time = stop - start
    ms = (time.days * 24 * 60 * 60 + time.seconds) * 1000 + time.microseconds / 1000.0
    print(node.solution())
    print(node.state)
    print(ms)
    ms = 0
    count = 0
    
    print()
    
    print("A*, h2:")
    print(problem.initial)
    start = datetime.now()
    node, count = astar_search(problem, problem.h2)
    stop = datetime.now()
    time = stop - start
    ms = (time.days * 24 * 60 * 60 + time.seconds) * 1000 + time.microseconds / 1000.0
    print(node.solution())
    print(node.state)
    print(ms)
    
    print()
    
    depth = len(node.solution())
    
    if (depth < 20):
        print("IDS:")
        print(problem.initial)
        start = datetime.now()
        node = iterative_deepening_search(problem)
        stop = datetime.now()
        time = stop - start
        ms = (time.days * 24 * 60 * 60 + time.seconds) * 1000 + time.microseconds / 1000.0
        print(node.solution())
        print(node.state)
        print(ms)
        print()
        
    ms = 0
    count = 0
    


# In[ ]:





# In[ ]:





# In[ ]:




