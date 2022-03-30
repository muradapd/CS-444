import random
import copy
import numpy
from collections import defaultdict

"""File completed by Patrick Muradaz"""


class Problem:
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
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError


# ______________________________________________________________________________


class CSP(Problem):
    """This class describes finite-domain Constraint Satisfaction Problems.
    A CSP is specified by the following inputs:
        variables   A list of variables; each is atomic (e.g. int or string).
        domains     A dict of {var:[possible_value, ...]} entries.
        neighbors   A dict of {var:[var,...]} that for each variable lists
                    the other variables that participate in constraints.
        constraints A function f(A, a, B, b) that returns true if neighbors
                    A, B satisfy the constraint when they have values A=a, B=b


    This class supports data structures and methods that help you
    solve CSPs by calling a search function on the CSP. Methods and slots are
    as follows, where the argument 'a' represents an assignment, which is a
    dict of {var:val} entries:
        assign(var, val, a)     Assign a[var] = val; do other bookkeeping
        unassign(var, a)        Do del a[var], plus other bookkeeping
        nconflicts(var, val, a) Return the number of other variables that
                                conflict with var=val
        curr_domains[var]       Slot: remaining consistent values for var
                                Used by constraint propagation routines.
    """

    # arguments.
    #         variables   A list of variables; each is atomic (e.g. int or string).
    #                     For map coloring, you can think of these as IDs of the nodes
    #                     in the graph (thus, the number of numbers should equal the
    #                     number of items in this list).
    #                     e.g., ['a','b','c']
    #         domains     A dict of {var:[possible_vlaue, ...]} entries.  For map coloring,
    #                     this is the colors that each node can be assigned.
    #                     e.g., {'a':['1','2','3'],'b':['1','2','3']}
    #         neighbors   A dict of {var:[var,...]} that for each variable lists
    #                     the other variables that participate in constraints.
    #                     For the map coloring problem, this is the nodes that
    #                     share an edge (an adjancency list).
    #         constraint  Some function that called to see if neighbors violate
    #                     the constraint.  For map coloring, this would be two
    #                     neighbors that share the same color.
    #
    def __init__(self, variables, domains, neighbors, constraints):
        """Construct a CSP problem. If variables is empty, it becomes domains.keys()."""
        super().__init__(())
        variables = variables or list(domains.keys())
        self.variables = variables
        self.domains = domains
        self.neighbors = neighbors
        self.constraints = constraints
        self.curr_domains = None
        self.nassigns = 0

    def assign(self, var, val, assignment):
        """Add {var: val} to assignment; Discard the old value if any."""
        assignment[var] = val
        self.nassigns += 1

    def unassign(self, var, assignment):
        """Remove {var: val} from assignment.
        DO NOT call this if you are changing a variable to a new value;
        just call assign for that."""
        if var in assignment:
            del assignment[var]

    """ I WROTE THIS """
    def nconflicts(self, var, val, assignment):
        """Return the number of conflicts var=val has with other variables in the assignment."""
        conflicts = 0
        neighbors = dict(self.neighbors).get(var)
        for neighbor in neighbors:
            if assignment.get(neighbor) == val:
                conflicts += 1

        return conflicts

    def goal_test(self, state):
        """The goal is to assign all variables, with all constraints satisfied."""
        assignment = dict(state)
        return (len(assignment) == len(self.variables)
                and all(self.nconflicts(variables, assignment[variables], assignment) == 0
                        for variables in self.variables))

    # This is for min_conflicts search

    """ I WROTE THIS """
    def conflicted_vars(self, current):
        """Return a list of variables in current assignment that are in conflict"""
        conflicted_vars = list()
        for var in self.variables:
            if self.nconflicts(var, dict(current).get(var), current) > 0:
                conflicted_vars.append(var)
        return conflicted_vars

    # provided to student
    def choices(self, var):
        """Return all values for var that aren't currently ruled out."""
        return (self.curr_domains or self.domains)[var]

# _________________________________________________________________________________


""" I WROTE THIS """


def min_conflicts(csp, max_steps=100000):
    """Randomly select conflicted variables and reassign them low conflict values
    until we have a solution. Return the solution if found and indicate it's a
    solution. Else, return the time out state and indicate a time out."""
    current = initial_assignment(csp)
    # print("initial conflicts: ", initial_conflicts(csp, current)) USED FOR TESTING VARIANCE
    for i in range(0, max_steps):
        if csp.goal_test(current):
            print("goal found")
            return current
        var = random.choice(csp.conflicted_vars(current))
        val = min_conflicts_reassign(csp, var, current)
        csp.assign(var, val, current)
    print("timed out")
    return current


def initial_assignment(csp):
    """Provide random initial assignments of values to the variables"""
    assignment = dict()
    for var in csp.variables:
        colors = csp.domains[0]
        color = random.choice(colors)
        csp.assign(var, color, assignment)
    return assignment


def min_conflicts_reassign(csp, var, current):
    """Reassign a given variable to a value that causes minimum conflicts"""
    current_conflicts = float("inf")
    choices = csp.choices(var)
    val = None
    for choice in choices:
        choice_conflicts = csp.nconflicts(var, choice, current)
        if choice_conflicts < current_conflicts:
            current_conflicts = choice_conflicts
            val = choice
    return val


def initial_conflicts(csp, current):
    """Testing method used to determine the number of conflicts with the
    initial assignment of the variables."""
    conflicts = 0
    for var in csp.variables:
        conflicts += csp.nconflicts(var, current.get(var), current)
    return conflicts

# _________________________________________________________________________________
# CSP Backtracking Search

# Variable ordering


def first(iterable, default=None):
    """Return the first element of an iterable; or default."""
    return next(iter(iterable), default)


def first_unassigned_variable(assignment, csp):
    """The default variable order."""
    return first([var for var in csp.variables if var not in assignment])


def unordered_domain_values(var, assignment, csp):
    """The default value order."""
    return csp.choices(var)

# _________________________________________________________________________________


"""I WROTE THIS"""


def individual_choices(csp):
    """Maps variables to their available choices"""
    indiv_choices = dict()
    for var in csp.variables:
        choices = copy.deepcopy(csp.choices(var))
        indiv_choices[var] = choices
    return indiv_choices


def backtracking_search(csp):
    """Method to start the backtracking search recursion"""
    assignment = dict()
    neighbor_choices = individual_choices(csp)
    return backtracking_search_recur(assignment, csp, neighbor_choices)


def backtracking_search_recur(current, csp, neighbor_choices):
    """Method for recursive backtracking search using the highest degree variable heuristic"""
    if len(current) == len(csp.variables):
        print("goal found")
        return current
    var = highest_deg_variable(current, csp)
    for val in unordered_domain_values(var, current, csp):
        if csp.nconflicts(var, val, current) == 0:
            csp.assign(var, val, current)
            inference = infer(csp, var, val, neighbor_choices)
            if inference:
                result = backtracking_search_recur(current, csp, neighbor_choices)
                if len(current) == len(csp.variables):
                    return result
        csp.unassign(var, current)
        uninfer(csp, val, neighbor_choices)
    print("failure")
    exit()
    return current


def backtracking_search_mrv(csp):
    """Method to start the backtracking search with MRV recursion"""
    assignment = dict()
    neighbor_choices = individual_choices(csp)
    return backtracking_search_mrv_recur(assignment, csp, neighbor_choices)


def backtracking_search_mrv_recur(current, csp, neighbor_choices):
    """Method for recursive backtracking search using the minimum remaining values heuristic"""
    if len(current) == len(csp.variables):
        print("goal found")
        return current
    var = min_remaining_value(current, csp, neighbor_choices)
    for val in unordered_domain_values(var, current, csp):
        if csp.nconflicts(var, val, current) == 0:
            csp.assign(var, val, current)
            inference = infer(csp, var, val, neighbor_choices)
            if inference:
                result = backtracking_search_recur(current, csp, neighbor_choices)
                if len(current) == len(csp.variables):
                    return result
        csp.unassign(var, current)
        uninfer(csp, val, neighbor_choices)
    print("failure")
    exit()
    return current


def infer(csp, var, val, neighbor_choices):
    """Method for adding to the inference"""
    neighbors = csp.neighbors
    for neighbor in neighbors:
        choices = neighbor_choices[neighbor]
        if len(choices) == 0:
            return False
        if var in neighbor_choices[neighbor]:
            neighbor_choices[neighbor].remove(val)
    return True


def uninfer(csp, val, neighbor_choices):
    """Method for removing from the inference"""
    neighbors = csp.neighbors
    for neighbor in neighbors:
        neighbor_choices[neighbor].append(val)


def highest_deg_variable(assignment, csp):
    """Method to implement the highest degree variable heuristic"""
    max_deg = -1
    max_deg_var = None
    for var in csp.variables:
        if var not in assignment:
            var_deg = len(dict(csp.neighbors).get(var))
            if var_deg > max_deg:
                max_deg = var_deg
                max_deg_var = var
    return max_deg_var


def min_remaining_value(assignment, csp, choices):
    """Method to implement the minimum remaining values heuristic"""
    min_values = float("inf")
    mrv = None
    for var in csp.variables:
        if var not in assignment:
            var_choices = len(choices[var])
            if var_choices < min_values:
                min_values = var_choices
                mrv = var
    return mrv



# _________________________________________________________________________________


class UniversalDict:
    """A universal dict maps any key to the same value. We use it here
    as the domains dict for CSPs in which all variables have the same domain.
    >>> d = UniversalDict(42)
    >>> d['life']
    42
    """

    def __init__(self, value): self.value = value

    def __getitem__(self, key): return self.value

    def __repr__(self): return '{{Any: {0!r}}}'.format(self.value)


# ______________________________________________________________________________
# argmin and argmax.  Might be handle, but you do not have to use these functions

identity = lambda x: x


def argmin_random_tie(seq, key=identity):
    """Return a minimum element of seq; break ties at random."""
    return min(shuffled(seq), key=key)


def argmax_random_tie(seq, key=identity):
    """Return an element with highest fn(seq[i]) score; break ties at random."""
    return max(shuffled(seq), key=key)


def shuffled(iterable):
    """Randomly shuffle a copy of iterable."""
    items = list(iterable)
    random.shuffle(items)
    return items


def different_values_constraint(A, a, B, b):
    """A constraint saying two neighboring variables must differ in value."""
    return a != b


def MapColoringCSP(colors, neighbors):
    """Make a CSP for the problem of coloring a map with different colors
    for any two adjacent regions. Arguments are a list of colors, and a
    dict of {region: [neighbor,...]} entries. This dict may also be
    specified as a string of the form defined by parse_neighbors."""
    if isinstance(neighbors, str):
        neighbors = parse_neighbors(neighbors)
    # CSP constructor takes:
    # -- variables (all the keys/nodes in the dict)
    # -- domains (the colors), neighbors, constraints):
    return CSP(list(neighbors.keys()), UniversalDict(colors), neighbors, different_values_constraint)


def parse_neighbors(neighbors):
    """Convert a string of the form 'X: Y Z; Y: Z' into a dict mapping
    regions to neighbors. The syntax is a region name followed by a ':'
    followed by zero or more region names, followed by ';', repeated for
    each region name. If you say 'X: Y' you don't need 'Y: X'.
    >>> parse_neighbors('X: Y Z; Y: Z') == {'Y': ['X', 'Z'], 'X': ['Y', 'Z'], 'Z': ['X', 'Y']}
    True
    """
    dic = defaultdict(list)
    specs = [spec.split(':') for spec in neighbors.split(';')]
    for (A, Aneighbors) in specs:
        A = A.strip()
        for B in Aneighbors.split():
            dic[A].append(B)
            dic[B].append(A)
    return dic
