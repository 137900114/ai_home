from abc import ABCMeta, abstractmethod

import util


class SearchProblem(metaclass=ABCMeta):
    @abstractmethod
    def get_start_state(self):
        pass

    @abstractmethod
    def is_goal_state(self, state):
        pass

    @abstractmethod
    def get_successor(self, state):
        # return (next_state, action, cost)
        pass

    @abstractmethod
    def get_costs(self, actions):
        pass

    @abstractmethod
    def get_goal_state(self):
        pass


class Node:
    def __init__(self, state, path=[], priority=0):
        self.state = state
        self.path = path
        self.priority = priority

    def __le__(self, other):
        return self.priority <= other.priority

    def __lt__(self, other):
        return self.priority < other.priority


def search(problem, fringe, calc_heuristic=None, heuristic=None):
    """
    This is an simple abstracted graph search algorithm. You could
    using different combination of fringe storage, calc_heuristic, heuristic
    to implement different search algorithm.

    For example:
    LIFO Queue(Stack), None, None -> Depth First Search
    FIFO Queue, None, None -> Breadth First Search
    PriorityQueue, cost compute function, None -> Uniform Cost Search

    In order to avoid infinite graph/tree problem we setup a list (visited) to
    avoid expanding the same node.

    hint: please check the node first before expanding:

    if node.state not in visited:
        visited.append(node.state)
    else:
        continue

    hint: you could get the successor by problem.get_successor method.

    hint: for fringe you may want to use
        fringe.pop  get a node from the fringe
        fringe.push   put a node into the fringe
        fringe.empty  check whether a fringe is empty or not. If the fringe is empty this function return True
        problem.is_goal_state check whether a state is the goal state
        problem.get_successor get all successor from current state
            return value: [(next_state, action, cost)]
    """
    start_state = problem.get_start_state()
    if isinstance(fringe, util.Stack) or isinstance(fringe, util.Queue):
        fringe.push(Node(start_state))
    else:
        fringe.push(Node(start_state), 0)
    visited = []
    step = 0
    while not fringe.empty():
        #"*** YOUR CODE HERE ***"

        curr_state_node  = fringe.pop()
        successors = problem.get_successor(curr_state_node.state)

        for successor in successors:
            (next_state,action,cost) = successor
            
            if next_state in visited:
                continue
            visited.append(next_state)
            if next_state == problem.get_goal_state():
                path = curr_state_node.path[:]
                path.append(action)
                return (path,len(path))
            if isinstance(fringe,util.Stack) or isinstance(fringe,util.Queue):
                path = curr_state_node.path[:]
                path.append(action)
                fringe.push(Node(next_state,path))
            else:
                path = curr_state_node.path[:]
                path.append(action)

                cost = calc_heuristic(problem,successor,curr_state_node,heuristic)
                fringe.push(Node(next_state,path),cost)



        #"*** END YOUR CODE HERE ***"
    return [] # no path is found


def a_start_heuristic(problem, current_state):
    h = 0

    "*** YOUR CODE HERE ***"
    for i in range(0,3):
        for j in range(0,3):
            num = i * 3 + j
            h += 1 if num != current_state.cells[i][j] else 0

    "*** END YOUR CODE HERE ***"
    return h


def a_start_cost(problem, successor, node, heuristic):
    cost = 0
    "*** YOUR CODE HERE ***"
    path = node.path    
    cost = problem.get_costs(path)
    (next_state,trash2,next_cost) = successor

    cost += heuristic(problem,next_state) + next_cost

    "*** END YOUR CODE HERE ***"
    return cost


def a_start_search(problem):
    path = []
    step = 0
    "*** YOUR CODE HERE ***"
    path,step = search(problem,util.PriorityQueue(),\
                       a_start_cost,a_start_heuristic)

    "*** END YOUR CODE HERE ***"
    return path, step


def ucs_compute_node_cost(problem, successor, node, heuristic):
    """
    Define the method to compute cost within unit cost search
    hint: successor = (next_state, action, cost).
    however the cost for current node should be accumulative
    problem and heuristic should not be used by this function
    """
    cost = 0
    "*** YOUR CODE HERE ***"
    # TODO ucs_compute_node_cost
    path = node.path    
    cost = problem.get_costs(path)
    (trash1,trash2,next_cost) = successor

    "*** END YOUR CODE HERE ***"
    return cost + next_cost


def uniform_cost_search(problem):
    """
    Search the solution with minimum cost.
    """
    return search(problem, util.PriorityQueue(), ucs_compute_node_cost)


def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    hint: using util.Queue as the fringe
    """
    path = []
    step = 0
    "*** YOUR CODE HERE ***"
    
    path,step = search(problem,util.Queue())
    "*** END YOUR CODE HERE ***"
    return path, step


def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.get_start_state()
    print "Is the start a goal?", problem.is_goal(problem.get_start_state())
    print "Start's successors:", problem.get_successors(problem.get_start_state())

    hint: using util.Stack as the fringe
    """
    path = []
    step = 0
    "*** YOUR CODE HERE ***"
    # TODO a_start_search
    path,step = search(problem,util.Stack())
    "*** END YOUR CODE HERE ***"
    return path, step
