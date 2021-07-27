# structures.py
# some data structure practice
from typing import *
from enum import IntEnum


#basic node classes are more like structs than classes
class Node: # basest node for linking
    def __init__(self, data=None, next:'Node'=None):
        self.data = data
        self.next = next


class TwoWayNode(Node): # for double links
    # inherits from Node spiritually but doesn't super
    def __init__(self, data=None, 
            next:'TwoWayNode'=None, prev:'TwoWayNode'=None):
        self.data = data
        self.next = next
        self.prev = prev


class BiDirect(IntEnum):
    LEFT = 0
    RIGHT = 1


class BTNode:
    def __init__(self, key=None, data=None, 
            left:'BTNode'=None, right:'BTNode'=None):
        self.key = key
        self.data = data
        self._children:'List[Optional[BTNode]]' = [None, None]
        self._children[BiDirect.LEFT] = left
        self._children[BiDirect.RIGHT] = right

    def get_child(self, d:BiDirect):
        return self._children[d]

    def set_child(self, d:BiDirect, node:'Optional[BTNode]'):
        '''set to None to remove'''
        self._children[d] = node


# helper methods make special BT nodes less struct-y
class BTPNode(BTNode):
    # BTNode with parent pointer; unused at the moment
    # (BTHNode and BTCNode could inherit but current implementations
    #  of balanced trees record traversals as needed instead)
    def __init__(self, key=None, data=None,
            left:'BTPNode'=None, right:'BTPNode'=None):
        super().__init__(key, data, left, right)
        self.parent = None
        # parents are always set when connected from parent side

    # essentially setters, but not named so because of logic
    def connect_child(self, node:'BTPNode', d:BiDirect):
        '''connect child (setting child's parent as well)'''
        self.set_child(d, node)
        node.parent = self

    def pop_child(self, d:BiDirect):
        '''disconnect and return child'''
        node:'BTPNode' = self.get_child(d)
        if node is None:
            return Node
        self.set_child(d, None)
        node.parent = None
        return node


class BTHNode(BTNode):
    # BTNode with height and some convenience methods
    def __init__(self, key=None, data=None, 
            left:'BTHNode'=None, right:'BTHNode'=None):
        super().__init__(key, data, left, right)
        self.needs_height = False  # toggle as needed
        self._height = 1 # starts with _ because not used outside class
        if left is not None or right is not None:
            self.needs_height = True
            self.determine_height()

    @staticmethod
    def _determine_height(node:'BTHNode') -> int:
        '''
        sets and returns height for node recursively
        (recursively setting child heights as needed)
        '''
        if node is None:
            return 0
        if not node.needs_height:
            return node._height
        node._height = 1 + max(node.child_heights())
        node.needs_height = False
        return node._height

    def determine_height(self) -> int:
        '''
        sets (only if flagged necessary) and returns height of instance
        (use this to find height instead of accessing _height
        because this method respects the state of needs_height)
        '''
        return self._determine_height(self)

    def child_heights(self) -> Tuple[int, int]:
        '''
        helper method for balancing trees;
        returns (left-height, right-height) tuple for node,
        with None-child having 0 height
        (tuple indices match BiDirect and _children indices for consistency)
        '''
        return (self._determine_height(self.get_child(BiDirect.LEFT)), 
                self._determine_height(self.get_child(BiDirect.RIGHT)))


class BTCNode(BTNode):
    # BTNode with color (not coin)
    # (color is boolean to indicate presence, not qualitative description)
    # node and tree based on Sedgewick's discussion at
    # https://www.cs.princeton.edu/~rs/talks/LLRB/LLRB.pdf
    def __init__(self, key=None, data=None, color:bool=True, 
            left:'BTCNode'=None, right:'BTCNode'=None):
        super().__init__(key, data, left, right)
        # color defaults to True because RBTree insertions start colored
        # to maintain black-depth constraint
        self.color = color

    @staticmethod
    def is_colored(node:'Optional[BTCNode]'):
        '''returns color of edge leading to node (node can be None)'''
        return False if node is None else node.color

    def flip_colors(self):
        '''
        used in RB trees when both outgoing edges are colored
        so a child being None shouldn't occur
        (method still checks for None but not necessary)

        (node and children all having same colored edges also shouldn't occur
        if tree starts off in proper RB-shape)
        '''
        self.color = not self.color
        for child in self._children:
            if child is not None:
                child = cast(BTCNode, child)
                child.color = not child.color


class LinkedList: # single link
    def __init__(self):
        self.head = self.tail = None

    # makes HashMap easier but breaks abstraction
    def _find_node(self, test_fnc:Callable[[], bool]) -> Node:
        '''returns first node (node, not data) satisfying test_fnc'''
        cur = self.head
        while cur is not None:
            if test_fnc(cur.data):
                break
            cur = cur.next
        return cur

    def insert_head(self, data):
        new_node = Node(data=data)
        if self.head is None:
            self.head = self.tail = new_node
            return new_node
        new_node.next = self.head
        self.head = new_node

    def insert_tail(self, data):
        if self.head is None:
            return self.insert_head(data)
        self.tail.next = Node(data)
        self.tail = self.tail.next

    # no pop-tail because inefficient (should use doubly linked for that)
    # pops return values (data) not nodes to keep up the abstraction
    def pop_head(self):
        if self.head is None:
            return None
        poppy = self.head.data
        self.head = self.head.next
        if self.head is None:
            self.tail = None
        # poppy's node should be deallocated here but not done in python
        return poppy

    def to_list(self) -> list:
        arr = []
        cur = self.head
        while cur is not None:
            arr.append(cur.data)
            cur = cur.next
        return cur

    def find(self, test_fnc:Callable[[], bool]):
        '''returns data for first node satisfying test_fnc'''
        cur = self._find_node(test_fnc)
        return None if cur is None else cur.data

    # not sure if this should be supported operation for linked lists,
    # but it will make hashmaps easier
    def remove(self, test_fnc:Callable[[], bool]):
        '''removes and returns data for first node satisfying test_fnc'''
        if self.head is None:
            return None
        cur = self.head
        found = None
        if test_fnc(cur.data):
            found = cur.data
            self.head = cur.next
            # deallocate cur
            return found
        while cur.next is not None:
            if test_fnc(cur.next.data):
                found = cur.next.data
                new_next = cur.next.next
                # deallocate cur.next
                cur.next = new_next
                if cur.next is None:
                    self.tail = cur
                break
            cur = cur.next
        return found

    def __str__(self) -> str:
        cur = self.head
        node_datums = []
        while cur is not None:
            node_datums.append(str(cur.data))
            cur = cur.next
        node_datums.append('None')
        return ' -> '.join(node_datums)


class DoubleLinkedList(LinkedList):
    # currently unused, hashmaps use single links
    def __init__(self):
        super().__init__()

    def insert_head(self, data):
        new_node = TwoWayNode(data=data)
        if self.head is None:
            self.head = self.tail = new_node
            return new_node
        self.head.prev = new_node
        new_node.next = self.head
        self.head = new_node

    def insert_tail(self, data):
        if self.head is None:
            return self.insert_head(data)
        new_node = TwoWayNode(data=data)
        self.tail.next = new_node
        new_node.prev = self.tail
        self.tail = new_node

    def pop_head(self):
        old_head_data = super().pop_head()
        if self.head is not None:
            self.head.prev = None
        return old_head_data

    def pop_tail(self):
        if self.head == self.tail: # simplifying logic a bit
            return self.pop_head()
        poppy = self.tail.data
        self.tail = self.tail.prev
        self.tail.next = None
        # poppy's node should be deallocated
        return poppy

    def remove(self, test_fnc:Callable[[], bool]):
        '''removes and returns data for first node satisfying test_fnc'''
        if self.head is None:
            return None
        cur = self.head
        found = None
        if test_fnc(cur.data):
            found = cur.data
            self.head = cur.next
            # deallocate cur
            return found
        while cur.next is not None:
            if test_fnc(cur.next.data):
                found = cur.next.data
                new_next = cur.next.next
                # deallocate cur.next
                cur.next = new_next
                if cur.next is None:
                    self.tail = cur
                else:
                    cur.next.prev = cur
                break
            cur = cur.next
        return found


class HashMap: # also known as worse dict
    # k->v pairs will be tuples of (k, v) and accessed by index accordingly
    # (avoiding namedtuple overhead, though enums may have greater overhead)
    # key needs to be hashable with default Python hash
    def __init__(self, initial_size:int = 10, 
            hashfnc:Callable[[], int] = hash, resize_scalar=2):
        self.hashfnc = hashfnc
        self.array:List[Optional[LinkedList]] = initial_size * [None]
        self._count = 0
        self.resize_scalar = resize_scalar

    def count(self):
        return self._count

    # for internal use during resize
    def _add(self, key, value, array:List[Optional[LinkedList]]): 
        idx = self.hashfnc(key) % len(array)
        if array[idx] is None:
            new_link = LinkedList()
            new_link.insert_head((key, value))
            array[idx] = new_link
        else:
            # head or tail is fine, but acting stacky
            array[idx].insert_head((key, value))

    def _get_node(self, key):
        idx = self.hashfnc(key) % len(self.array)
        if self.array[idx] is not None:
            return self.array[idx]._find_node(lambda node: node[0] == key)
        return None

    def _size_up(self):
        new_array = [None] * (len(self.array) * 2)
        for idx in range(len(self.array)):
            if self.array[idx] is not None:
                cur = self.array[idx].head
                while cur is not None:
                    self._add(cur.data[0], cur.data[1], new_array)
                    cur = cur.next
        self.array = new_array
    # not sure if it makes sense to make a size_down

    def set(self, key, value):
        # check if node exists
        found = self._get_node(key)
        if found:
            # replace if exists
            found.data = (key, value)
        else:
            # resize before adding because no need to double add new thing
            if self._count + 1 > len(self.array) * self.resize_scalar:
                self._size_up()
            self._add(key, value, self.array)
            self._count += 1

    def get(self, key):
        '''returns None if not found rather than raising exception'''
        found = self._get_node(key)
        return None if found is None else found.data[1]

    def remove(self, key):
        '''pops value if found, returns None if not found'''
        idx = self.hashfnc(key) % len(self.array)
        if self.array[idx] is not None:
            found_data = self.array[idx].remove(lambda data: data[0] == key)
            if found_data is not None:
                self._count -= 1
                if self.array[idx].head is None:
                    # deallocate unused linkedlist
                    self.array[idx] = None
                return found_data[1]
        return None

    def array_strings(self):
        '''
        returns array of strings instead of single string,
        but __str__ works correctly with single string
        '''
        return [str(element) for element in self.array]

    def __str__(self):
        return f'[{"; ".join(self.array_strings())}]'

# A map with balanced BST might perform better than linked-list on average,
# but trees and nodes take up more space than linked list;
# (code could be mostly reused but the string depiction wouldn't work as is)

# classes below are moving into tree territory; string depiction methods are
# no longer included because trees don't fit as well into a linear format
# (heap array could be shown as-is, but that doesn't show tree structure well)

class MinHeap:
    # not as straightforward as previous structures;
    # need a complete tree, so use array as storage
    # with children of node n at node (2n + 1, 2n + 2)
    # values stored with order-stamp as (value, stamp) in tuple,
    # need to index properly when returning values
    def __init__(self, initial_height=2):
        self.height = initial_height
        if self.height < 1:
            raise ValueError("Initial height can't be below 1")
        self.array = [None] * (2 ** self.height - 1)
        self.first_empty_idx = self.next_stamp = 0
    
    @staticmethod
    def _left_child_idx(idx:int) -> int:
        return 2 * idx + 1

    @staticmethod
    def _right_child_idx(idx:int) -> int:
        return 2 * idx + 2

    @staticmethod
    def _parent_idx(idx:int) -> int:
        if idx < 1:
            return None  # don't be trying to parent the root
        return (idx - 1) // 2

    def _size_up(self):
        self.array += [None] * (2 ** self.height)
        self.height += 1
    # not sure if it makes sense to make a size_down

    def _heap_up(self, idx):
        parent_idx = self._parent_idx(idx)
        while parent_idx is not None:
            if self.array[parent_idx] <= self.array[idx]:
                break # stop heaping up if bigger than parent
            temp = self.array[idx] # using temp to stay under 80char width
            self.array[idx] = self.array[parent_idx]
            self.array[parent_idx] = temp
            idx = parent_idx
            parent_idx = self._parent_idx(idx)
            
    def _heap_down(self, idx):
        # probably shouldn't be called on any idx besides 0,
        # but the logic would work even on other idx
        left_idx = self._left_child_idx(idx)
        right_idx = self._right_child_idx(idx)
        small_idx = idx  # to make loop logic easier to follow
        while left_idx < self.first_empty_idx:
            if self.array[left_idx] < self.array[small_idx]:
                small_idx = left_idx
            if (right_idx < self.first_empty_idx and 
                    self.array[right_idx] <= self.array[small_idx]):
                small_idx = right_idx
            if small_idx == idx:
                break # didn't need to heap down, so we're done
            temp = self.array[idx]
            self.array[idx] = self.array[small_idx]
            self.array[small_idx] = temp
            idx = small_idx
            left_idx = self._left_child_idx(idx)
            right_idx = self._right_child_idx(idx)

    def push(self, value):
        # for priority queues, just use tuple (key, actual_data);
        # internally, using (value, stamp) for tiebreaking
        if self.first_empty_idx >= len(self.array):
            self._size_up()
        self.array[self.first_empty_idx] = (value, self.next_stamp)
        self.next_stamp += 1
        self.first_empty_idx += 1
        self._heap_up(self.first_empty_idx - 1)

    def pop(self):
        if self.first_empty_idx == 0:
            return None
        smallest = self.array[0]
        self.array[0] = self.array[self.first_empty_idx - 1]
        self.array[self.first_empty_idx - 1] = None
        self.first_empty_idx -= 1
        self._heap_down(0)
        return smallest[0]

    def peek(self):
        return self.array[0]

    def _swap_top_and_heap_down(self, value):
        smallest = self.array[0]
        self.array[0] = (value, self.next_stamp)
        self.next_stamp += 1
        self._heap_down(0)
        return smallest[0]
    
    def push_pop(self, value):
        '''push then pop'''
        # because it could be faster than push() -> pop()
        if (self.first_empty_idx == 0 or
                (value, self.next_stamp) < self.array[0]):
            return value  # new value is smaller, nothing to push
        return self._swap_top_and_heap_down(value)

    def pop_push(self, value):
        '''pop then push'''
        # again, because it could be faster
        if self.first_empty_idx == 0:
            self.push(value)  # normal push because nothing to pop
            return None
        return self._swap_top_and_heap_down(value)


# abstract class with simple methods to avoid duplicating for different trees
class BinaryTree:
    def __init__(self):
        self.root = None
        self._count = 0

    def count(self):
        return self._count

    # trees need add/remove to be useful but implementation depends on tree
    def add(self, key, data):
        '''base BinaryTree add() only raises NotImplementedError'''
        raise NotImplementedError

    def remove(self, key, data):
        '''base BinaryTree remove() only raises NotImplementedError'''
        raise NotImplementedError

    @staticmethod
    def _in_order(op_fnc, node:BTNode):
        if node is not None:
            BinaryTree._in_order(op_fnc, node.get_child(BiDirect.LEFT))
            op_fnc(node)
            BinaryTree._in_order(op_fnc, node.get_child(BiDirect.RIGHT))

    @staticmethod
    def _pre_order(op_fnc, node:BTNode):
        if node is not None:
            op_fnc(node)
            BinaryTree._pre_order(op_fnc, node.get_child(BiDirect.LEFT))
            BinaryTree._pre_order(op_fnc, node.get_child(BiDirect.RIGHT))

    @staticmethod
    def _post_order(op_fnc, node:BTNode):
        if node is not None:
            BinaryTree._post_order(op_fnc, node.get_child(BiDirect.LEFT))
            BinaryTree._post_order(op_fnc, node.get_child(BiDirect.RIGHT))
            op_fnc(node)

    # leftmost/rightmost children for in-order predecessor/successor
    @staticmethod
    def _farthest_child(d:BiDirect, node:BTNode) -> \
            Tuple[Optional[BTNode], BTNode]:
        '''
        returns (farthest, parent) along specified direction;
        if childless in specified direction returns (None, node)
        '''
        if node.get_child(d) is None:
            return (None, node)
        while node.get_child(d).get_child(d) is not None:
            node = node.get_child(d)
        return (node.get_child(d), node)

    @staticmethod
    def _farthest_path(d:BiDirect, node:BTNode) -> List[BTNode]:
        '''returns path to farthest child of node along specified direction'''
        node_path = []
        while node is not None:
            node_path.append(node)
            node = node.get_child(d)
        return node_path

    def in_order(self, op_fnc):
        self._in_order(op_fnc, self.root)

    def pre_order(self, op_fnc):
        self._pre_order(op_fnc, self.root)

    def post_order(self, op_fnc):
        self._post_order(op_fnc, self.root)


class BSTree(BinaryTree):
    # simple and unbalanced search tree; < on left, > on right
    # existing key leads to overwriting data
    def __init__(self):
        super().__init__()

    def find_node_and_parent(self, key) -> \
            Tuple[Optional[BTNode], Optional[BTNode], Optional[BiDirect]]:
        '''
        returns (node, parent, direction) if found;
        returns (node, None, None) if root;
        returns (None, last-seen, direction) if not found
        '''
        parent = None
        cur = self.root
        d = None
        while cur is not None:
            if key < cur.key:
                d = BiDirect.LEFT
            elif key > cur.key:
                d = BiDirect.RIGHT
            else: # found key
                break
            parent = cur
            cur = cur.get_child(d)
        return (cur, parent, d)

    def find_node_path(self, key) -> \
                Tuple[List[BTNode], List[Optional[BiDirect]]]:
        '''
        returns list of nodes possibly ending with node having provided key
        (list is returned even if not found, so check last node in caller)
        and list of directions taken to get there
        (direction list starts with None for 'direction to root'
        to maintain same length as node list)
        '''
        node_path:List[BTNode] = []
        dir_path:List[Optional[BiDirect]] = []
        cur = self.root
        d = None
        while cur is not None:
            dir_path.append(d)
            node_path.append(cur)
            if key < cur.key:
                d = BiDirect.LEFT
            elif key > cur.key:
                d = BiDirect.RIGHT
            else:
                break
            cur = cur.get_child(d)
        return (node_path, dir_path)

    @staticmethod
    def _rotate(d:BiDirect, pivot:BTNode) -> BTNode:
        '''
        d indicates new position pivot moves down towards;
        node is pivot node;

        returns the node that moved up to replace pivot node
        '''
        going_up = pivot.get_child(1 - d)
        detached_child = going_up.get_child(d)
        going_up.set_child(d, pivot)
        pivot.set_child(1 - d, detached_child)
        return going_up

    def add(self, key, data):
        '''add key -> data to tree but replaces data if key already exists'''
        new_node = BTNode(key=key, data=data)
        if self.root is None:
            self.root = new_node
            self._count += 1
            return
        cur = self.root
        visit_next = cur
        d = None 
        while visit_next is not None:
            if new_node.key < cur.key:
                d = BiDirect.LEFT
            elif new_node.key > cur.key: 
                d = BiDirect.RIGHT
            else:
                cur.data = data
                # new_node unused and removed from existence
                return
            visit_next = cur.get_child(d)
            if visit_next is not None:
                # loop updates cur until cur is insertion parent
                cur = visit_next
        cur.set_child(d, new_node)
        self._count += 1

    def get(self, key) -> Tuple[Any, Any]:
        '''returns (key, val) if found, else (None, None)'''
        cur = self.find_node_and_parent(key)[0]
        return (None, None) if cur is None else (cur.key, cur.data)

    def remove(self, key) -> Tuple[Any, Any]:
        '''removes and returns (key, val) if found, else (None, None)'''
        cur, parent, d = self.find_node_and_parent(key)
        if cur is None:
            return (None, None)
        replacement = None
        left_child = cur.get_child(BiDirect.LEFT)
        right_child = cur.get_child(BiDirect.RIGHT)
        if right_child is None:
            # rightless node must go left (left could be None)
            replacement = left_child
        elif right_child.get_child(BiDirect.LEFT) is None:
            # leftmost child of leftless node is node itself
            right_child.set_child(BiDirect.LEFT, left_child)
            replacement = right_child
        else:
            # leftmost child of right
            leftmost, leftmost_parent = \
                    self._farthest_child(BiDirect.LEFT, right_child)
            leftmost_parent.set_child(BiDirect.LEFT,
                    leftmost.get_child(BiDirect.RIGHT))
            leftmost.set_child(BiDirect.LEFT, left_child)
            leftmost.set_child(BiDirect.RIGHT, right_child)
            replacement = leftmost
        if parent is None:
            self.root = replacement
        else:
            parent.set_child(d, replacement)
        # in non-memory-managed language, deallocate cur before return
        # (need to store key and data beforehand)
        self._count -= 1
        return (cur.key, cur.data)

class BalanceTree(BSTree):
    # abstract class to hold some common balancing operations
    # (with balancing method undefined, not useful to instantiate)
    def __init__(self):
        super().__init__()

    def _balance_fix(self, cur:BTNode, parent:Optional[BTNode],
                d:Optional[BiDirect]):
        '''
        base BalanceTree _balance_fix() only raises NotImplementedError; 
        override and implement balancing rules in subclass
        '''
        raise NotImplementedError

    def _rebalance_node_path(self, node_path:List[BTNode],
                             dir_path:List[Optional[BiDirect]],
                             single_action:bool=False):
        '''
        node_path is list of nodes along some traversal path;

        dir_path holds the directions from node at [idx - 1] to node at [idx];
        if dir_path[0] is not root, no adjustment is performed on the node
        (because parent for attachment is unclear);

        single_action indicates whether at most one adjustment is needed
        (after AVL insertions, one operation should be enough to fix balance
        with a double rotation considered a single operation)

        _balance_fix(current_node, parent_of_current, direction)
        must be implemented in subclass for this method to work
        '''
        for idx in range(len(node_path) - 1, 0, -1):
            if (self._balance_fix(node_path[idx], node_path[idx - 1], 
                    dir_path[idx]) and single_action):
                # single_action must be second because of short circuit eval
                return
        if node_path[0] is self.root:
            self._balance_fix(node_path[0], None, dir_path[0])

    def _add_node_and_get_path(self, new_node:BTNode) -> \
                Tuple[List[BTNode], List[Optional[BiDirect]]]:
        '''
        helper method for subclasses to add node while recording path;
        if key exists, data is overwritten and new_node is discarded instead;

        returns (list of nodes, list of directions) if new_node added to tree
        (inclusive of new_node even though new_node already known to caller);
        returns (blank list, blank list) if new_node discarded
        '''
        if self.root is None:
            self.root = new_node
            self._count += 1
            return ([self.root], [None])
        node_path, dir_path = self.find_node_path(new_node.key)
        last_node = node_path[-1]
        if last_node.key == new_node.key:
            last_node.data = new_node.data
            # new_node unused so no path to new_node in tree
            return ([], [])
        d = BiDirect.LEFT if last_node.key > new_node.key else BiDirect.RIGHT
        last_node.set_child(d, new_node)
        self._count += 1
        node_path.append(new_node)
        dir_path.append(d)
        return (node_path, dir_path)

    def _remove_node_and_get_path(self, key) -> \
            Tuple[Optional[BTNode], List[BTNode], List[BiDirect]]:
        '''
        analogue of _add_node_and_get_path() for deletions;
        pops node if found and returns path of nodes that may have balance
        affected by removal of popped node;

        returns (node, list of nodes, list of directions) if node found;
        returns (None, blank list, blank list) if key not found;
        '''
        node_path, dir_path = self.find_node_path(key)
        if len(node_path) == 0 or node_path[-1].key != key:
            return (None, [], [])
        cur = node_path.pop()
        # len 0 node_path after pop means cur is root
        parent = None if len(node_path) == 0 else node_path[-1]
        d = dir_path[-1]
        replacement = None
        left_child = cur.get_child(BiDirect.LEFT)
        right_child = cur.get_child(BiDirect.RIGHT)
        if right_child is None:
            replacement = left_child
        elif right_child.get_child(BiDirect.LEFT) is None:
            replacement = right_child
            right_child.set_child(BiDirect.LEFT, left_child)
            node_path.append(replacement)
        else:
            left_path:List[BTHNode] = self._farthest_path(BiDirect.LEFT,
                    right_child)
            # unlike node_path, we know left_path has at least two nodes
            leftmost = left_path[-1]
            leftmost_parent = left_path[-2]
            leftmost_parent.set_child(BiDirect.LEFT, 
                    leftmost.get_child(BiDirect.RIGHT))
            leftmost.set_child(BiDirect.LEFT, left_child)
            leftmost.set_child(BiDirect.RIGHT, right_child)
            replacement = leftmost
            node_path.append(replacement)
            # left_path doesn't overlap with node_path
            node_path += left_path[:-1]
            # node_path has length (|initial_node_path| - 1 + |left_path|)
            # with additional traversal at one right then all lefts
            dir_path.append(BiDirect.RIGHT)
            dir_path += [BiDirect.LEFT] * (len(left_path) - 2)
        if parent is None:
            self.root = replacement
        else:
            parent.set_child(d, replacement)
        self._count -= 1
        return (cur, node_path, dir_path)


class AVLTree(BalanceTree):
    def __init__(self):
        super().__init__()

    @classmethod
    def _rotate(cls, d:BiDirect, pivot:BTHNode) -> BTHNode:
        '''
        d indicates position node moves down towards;
        node is pivot and parent is needed to reattach after pivoting;
        right child must exist if rotating left, left child if rotating right;

        this method sets needs_height but doesn't resolve heights;
        be sure to resolve heights (starting from pivot's parent) in caller
        '''
        going_up:BTHNode = pivot.get_child(1 - d)
        if going_up is None:
            raise TypeError('Cannot rotate None upwards')
        child_heights = going_up.child_heights()
        if child_heights[d] > child_heights[1 - d]:
            # double rotate when child is heavy on opposing side
            new_up:BTHNode = going_up.get_child(d)
            new_up.needs_height = True
            pivot.set_child(1 - d, super()._rotate(1 - d, going_up))
        going_up.needs_height = pivot.needs_height = True
        return super()._rotate(d, pivot)

    def _balance_fix(self, cur:BTHNode, parent:Optional[BTHNode], 
                        d:Optional[BiDirect]) -> bool:
        '''
        fixes balance of node by rotating if necessary; 

        parent should be None if cur is root (and not None if cur not root);
        d should not be None if cur is not root;

        returns True if rotation happened, False if not
        '''
        is_root = (cur is self.root)
        if not is_root and parent is None:
            raise ValueError('parent not provided')
        left_height, right_height = cur.child_heights()
        rotated:Optional[BTHNode] = None
        if left_height - right_height > 1: # left heavy
            rotated = self._rotate(BiDirect.RIGHT, cur)
        elif right_height - left_height > 1: # right heavy
            rotated = self._rotate(BiDirect.LEFT, cur)
        else: # balance within |1| so no action necessary
            return False
        if is_root:
            parent = self.root = rotated
        else:
            parent.set_child(d, rotated)
            parent.needs_height = True
        parent.determine_height()
        return True

    def add(self, key, data):
        new_node = BTHNode(key=key, data=data)
        node_path, dir_path = self._add_node_and_get_path(new_node)
        node_path = cast(List[BTHNode], node_path)
        if len(node_path) <= 1:
            # nothing to balance after adding root or if new_node discarded
            return  
        for node in node_path[:-1]:
            node.needs_height = True
        # new_node is excluded from rebalancing because it's a terminal node
        self._rebalance_node_path(node_path[:-1], dir_path[:-1], True)
        self.root.determine_height()

    def remove(self, key) -> Tuple[Any, Any]:
        found, node_path, dir_path = self._remove_node_and_get_path(key)
        if found is None:
            return (None, None)
        node_path = cast(List[BTHNode], node_path)
        for node in node_path:
            node.needs_height = True
        self._rebalance_node_path(node_path, dir_path)
        self.root.determine_height()
        return (found.key, found.data)


class RBTree(BalanceTree):
    # for nodes, red is color, black is noncolor
    # based on Sedgewick's left-leaning RB trees
    # (lean direction can be flipped at instantiation)
    def __init__(self, lean:BiDirect=BiDirect.LEFT):
        super().__init__()
        self._lean = lean

    def lean(self):
        '''
        returns direction of tree's lean
        (because _lean property shouldn't be altered directly)
        '''
        return self._lean

    def set_lean(self, lean:BiDirect):
        '''
        change lean if tree is empty;
        throws exception if tree not empty
        '''
        if self._count > 0:
            raise RuntimeError('tree already has nodes')
        self._lean = lean

    @classmethod
    def _rotate(cls, d:BiDirect, pivot:BTCNode) -> BTCNode:
        '''
        d indicates position node moves down towards;
        node is pivot and parent is needed to reattach after pivoting;
        right child must exist if rotating left, left child if rotating right;

        returns node that moved into pivot's position;

        RBTree rotate adjusts color on involved nodes to maintain edge color
        '''
        going_up:BTCNode = pivot.get_child(1 - d)
        if going_up is None:
            raise TypeError('Cannot rotate None upwards')
        # color changes keep edges proper color through rotation
        # (color of node indicates color of incoming edge)
        down_edge_color = going_up.color
        going_up.color = pivot.color
        pivot.color = down_edge_color
        return super()._rotate(d, pivot)

    # not a class or static method because dependent on tree's lean
    def _move_edge_color(self, d:BiDirect, cur:BTCNode) -> BTCNode:
        '''
        like _rotate(), direction indicates position red edge moves towards;
        goal is to adjust structure while traversing so that red edges
        are used through path; 
        terminal red removal wouldn't violate tree structure;

        makes cur's child red or one of that child's children red;

        returns node that replaces cur if structure changes
        (returns cur if cur remains in same position after edge color changes)
        '''
        # Sedgewick mentions that color-flip is used to split cur's red edge
        # to its two children, but this check never happens in the code logic;
        # we assume flip_colors() is valid because of tree's structure
        # constraints being maintained by other methods that would call
        # flip_colors() or _move_edge_color()
        cur.flip_colors()
        if (cur.is_colored(cur.get_child(1 - d).get_child(self._lean))):
            if d == self._lean:
                cur.set_child(1 - d, 
                    self._rotate(1 - d, cur.get_child(1 - d)))
            cur = self._rotate(d, cur)
            cur.flip_colors()
        return cur

    def _balance_node(self, cur:BTCNode) -> BTCNode:
        '''
        helper for _balance_fix() that takes cur and returns fixed cur;
        fixes color violations by rotating or flipping edge colors;
        moved out for readability but still only called by _balance_fix()
        
        returns cur or cur's replacement after necessary adjustments
        '''
        # comments based on left-leaning, reverse for right-lean
        # black L <- node -> red R to node <- red R
        if (cur.is_colored(cur.get_child(1 - self._lean)) and 
                not cur.is_colored(cur.get_child(self._lean))):
            cur = self._rotate(self._lean, cur)
        # red LL <- red L <- node; rotate to red LL <- L -> red node
        # (colors will be flipped shortly after)
        if (cur.is_colored(cur.get_child(self._lean)) and cur.is_colored(
                cur.get_child(self._lean).get_child(self._lean))):
            cur = self._rotate(1 - self._lean, cur)
        # split double-red children and pass red up a level
        if (cur.is_colored(cur.get_child(BiDirect.RIGHT)) and
                cur.is_colored(cur.get_child(BiDirect.LEFT))):
            # flip_colors() would fail to pass red children edges up if node
            # is already red, but proper RB trees won't have this situation
            cur.flip_colors()
        return cur

    def _balance_fix(self, cur:BTCNode, parent:Optional[BTCNode],
            d:Optional[BiDirect]):
        '''
        fixes color violations by rotating or flipping edge colors;

        parent should be None if cur is root (and not None if cur not root);
        d should not be None if cur is not root;

        like AVLTree _balance_fix(), returns True if adjustment happened
        and returns False if nothing changed;
        (this information is unused for RBTree path balancing)
        '''
        is_root = cur is self.root
        if not is_root and parent is None:
            raise ValueError('parent not provided')
        initial_cur = cur
        initial_color = cur.color
        cur = self._balance_node(cur)
        if is_root:
            self.root = cur
        else:
            parent.set_child(d, cur)
        return initial_cur is not cur or initial_color != cur.color
  
    def add(self, key, data):
        new_node = BTCNode(key=key, data=data)
        node_path, dir_path = self._add_node_and_get_path(new_node)
        # rebalancing not needed with single node or unchanged structure
        if len(node_path) > 1:
            # new_node excluded because it won't need adjustment
            # (conditions in _balance_fix() never apply to terminal node)
            self._rebalance_node_path(node_path[:-1], dir_path[:-1])
        # maintain uncolored root convention (no incoming edge to color)
        self.root.color = False

    def _rb_farthest_path(self, d:BiDirect, cur:BTCNode) -> List[BTCNode]:
        '''
        traverses tree starting from cur to farthest node along direction;
        moves red edges to traversal path as necessary along the way;
        (this method doesn't reuse standard farthest methods because of
        structural adjustment on the way down to final node)

        returns path to node without fixing tree structure after traversal,
        so must fix structure in caller with path information
        '''
        # like balance methods, comments based on left-leaning
        node_path:List[BTCNode] = []
        while cur is not None:
            # move left red edge over if traversing down right 
            if d != self._lean and cur.is_colored(cur.get_child(self._lean)):
                cur = self._rotate(d, cur)
            if (cur.get_child(d) is not None and not 
                    (cur.is_colored(cur.get_child(d)) or 
                    cur.is_colored(cur.get_child(d).get_child(self._lean)))):
                cur = self._move_edge_color(d, cur)
            node_path.append(cur)
            cur = cur.get_child(d)
        return node_path

    def remove(self, key) -> Tuple[Any, Any]:
        # comments based on left-lean logic
        kv_pair = (None, None)
        if self.root is None:
            return kv_pair
        node_path:List[BTCNode] = []
        dir_path:List[BiDirect] = []
        cur = self.root
        d = None
        while cur is not None:
            dir_path.append(d)
            if key < cur.key:
                d = BiDirect.LEFT
            elif key > cur.key:
                d = BiDirect.RIGHT
            else: # found node, but may not be done adjusting structure
                d = 1 - self._lean
            # attempt to rotate over red edge from left if moving right
            if (self._lean != d and 
                    cur.is_colored(cur.get_child(self._lean))):
                cur = self._rotate(d, cur)
            # move red to path with _move_color()
            if (cur.get_child(d) is not None and not
                    (cur.is_colored(cur.get_child(d)) or
                    cur.is_colored(cur.get_child(d).get_child(self._lean)))):
                cur = self._move_edge_color(d, cur)
            node_path.append(cur)
            if cur.key == key:
                break # stop traversal if cur has desired key after adjustment
            cur = cur.get_child(d)
        self.root = node_path[0] # re-root in case root rotated during loop
        if node_path[-1].key == key:
            # mostly same as AVL replacement finding with extra steps
            cur = node_path.pop()
            kv_pair = (cur.key, cur.data)
            parent = None if len(node_path) == 0 else node_path[-1]
            d = dir_path[-1]
            replacement = None
            # anti because it's against lean direction
            anti_child = cur.get_child(1 - self._lean)
            # initial left/right checks from AVL should be unnecessary;
            # nodes (and red edges) are pushed into position during 
            # traversal so getting in-order replacement should be possible
            if anti_child is not None:
                # unless cur is childless, in which case this block is skipped
                lean_path = self._rb_farthest_path(self._lean, anti_child)
                leanmost = lean_path.pop()
                # this assertion should probably hold
                # assert(leanmost.color == True)
                node_path.append(leanmost)
                leanmost.set_child(self._lean, cur.get_child(self._lean))
                leanmost.color = cur.color
                if len(lean_path) > 0:
                    leanmost_parent = lean_path[-1]
                    leanmost_parent.set_child(self._lean, 
                            leanmost.get_child(1 - self._lean))
                    leanmost.set_child(1 - self._lean, anti_child)
                    node_path += lean_path
                    dir_path.append(1 - self._lean)
                    dir_path += [self._lean] * (len(lean_path) - 1)
                replacement = leanmost
            # cur can be deallocated now
            if parent is None:
                self.root = replacement
            else:
                parent.set_child(d, replacement)
            self._count -= 1
        # regardless of removal success, node_path needs fixing
        if len(node_path) > 0:
            self._rebalance_node_path(node_path, dir_path)
            self.root.color = False
        return kv_pair
