#!/usr/bin/python
#Project 4:
#Group 23
#Members: Albert Chang, Royce Hwang, Jessica Huang
#11/25/2016
#
#Purpose: Given a set of n cities c_i with their x, y-coordinates, a minimal
#   tour is found. This tour must start at one city, include all other cities, 
#   and return to the original city. The distance traveled is the sum of the 
#   distances traveled from beginning city, through all the cities, and back
#   to the beginning city. The distances computed are all rounded to the nearest
#   integer value.
#
#   The input is a text file with the city's id, x-coordinate, and y-coordinate.
#   The output is a text file with the length of the tour and the order of the 
#   tour listed by city identification number.
#
#A sample of the input, since the class files aren't included:
#
#(SAMPLE STARTS BELOW)
#0 200 40
#1 12 34
#2 430 20
#(SAMPLE ENDS ABOVE)
#
#(Without the # signs in an actual input file.)
#The class files all used integer coordinates, but given Python's non-strict typing
#for variables, floating point should work as well. It may be safer to make all coordinates
#floating point if doing so, even whole number coordinates (like 1.0), to avoid unexpected
#operator bugs from mixing different numerical types.

from __future__ import print_function
import os
import math
import heapq
import copy
import sys
#TIMING - Uncomment these lines in order to enable timing functionality 
#import timeit

#Class for nodes
#Pretty much a struct, with two simple helper functions.
#Most data will be accessed like a struct, with dot notation.
class cityNode:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y
        self.key = float("inf")
        self.pred = None 
        #A "pointer" to a different node, but Python doesn't have real pointers.
    
    #As mentioned, nodes are pretty much just dumb structs, but "distance"
    #can be used as a helper, like firstNode.distanceTo(secondNode).
    #This is in no way better than having a separate function, just a matter of preference.
    def distanceTo(self, otherNode):
        return int(round(math.hypot(otherNode.y - self.y, otherNode.x - self.x) ) )
    
    #We'll probably never need this, but it could help if we want to do more later.
    def resetNode(self):
        self.key = float("inf")
        self.pred = None
        
        
        
#A class for graphs
#This is like an intermediate between struct and class.
#More helpers than the cityNode for graph adjustment,
#but some data still meant to be accessed via dot notation.
class tspGraph:
    def __init__(self):
        self.vertices = {}
        self.edges = {}
        #Hash tables for both. Not sure if this makes sense.
        #Our initial graph won't have any entries to edges but the MST will.
        #(it will have the minimum total weight's worth of edges)
        self.edgeCount = 0
        self.nodeCount = 0
        
    def addNode(self, nodeObject):
        self.vertices[nodeObject.name] = nodeObject #Associating a name with the actual object.
        self.edges[nodeObject.name] = {} #An edge list, in hash form.
        self.nodeCount += 1
        
    def removeNode(self, nodeName):
        self.vertices.pop(nodeName, None)
        #Not sure we should remove edge list too.
        self.nodeCount -= 1
        #The next part (for edgeCount) will only be accurate if the graph
        #has NO duplicate edges. So it will work for MST, but not the multigraph.
        self.edgeCount -= len(self.edges[nodeName])
        self.edges.pop(nodeName)

        
    def addEdge(self, startName, endName, distance):
        #Edge list uses hash tables too.
        #So the edge list is a hash table of hash tables (of lists).
        #Keeping edges as lists may not make sense, but we need a way
        #for the multigraph to have repeated edges.
        #When figuring out a tour, we'll pop edges from the list.
        
        #Additional verification, not necessary if we know that edges['foo']
        #is initialized as an empty dict
        #(a matter of preference again, between initialisation here or above in addNode)
        #if startName not in self.keys():
        #    self.edges[startName] = {}
        if endName not in self.edges[startName].keys():
            self.edges[startName][endName] = []
        self.edges[startName][endName].append(distance)
        self.edgeCount += 1
        
    def popEdge(self, startName, endName):
        if endName in self.edges[startName].keys():
            returnValue = self.edges[startName][endName].pop()
            if len(self.edges[startName][endName]) == 0:
                #If there are no more edges from startName to endName,
                #remove the key from the dictionary.
                self.edges[startName].pop(endName)
            self.edgeCount -= 1
            return returnValue
            
        return None 
        #If not possible, give None.
        
    #This is simple to find and might be useful.
    def isEdgeless(self, nodeName):
        if len(self.edges[nodeName]) == 0:
            return True
        return False
        
    def getNode(self, nodeName):
        return self.vertices[nodeName]
        
        
        
#Custom priority queue class based off the standard Python heapq
#https://docs.python.org/2/library/heapq.html
class PriorityQueue:
    def __init__(self):
        self.queue = []
        self.size = 0
    
    def push(self, distance, city):
        heapq.heappush(self.queue, (distance, city))
        self.size += 1

    def pop(self):
        self.size -= 1
        return heapq.heappop(self.queue)
        
    def peek(self):
        return self.queue[0]

    def empty(self):
        return (self.size == 0)
        
    def oldDecreaseKey(self, start, pos):
        heapq._siftdown(self.queue, start, pos)
    
    #this is the custom part, for Prim's.
    def decreaseKey(self, weight, nodeName):
        foundIndex = -1
        for index in xrange(self.size):
            #[index] gets the vertex name tuple.
            #[1] gets the city name part.
            #This is a very specific implementation for Prim's, and not very
            #extensible to other purposes of using the priority queue.
            if self.queue[index][1] == nodeName:
                foundIndex = index
                break
        if foundIndex >= 0:
            self.queue[index] = (weight, nodeName)
        heapq._siftdown(self.queue, 0, foundIndex)



#Prim's algorithm
#This is a slight simplification based on the fact that our graphs are complete
# graphs in 2D space.
#"startName" is the name of the starting node, not the actual node object.
#For actually running, we can always have startName = 0. Since the nodes are numbered.
#"None" would be a safer default option, but this is highly specialised for the assignment.
#RUNTIME: Our PQ is binary heap based, so we have O(E lg V)
#These are complete graphs, so it's O(V^2 lg V)
#This means array would be better, but binary heap is useful in later parts too.
def prim(graph, startName = 0):
    pQueue = PriorityQueue()
    addedNodes = set()
    startNode = graph.getNode(startName)
    startNode.mstKey = 0 #0 distance to itself.
    addedNodes.add(startName)
    mstGraph = tspGraph()
    mstGraph.addNode(startNode)
    for nodeName in graph.vertices:
        if nodeName != startName:
            currentNode = graph.getNode(nodeName)
            currentNode.key = startNode.distanceTo(currentNode)
            currentNode.pred = startNode #Actual pred pointer, not just name.
            pQueue.push(currentNode.key, nodeName)
    while not pQueue.empty(): #while pQueue not empty
        weight, currentName = pQueue.pop()
        #weight is discarded, this just allows us to access tuple's second part easier.
        if currentName not in addedNodes:
            #Some basic assignment of variables.
            currentNode = graph.getNode(currentName)
            predNode = currentNode.pred
            currentDistance = predNode.distanceTo(currentNode)
            #Add the newly popped node to our ongoing MST.
            addedNodes.add(currentName)
            mstGraph.addNode(currentNode)
            #Our graphs are undirected, so we add two edges per edge.
            mstGraph.addEdge(predNode.name, currentNode.name, currentDistance)
            mstGraph.addEdge(currentNode.name, predNode.name, currentDistance)
            #Updated decrease-key logic.
            #For each node in the set of graph vertices:
            #(This is usually just all adjacent nodes, but our graphs are complete graphs)
            for outsideName in graph.vertices:
                #If that node is not already added to our MST
                if outsideName not in addedNodes:
                    #Get the actual node object for reference
                    outsideNode = graph.getNode(outsideName)
                    #Calculate distance from most recently added node to that outside node
                    newDistance = currentNode.distanceTo(outsideNode)
                    #If distance is smaller than the previously seen smallest distance
                    if newDistance < outsideNode.key:
                        #We relax and decrease the key
                        pQueue.decreaseKey(newDistance, outsideName)
                        outsideNode.key = newDistance
                        outsideNode.pred = currentNode
    return mstGraph



#The next step in Christofides' would be minimum weight matching of odd-degree vertices.
#The minimum weight matching would produce a set of edges that we add to
# our MST (making it no longer a valid MST), and then we'd attempt to create
# an Euler tour through the graph where we skip repeated vertices.
#Actual minimum matching would be too slow, so we instead develop a greedy approximation.

#Determines the vertices of odd degree in a MST, and creates an approximation 
# for the minimum weight perfect match (all vertices with exactly 1 edge) using
# a greedy choice.
#These edges are layered with the MST's edges and returned in a new tspGraph object.
#The input mstGraph is the minimum spanning tree returned by prim()
#Copying the graph is roughly O(V). It's just copying vertex list and edge lists.
#Keep in mind that MST has E = O(V), so it's V+V work, for O(V) work.
#Finding odd vertices is O(V). It iterates over all vertices, constant time comparisons.
#The greedy choice part is O(V^2). This involves iterating over all odd vertices in a double loop.
#Is it possible for all vertices in the graph to be odd in the MST? Yes, but maybe not often.
#Think of a T-shaped graph of four vertices.
#The loops actually do V + V-1 + V-2 ... work, and only up to half the number of odd vertices.
#But doing the sum, dividing by two, still gives a V^2 term, and we drop constants and multipliers.
#Overall, this is O(V^2).
def getMinPerfectMatch(mstGraph):

    #Returns the subgraph containing all vertices of odd degree
    minGraph = tspGraph()
    minGraph.vertices = copy.copy(mstGraph.vertices)
    minGraph.edges = copy.deepcopy(mstGraph.edges)
    minGraph.edgeCount = mstGraph.edgeCount
    
    oddVertices = []
    
    for vertexName in mstGraph.vertices:
        
        #This is not always accurate for tspGraph now because of the lists for edges,
        #but it will still remain accurate for a MST because a MST won't have
        #duplicate edges (aside from the bidirectional-ness)
        if len(mstGraph.edges[vertexName]) % 2 == 1:
            #newNode = mstGraph.getNode(vertexName)
            oddVertices.append(vertexName)

    
    #Greedy method for determining perfect matching - selects lowest distance
    #for a given vertex until all vertices in minGraph are covered
    coveredVertices = set()
    for index_1 in xrange(len(oddVertices) ):
        vertexName_1 = oddVertices[index_1]
        if vertexName_1 not in coveredVertices:
            minDistance = float("INF")
            #I think this should help, but I'm not sure.
            #idea is that when vertexName_1 is already covered,
            #we still need minVertexName variables.
            minVertexName_1 = None
            minVertexName_2 = None
            
            for index_2 in xrange(index_1, len(oddVertices) ):
                vertexName_2 = oddVertices[index_2]
                if vertexName_2 not in coveredVertices and vertexName_1 != vertexName_2:
                    vertex1 = minGraph.getNode(vertexName_1)
                    vertex2 = minGraph.getNode(vertexName_2)
                    distance = vertex1.distanceTo(vertex2)
                        
                    if distance < minDistance:
                        minDistance = distance
                        minVertexName_1 = vertexName_1
                        minVertexName_2 = vertexName_2

            #Extra conditional helps to prevent "blank" nodes from being added.
            if minVertexName_1 is not None and minVertexName_2 is not None:
                #Graph is undirected, two edges are added per vertex pair            
                minGraph.addEdge(minVertexName_1, minVertexName_2, minDistance)
                minGraph.addEdge(minVertexName_2, minVertexName_1, minDistance)
                
                #Since each vertex can only have one edge, both are now covered
                coveredVertices.add(minVertexName_1)
                coveredVertices.add(minVertexName_2)
        
    
    return minGraph



#This is unused. An alternate, slower greedy method.
#It provides better approximations in some situations by finding a local minimum,
#but we believe it adds too much to the runtime to be useful.
#No running time analysis for this one because we felt that it would be too slow
#for large graphs and ended up never using it.
#Note the peek method of priority queues is only ever used in this unused function.
def altMinMatch(mstGraph):

    #Returns the subgraph containing all vertices of odd degree
    minGraph = tspGraph()
    minGraph.vertices = copy.copy(mstGraph.vertices)
    minGraph.edges = copy.deepcopy(mstGraph.edges)
    minGraph.edgeCount = mstGraph.edgeCount
    
    oddVertices = []
    
    oddPQDict = {}
    
    for vertexName in mstGraph.vertices:
        
        #This is not always accurate for tspGraph now because of the lists for edges,
        #but it will still remain accurate for a MST because a MST won't have
        #duplicate edges (aside from the bidirectional-ness)
        if len(mstGraph.edges[vertexName]) % 2 == 1:
            #newNode = mstGraph.getNode(vertexName)
            oddVertices.append(vertexName)
            #We create a PQ so we can match each odd vertex with its optimal odd neighbor.
            oddPQDict[vertexName] = PriorityQueue()
    

    #Greedy method for determining perfect matching - selects lowest distance
    #for a given vertex until all vertices in minGraph are covered
    coveredVertices = set()
    coveredCount = 0
    oddCount = len(oddVertices)
    for index_1 in xrange(oddCount):
        for index_2 in xrange(oddCount):
            vertexName_1 = oddVertices[index_1]
            vertexName_2 = oddVertices[index_2]
            if vertexName_1 != vertexName_2:
                vertex1 = minGraph.getNode(vertexName_1)
                vertex2 = minGraph.getNode(vertexName_2)
                distance = vertex1.distanceTo(vertex2)
                    
                oddPQDict[vertexName_1].push(distance, vertexName_2)
                
    while coveredCount < oddCount:
        for vertexName_1 in oddVertices:
            if vertexName_1 not in coveredVertices:
                #peek()[1] gets the second part of the tuple, the city's number
                while oddPQDict[vertexName_1].peek()[1] in coveredVertices:
                    oddPQDict[vertexName_1].pop()
                vertexName_2 = oddPQDict[vertexName_1].peek()[1]
                while oddPQDict[vertexName_2].peek()[1] in coveredVertices:
                    oddPQDict[vertexName_2].pop()
                if oddPQDict[vertexName_2].peek()[1] == vertexName_1:
                    coveredVertices.add(vertexName_1)
                    coveredVertices.add(vertexName_2)
                    coveredCount += 2
                    #Getting the edge length. Could get from either PQ.
                    distance = oddPQDict[vertexName_2].peek()[0]
                    
                    minGraph.addEdge(vertexName_1, vertexName_2, distance)
                    minGraph.addEdge(vertexName_2, vertexName_1, distance)

    return minGraph
    


#This is an euler path but not a hamiltonian circuit.
#Source: http://www.graph-magics.com/articles/euler.php
#No edge weights yet in this function.
#The graph used is the multigraph created earlier to ensure all vertices have even degree.
#startName is again the name of the starting node, not the actual node object.
#This is O(V) time. A bit harder to explain, but think of it this way.
#The MST has E = O(V), because MST has V-1 edges.
#The multigraph has additional edges compared to the MST.
#The additional edges number V/2, at most. (At most one edge per two odd vertices.)
#So the maximum number of edges in the multigraph is around 1.5V.
#That's still O(V) edges.
#This function iterates through all edges, traversing them.
#This is done in O(V) time, because the work done at each step is constant time.
#It's possible we end up with many vertices on the stack, and then have to pop them all,
#but that too is O(V) time. The stack can't have more than 1.5V elements.
#So overall, everything is O(V) time.
def getEulerTour(graph, startName = 0):
    tourList = []
    tourGraph = tspGraph()
    tourGraph.vertices = copy.copy(graph.vertices)
    tourGraph.edges = copy.deepcopy(graph.edges)
    tourGraph.edgeCount = graph.edgeCount
    
    #We need this in case we make a circuit before using all edges.
    nodeStack = []
    
    currentName = startName

    while tourGraph.edgeCount > 0:
        if len(tourGraph.edges[currentName].keys() ) > 0:
            #Add node with edges to stack.
            nodeStack.append(currentName)
            #Choose random edge from currentNode's list.
            #The 0th neighbor is non-random, but dictionaries don't have set order anyway.
            nextName = tourGraph.edges[currentName].keys()[0]
            #Bidirectional edges, so we pop both ends.
            #We could pop a single end, but that would make this algorithm take more time.
            tourGraph.popEdge(currentName, nextName)
            tourGraph.popEdge(nextName, currentName)
            currentName = nextName
        else:
            tourList.append(currentName)
            currentName = nodeStack.pop()
            
    while len(nodeStack) > 0:
        currentName = nodeStack.pop()
        tourList.append(currentName)

    #In a directed graph, we would have to reverse this to be accurate.
    #But our graphs aren't directed, so we don't have to worry about that.
    return tourList



#A function to turn a list of node names into a list for a Hamiltonian circuit.
#This function doesn't even need the graph, since it's just removing duplicates.
#(There's an implicit assumption that nodes are all on a complete graph.)
#Again, no edge weights yet.
#pathList is a list of node names.
#This too is O(V) time. It does constant work, iterating over the Euler tour we created.
#But that tour has at most 1.5V elements, so this too can iterate at most 1.5V times.
#Appending items to the list is clearly constant time. So overall, it's O(V) time.
def tourToCircuit(pathList):
    seenNodes = set()
    shortList = []
    for index in xrange(len(pathList)):
        currentName = pathList[index]
        if currentName not in seenNodes:
            shortList.append(currentName)
            seenNodes.add(currentName)
    
    #Can't forget to make it into a circuit.
    #The fact that we use a set to create the list means we will always have
    #a non-circuit at the end, which we fix by simply appending the starting node.
    shortList.append(shortList[0])
    return shortList



#A function to take the circuit information and create our solution.
#This doesn't actually need edge lists, since we can use calculate distances directly.
#We still need the graph to be able to get cityNode objects.
#Note that while this takes "pathList" as an argument, we are passing it the
#list obtained from the tourToCircuit function, not the list obtained from eulerTour.
#The graph can be any graph that contains all necessary vertices.
#We are only calculating distances, not traversing actual edges.
#(This is specifically for the nodes on a 2D-plane like what we're working with.)
#This is doing constant work, while iterating over the Hamiltonian circuit.
#It should be plain to see that this is O(V) time.
#There are V+1 nodes in the Hamiltonian path, so the loop runs V+1 times, doing constant work.
def totalPathCost(graph, pathList):
    totalCost = 0
    #Starts at 1 because of some indexing math.
    for index in xrange(1, len(pathList) ):
        startName = pathList[index - 1]
        endName = pathList[index]
        startNode = graph.getNode(startName)
        endNode = graph.getNode(endName)
        distance = startNode.distanceTo(endNode)
        totalCost += distance
        
    return totalCost



#slower, more precise 2-opt
#Chooses the best two edges to swap each round, up to specified rounds.
#source: https://en.wikipedia.org/wiki/2-opt
#The input graph can be any graph with all necessary vertices.
#hamPath needs to be a Hamiltonian circuit. (The name is somewhat deceptive.)
#maxRounds is an integer representing a hard limit to optimisation rounds.
#This is O(V^2) time, with a caveat.
#The V^2 is multiplied by maxRounds, which can be anything we choose (up to V).
#It should be clear then that with a maximal maxRounds, this is O(V^3) time.
#Disregarding maxRounds for a moment, this iterates through every edge in hamPath
#and compares it to every following edge in the path. This would be
#(V+1) + V + (V-1) + (V-2) ... work in the various iterations, and sum to
#about V^2 with some constants and multipliers.
#Within each iteration (each of the V^2 total iterations of the double loop),
#the work should be constant.
#The list reversal to switch the edges can take potentially V steps (though
#Python likely handles this decently) which is insignificant compared to the V^2 loop iterations.
#Finally, as mentioned at the beginning, this can repeat up to V times,
#for V*V^2 = O(V^3).
def twoOpt(graph, hamPath, maxRounds):
    newPath = hamPath[:]
    for roundIndex in xrange(maxRounds):
        currentSwitch = None
        currentMin = float("INF")
        #Ranges determined with simple reasoning.
        #Suppose hamPath has length 10. Then index goes from 0 to 6.
        #jndex goes from index to 8. At maximum index of 6, jndex goes from 8 to 8.
        for index in xrange(0, len(newPath) - 4):
            for jndex in xrange(index + 2, len(newPath) - 2):
                edge1vert1 = graph.getNode(newPath[index])
                edge1vert2 = graph.getNode(newPath[index + 1])
                edge2vert1 = graph.getNode(newPath[jndex])
                edge2vert2 = graph.getNode(newPath[jndex + 1])
                edge1orig = edge1vert1.distanceTo(edge1vert2)
                edge2orig = edge2vert1.distanceTo(edge2vert2)
                edge1new = edge1vert1.distanceTo(edge2vert1)
                edge2new = edge2vert2.distanceTo(edge1vert2)
                
                if ( (edge2new + edge1new) < (edge1orig + edge2orig) ) and (edge2new + edge1new) < currentMin:
                    currentMin = edge1new + edge2new
                    #This tuple will make sense with Python list slicing later.
                    currentSwitch = (index + 1, jndex)
                    
        if currentSwitch is not None:
            startVert = currentSwitch[0]
            endVert = currentSwitch[1]
            #Reversing the middle part of the path to "switch" the edges.
            newPath[startVert:(endVert + 1)] = newPath[endVert:(startVert - 1):-1]
        else:
            #If no switch was made in a round, we are locally optimal and quit early.
            break
    return newPath

#fast 2-opt
#Chooses any two edges (that provide benefit) to swap each round, up to specified rounds.
#The runtime of this is similar to the slow 2-opt. The difference is that it has
#a shortcut. It attempts to do every useful switch as soon as they're found, regardless
#of whether they're locally optimal for the current state of the Hamiltonian path.
#But there's always the possibility that this shortcut doesn't help (if for some reason
#the only useful switch is always at the last two edges in the path, then second to
#last, third to last, etc.) so the end result is that not much time is saved.
#The logic is otherwise similar to the slow 2-opt, with the double loop and
#constant time work, so this is also O(V^2) * maxRounds.
#So this too would be O(V^3). But note that we do not actually run V rounds for
#most of the larger graphs, so it ends up being around O(V^2) in practice.
#(Inputs are the same as for the other 2-opt function.)
def twoOptFast(graph, hamPath, maxRounds):
    newPath = hamPath[:]
    for roundIndex in xrange(maxRounds):
        currentSwitch = None
        doubleBreak = False
        #Ranges determined with simple reasoning.
        #Suppose hamPath has length 10. Then index goes from 0 to 6.
        #jndex goes from index to 8. At maximum index of 6, jndex goes from 8 to 8.
        for index in xrange(0, len(newPath) - 4):
            for jndex in xrange(index + 2, len(newPath) - 2):
                edge1vert1 = graph.getNode(newPath[index])
                edge1vert2 = graph.getNode(newPath[index + 1])
                edge2vert1 = graph.getNode(newPath[jndex])
                edge2vert2 = graph.getNode(newPath[jndex + 1])
                edge1orig = edge1vert1.distanceTo(edge1vert2)
                edge2orig = edge2vert1.distanceTo(edge2vert2)
                edge1new = edge1vert1.distanceTo(edge2vert1)
                edge2new = edge2vert2.distanceTo(edge1vert2)
                
                if ( (edge2new + edge1new) < (edge1orig + edge2orig) ):
                    #This tuple will make sense with Python list slicing later.
                    currentSwitch = (index + 1, jndex)
                    doubleBreak = True
                    break
            
            if doubleBreak:
                break

        if currentSwitch is not None:
            startVert = currentSwitch[0]
            endVert = currentSwitch[1]
            #Reversing the middle part of the path to "switch" the edges.
            newPath[startVert:(endVert + 1)] = newPath[endVert:(startVert - 1):-1]
        else:
            #If no switch was made in a round, we are locally optimal and quit early.
            #Not actually optimal given the fast form makes the first swap it sees each time.
            break
    return newPath



# Takes in the file contents and converts each line to a vertex with integer 
# values of city id, x-coordinate, and y-coordinate. Puts the vertices into a 
# graph and returns the graph
def readFile(fileName):
    print("")
    print("Reading data from " + fileName)
    
    # Full path
    fullpath = os.path.dirname(__file__)
    newFile = open(os.path.join(fullpath, fileName), "r")
    
    #graph = []
    graph = tspGraph()
    
    # Reads in each line of input while lines can still be read
    while True:
        line = newFile.readline()

        # Stops when there is no more lines to read
        if not line:
            break

        # Gets rid of newlines
        line = line.split()
        line = [str.strip() for str in line]     
        
        # Converts the string values to integers
        line = [int(num) for num in line]
        city = cityNode(line[0], line[1], line[2])

        # Appends each vertex to the graph
        #graph.append(line)
        graph.addNode(city)

    print("Data successfully read from " + fileName)
    print("")
    newFile.close()
    return graph
    
#Writes data to user specified file name, appended with ".tour".

def writeFile(fileName, tourLength, hamPath):
    fileName = fileName + '.tour'
    print('')
    print("Writing results to " + fileName)

    #Opening file
    fullpath = os.path.dirname(__file__)
    newFile = open(os.path.join(fullpath, fileName), "w")

    #The first line is the length of the tour taken
    newFile.write(str(tourLength) + '\n')
    #List the cities visited in order
    for x in xrange(1, len(hamPath)):
        newFile.write(str(hamPath[x]) + '\n')

    newFile.close()
    print("Results successfully stored in " +  fileName)
    


#The actual program script starts here.

#TIMING - Uncomment these lines in order to enable timing functionality
#start_time = timeit.default_timer()

fileName = sys.argv[1]

graph = readFile(fileName)

print('Calculating TSP Solution...')

mstGraph = prim(graph)

minGraph = getMinPerfectMatch(mstGraph)

tourList = getEulerTour(minGraph)

hamPath = tourToCircuit(tourList)

newPath = hamPath
if len(hamPath) < 512:
    newPath = twoOpt(graph, hamPath, len(hamPath) )
elif len(hamPath) < 1024:
    #This is 512^3 / hamPath^2, so it will scale from 512 down to 128 at 1024.
    newPath = twoOptFast(graph, hamPath, pow(512,3)/pow(len(hamPath),2) )
elif len(hamPath) < 8192:
    #This goes from 128 at 1024 to 16 at 8192.
    newPath = twoOptFast(graph, hamPath, (128*1024) / len(hamPath) )
#Past this, no twoOpt is done. Neither fast nor long version.
#Main reason is that datasets much larger just take too long.

tourLength = totalPathCost(graph, newPath)
writeFile(fileName, tourLength, newPath)

#TIMING - Uncomment these lines in order to enable timing functionality
#elapsed_time = timeit.default_timer() - start_time
#print("Elapsed time" + str(elapsed_time))

#Uncomment this line to store times in "[inputFile].time" files
#writeTime(fileName, elapsed_time)

