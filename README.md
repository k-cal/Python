# Python
A repository for Python scripts.

While I've worked on several projects in Python, most of it is in algorithmic work rather than interactive software.

### TSP
This is an approximation of Christofides' algorithm for approximating the Travelling Salesman Problem. The algorithm has a step that strongly deviates from Christofides' algorithm when matching odd vertices to create a graph where each vertex has even degree. A greedy approach was used here for time considerations. There's an extra step to slightly improve results by running the result of this weaker approximation of Christofides' algorithm through 2-opt, but the results should not end up beating the proper Christofides' algorithm in a head-to-head comparison.

This work was done in collaboration with Royce Hwang and Jessica Huang.

### structures
structures.py is a short file with some common data structures implemented in Python. From time to time I like to work on data structures to maintain my grasp of basic principles, and this file was written in 2021. Included structures are mostly structures I've known since my student days and written from memory with occasional references to educational materials to iron out possible mistakes. The Red-Black trees at the bottom are an exception, and based on Robert Sedgewick's discussion of [Left-Leaning Red-Black Trees](https://www.cs.princeton.edu/~rs/talks/LLRB/LLRB.pdf).
