# Python
A repository for Python scripts.

While I've worked on several school projects in Python, most of it is in algorithmic work rather than interactive software.

### TSP
For now, I've cleaned up the comments of a single script for display, the one I've most enjoyed working on. It's an approximation of Christofides' algorithm for approximating the Travelling Salesman Problem. The algorithm has a step that strongly deviates from Christofides' algorithm when matching odd vertices to create a graph where each vertex has even degree. A greedy approach was used here for time considerations. There's an extra step to slightly improve results by running the result of this weaker approximation of Christofides' algorithm through 2-opt, but the results should not end up beating the proper Christofides' algorithm in a head-to-head comparison.

This work was done in collaboration with Royce Hwang and Jessica Huang.
