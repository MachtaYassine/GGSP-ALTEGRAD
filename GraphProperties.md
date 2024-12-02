# Graph Properties
This to define and visualize the different Graph properties used in this challenge

## Definitions

### 1. Triangles
A **triangle** is a set of three nodes $ (u, v, w) $ such that all three nodes are mutually connected by edges:
- Edge $ u \to v $,
- Edge $ v \to w $,
- Edge $ w \to u $.

<!-- ### Computing the Number of Triangles

To compute the number of triangles in a graph:
1. For each node $ u $, identify its **neighbors** (nodes directly connected to $ u $).
2. Check for every pair of neighbors $ v $ and $ w $ if there exists an edge between $ v $ and $ w $.
3. Count such sets $ (u, v, w) $.
#### Formula:
Let $ A $ be the adjacency matrix of the graph, where $ A[i][j] = 1 $ if there is an edge between node $ i $ and $ j $, and $ 0 $ otherwise. Then:

$$
\text{Number of triangles} = \frac{\text{trace}(A^3)}{6}
$$ -->


### 2. Triplets
A **triplet** is a set of three nodes $ (u, v, w) $ where:
- $ u $ is connected to $ v $,
- $ v $ is connected to $ w $.

Triplets can be either:
- **Closed triplets**: Formed by a triangle where all three nodes are interconnected.
- **Open triplets**: Formed when only two edges are present, e.g., $ u \to v $ and $ v \to w $, but $ u $ and $ w $ are not connected.

### Computing the Number of Triplets

To compute the total number of triplets:
1. For each node $ u $, calculate the number of pairs of neighbors it has.
2. Each such pair forms a triplet centered at $ u $.

#### Formula:
For a node $ u $, let $ d_u $ be its degree (number of neighbors). The total number of triplets is given by:

$$
\text{Number of triplets} = \sum_{u \in V} \binom{d_u}{2}
$$

Where:
- $ \binom{d_u}{2} = \frac{d_u (d_u - 1)}{2} $, the number of ways to choose two neighbors from $ u $'s neighbors.
- $ V $ is the set of all nodes in the graph.

---
---

## Global Clustering Coefficient

Using the above, the global clustering coefficient $ C $ is computed as:

$$
C = \frac{3 \cdot \text{Number of triangles}}{\text{Number of triplets}}
$$

Where:
- **Number of triangles**: Computed using $ \frac{\text{trace}(A^3)}{6} $.
- **Number of triplets**: Computed using $ \sum_{u \in V} \binom{d_u}{2} $.

---
# Maximum K-Core of a Graph

The **maximum K-core** is the K-core with the largest value of $ K $ in a graph $ G $. It represents the most cohesive subgraph, where every node has a degree of at least $ K_{\text{max}} $.

# Efficient Computation of Maximum K-Core

To compute the maximum K-core efficiently, we use a **degree peeling algorithm**:

---

## Steps:
1. **Initialization**:
   - Compute the degree $ d(v) $ for all nodes $ v $ in the graph $ G $.
   - Create a data structure to maintain nodes sorted by degree.

2. **Peeling Process**:
   - Start with $ K = 0 $.
   - Remove all nodes with $ d(v) \leq K $ from the graph.
   - For each removed node, decrease the degree of its neighbors.
   - Increment $ K $ and repeat until no nodes remain.

3. **Output**:
   - The highest $ K $ value where a subgraph still exists is the maximum $ K_{\text{max}} $.
   - The corresponding subgraph is the maximum K-core.

---

## Complexity:
- **Time**: $ O(V + E) $, where $ V $ is the number of nodes and $ E $ is the number of edges.
- **Space**: $ O(V + E) $.

---

## Implementation Tip:
Use a min-heap or bucket-sort structure to efficiently track and update nodes based on their degrees during peeling.
