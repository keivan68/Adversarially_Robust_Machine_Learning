# rosubgradient

We propose a discrete-time dynamical system-based approach to solving robust learning problems in a distributed manner. The robust learning problem is formulated as a robust optimization problem, and we introduce a novel discrete-time subgradient algorithm based on a saddle-point dynamical system for solving the (not necessarily smooth) robust optimization problem. One of the distinguishing features of the proposed subgradient algorithm is that it can be implemented in a distributed manner. This feature is exploited for the distributed implementation of the robust learning problem. Under the assumption that the cost function is convex and uncertainties enter concavely in the learning problem, we prove that the subgradient algorithm converges asymptotically to the robust optimal solution under diminishing step size. Furthermore, under the assumption of constant step-size, we show that the subgradient algorithm will converge to the arbitrary small neighborhood of robust optimal solution at a linear rate.
