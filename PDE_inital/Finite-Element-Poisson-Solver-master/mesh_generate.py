## Generate mesh grid for solving 2D poisson with mixed boundary condition
## using FEM

import numpy as np


def Rectangle_gen(n):
    # Pick any integer greater than one. The total number of rectangles will be n^2
    h = 1 / float(n)  # Lattice constant (well up to scaling by root 2)

    verticies = []
    rectangle = []



    for i in range(0, n + 1):
        for j in range(0, n + 1):
            verticies.append([h * i, h * j])
    verticies = np.array(verticies)
    # print verticies

    # Create the triangulation
    for i in range(1, n+1):
        for j in range(1, n+1):
            rectangle.append([verticies[(n+1)*i+j], verticies[(n+1)*i+j-1], \
                            verticies[(n+1)*i+j-n-1], verticies[(n+1)*i+j-n-2]])

    # This tells us if the k-th vertex is on the boundary
    # for i in verticies:
    #     if i[0] == 1 or i[1] == 1 or i[0] == 0 or i[1] == 0:
    #         boundary.append(1)
    #     else:
    #         boundary.append(0)

    # boundary = np.array(boundary)
    rectangle = np.array(rectangle)
    # print len(triangulation)

    return [rectangle, verticies]

    # print triangulation


def Triangle_gen(n):
    # Pick any integer greater than one. The total number of triangles will be 2n^2
    h = 1 / float(n)  # Lattice constant (well up to scaling by root 2)

    verticies = []
    triangulation = []


    for i in range(0,n+1):
        for j in range(0,n+1):
            verticies.append([h*j,h*i])
    verticies  = np.array(verticies)

    # Create the triangulation
    for i in range(0, n):
        for j in range(0, n):
            triangulation.append(
                [verticies[(n + 1) * i + j], verticies[(n + 1) * i + j + 1], verticies[(n + 1) * i + j + n + 1]])
            triangulation.append([verticies[(n + 1) * i + j + 1], verticies[(n + 1) * i + j + n + 2],
                                  verticies[(n + 1) * i + j + n + 1]])

    triangulation = np.array(triangulation)

    return [triangulation, verticies]