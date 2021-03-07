import numpy as np
from mesh_generate import Rectangle_gen
# import compute_generic_basis_quad_2d_lagrange as gen
import generic_basis as gen
from scipy.integrate import dblquad,quad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import log
import matplotlib.patches as mpatches

fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

elltwo_error_box = []
energy_error_box = []
H1_error_box = []
sup_error_box = []
node_spacing_box = []

# ## define f, g, and real solution
def forcing_function(x,y):
    return -1

def beta(x,y):
    return 4


def forcing_grad_function(y):
    return -y/2 + 1/4

def u0_function(y):
    return y*(y-1)/4


def forcing_bound_function(x,y):
    if y == 0 or y == 1:
        return (x-0.5)**2
    elif x == 1:
        return 1/4
    else:
        return 0

def analytic_soln(x,y):
    return ((x - 0.5)**2 + (y - 0.5)**2)/4 - 1/8

def analytic_grad(x,y):
    return [x/2 - 1/4, y/2 - 1/4]


## from 2 to 21 or just 26
for n in range(26, 27):
    print('iter: ', n)
    lagrange_points = []

    # Initialize the mass and stiffness matrix
    # the mass function is used to calculate the right hand side
    mass_matrix = []
    mass_grad_matrix = []
    stiffness_matrix = []
    bound_matrix = []

    for i in range(0, n ** 2 + 2 * n + 1):
        row = []
        for j in range(0, n ** 2 + 2 * n + 1):
            row.append(0)
        mass_matrix.append(row)
        mass_grad_matrix.append(row)
        stiffness_matrix.append(row)
        bound_matrix.append(row)

    mass_matrix = np.array(mass_matrix).tolist()
    mass_grad_matrix = np.array(mass_grad_matrix).tolist()
    stiffness_matrix = np.array(stiffness_matrix).tolist()
    bound_matrix = np.array(bound_matrix).tolist()

    # print stiffness_matrix

    for rectan in Rectangle_gen(n)[0]:

        # Compute the linear mapping part of the affine mapping taking our generic triangle to the informed triangle
        jacobian = np.array([[rectan[0][0] - rectan[2][0], 0],
                             [0, rectan[0][1] - rectan[1][1]]])
        # print jacobian

        # Compute where the lagrange test points are sent by the affine mapping

        for test_points in gen.lagrange_test_rec:
            lagrange_points.append(np.dot(jacobian, np.array(test_points)) + np.array(rectan[3]))

    # lagrange_points = np.array(lagrange_points)
    actual_lagrange_points = list(map(tuple, lagrange_points))
    lagrange_points = np.array(list(set(actual_lagrange_points)))

    print('node points counted.')

    # print lagrange_points

    # This tells us if the k-th lagrange point is on the boundary

    lagrange_boundary = []
    for i in lagrange_points:
        if i[0] == 0:
            lagrange_boundary.append(1)
        elif i[0] == 1 or i[1] == 1 or i[1] == 0:
            lagrange_boundary.append(2)
        else:
            lagrange_boundary.append(0)

    # Now we begin the assembly procedure

    Rect = Rectangle_gen(n)[0]
    H1_matrix_x = np.zeros((len(Rect), len(lagrange_points.tolist())))
    H1_matrix_y = np.zeros((len(Rect), len(lagrange_points.tolist())))
    real_grad = []
    count = -1

    for rectan in Rectangle_gen(n)[0]:
        count += 1
        real_grad.append(analytic_grad((rectan[0][0] + rectan[2][0]) / 2, (rectan[0][1] + rectan[1][1]) / 2))

        # Compute the linear mapping part of the affine mapping taking our generic triangle to the informed triangle
        jacobian = np.array([[rectan[0][0] - rectan[2][0], 0],
                             [0, rectan[0][1] - rectan[1][1]]])
        # print jacobian

        # Compute where the lagrange test points are sent by the affine mapping

        transformed_test_points = []
        for test_points in gen.lagrange_test_rec:
            transformed_test_points.append(np.dot(jacobian, np.array(test_points)) + np.array(rectan[3]))

        # determine determinant of the Jacobian of the affine mapping
        scalar_jacobian = abs(np.linalg.det(jacobian))
        scalar_jacobian_bound = abs(rectan[0][0] - rectan[2][0])

        # Invert the jacobian of the affine mapping
        inverse_jacobian = np.linalg.inv(jacobian)

        # # Compute the Mass submatrix
        mass_submatrix = []
        for i in range(0, 4):
            row = []
            for j in range(0, 4):
                row.append(dblquad(lambda x, y: gen.shape_rec(x, y, i) * gen.shape_rec(x, y, j) * scalar_jacobian, 0, 1, \
                                   lambda x: 0, lambda x: 1)[0])
            mass_submatrix.append(row)

        mass_submatrix = np.array(mass_submatrix)
        # print mass_submatrix

        for i in range(0, 4):
            lagrange_points = lagrange_points.tolist()
            temp_i = (np.dot(jacobian, np.array(gen.lagrange_test_rec[i])) + np.array(rectan[3])).tolist()
            H1_matrix_x[count, lagrange_points.index(temp_i)] = np.dot(gen.grad_rec(0.5, 0.5, i), inverse_jacobian)[0]
            H1_matrix_y[count, lagrange_points.index(temp_i)] = np.dot(gen.grad_rec(0.5, 0.5, i), inverse_jacobian)[1]
            lagrange_points = np.array(lagrange_points)
            for j in range(0, 4):
                # lagrange_points = list(set(map(tuple,lagrange_points)))
                lagrange_points = lagrange_points.tolist()
                temp_j = (np.dot(jacobian, np.array(gen.lagrange_test_rec[j])) + np.array(rectan[3])).tolist()
                # print lagrange_points.index(temp_j)
                if lagrange_boundary[lagrange_points.index(temp_i)] != 1 and lagrange_boundary[
                    lagrange_points.index(temp_j)] != 1:
                    ind_i = lagrange_points.index(temp_i)
                    ind_j = lagrange_points.index(temp_j)
                    # print(mass_matrix[ind_i][ind_j], mass_submatrix[i][j])
                    mass_matrix[ind_i][ind_j] = mass_matrix[ind_i][ind_j] + mass_submatrix[i][j]
                    # print(mass_matrix[ind_i][ind_j], mass_submatrix[i][j])
                    # mass_grad_matrix[lagrange_points.index(temp_i)][lagrange_points.index(temp_j)] += mass_grad_submatrix[i][j]
                    # stiffness_matrix[lagrange_points.index(temp_i)][lagrange_points.index(temp_j)] += stiffness_submatrix[i][j]
                lagrange_points = np.array(lagrange_points)

        mass_matrix = np.array(mass_matrix)
        # print mass_matrix

        # Compute the Mass grad submatrix
        mass_grad_submatrix = []
        for i in range(0, 4):
            row = []
            for j in range(0, 4):
                row.append(
                    dblquad(lambda x, y: gen.shape_rec(x, y, i) * np.dot(gen.grad_rec(x, y, j), inverse_jacobian)[1] \
                                         * scalar_jacobian, 0, 1, lambda x: 0, lambda x: 1)[0])
            mass_grad_submatrix.append(row)

        mass_grad_submatrix = np.array(mass_grad_submatrix)
        # print mass_submatrix

        for i in range(0, 4):
            for j in range(0, 4):
                # lagrange_points = list(set(map(tuple,lagrange_points)))
                lagrange_points = lagrange_points.tolist()
                temp_i = (np.dot(jacobian, np.array(gen.lagrange_test_rec[i])) + np.array(rectan[3])).tolist()
                temp_j = (np.dot(jacobian, np.array(gen.lagrange_test_rec[j])) + np.array(rectan[3])).tolist()
                # print lagrange_points.index(temp_j)
                if lagrange_boundary[lagrange_points.index(temp_i)] != 1 and lagrange_boundary[
                    lagrange_points.index(temp_j)] != 1:
                    mass_grad_matrix[lagrange_points.index(temp_i)][lagrange_points.index(temp_j)] = \
                        mass_grad_matrix[lagrange_points.index(temp_i)][lagrange_points.index(temp_j)] + \
                        mass_grad_submatrix[i][j]
                    # stiffness_matrix[lagrange_points.index(temp_i)][lagrange_points.index(temp_j)] += stiffness_submatrix[i][j]
                lagrange_points = np.array(lagrange_points)

        mass_grad_matrix = np.array(mass_grad_matrix)

        # Compute the Stiffness submatrix
        stiffness_submatrix = []
        for i in range(0, 4):
            row = []
            for j in range(0, 4):
                row.append(dblquad(lambda x, y: np.dot(np.dot(gen.grad_rec(x, y, i), inverse_jacobian),
                                                       np.dot(gen.grad_rec(x, y, j), inverse_jacobian)) \
                                                * scalar_jacobian, 0, 1, lambda x: 0, lambda x: 1)[0])
            stiffness_submatrix.append(row)

        stiffness_submatrix = np.array(stiffness_submatrix)
        # print stiffness_submatrix

        for i in range(0, 4):
            for j in range(0, 4):
                # lagrange_points = list(set(map(tuple,lagrange_points)))
                lagrange_points = lagrange_points.tolist()
                temp_i = (np.dot(jacobian, np.array(gen.lagrange_test_rec[i])) + np.array(rectan[3])).tolist()
                temp_j = (np.dot(jacobian, np.array(gen.lagrange_test_rec[j])) + np.array(rectan[3])).tolist()
                # print lagrange_points.index(temp_j)
                if lagrange_boundary[lagrange_points.index(temp_i)] != 1 and lagrange_boundary[
                    lagrange_points.index(temp_j)] != 1:
                    stiffness_matrix[lagrange_points.index(temp_i)][lagrange_points.index(temp_j)] += \
                    stiffness_submatrix[i][j]
                if lagrange_boundary[lagrange_points.index(temp_i)] == 2 and lagrange_boundary[
                    lagrange_points.index(temp_j)] == 2:
                    if temp_i[1] == 1 and temp_j[1] == 1:
                        bound_matrix[lagrange_points.index(temp_i)][lagrange_points.index(temp_j)] += \
                            quad(lambda x: gen.shape_rec(x, 1, i) * gen.shape_rec(x, 1, j) * scalar_jacobian_bound, 0,
                                 1)[0]
                        stiffness_matrix[lagrange_points.index(temp_i)][lagrange_points.index(temp_j)] += \
                            4 * \
                            quad(lambda x: gen.shape_rec(x, 1, i) * gen.shape_rec(x, 1, j) * scalar_jacobian_bound, 0,
                                 1)[0]
                        # dblquad(lambda x, y: gen.shape_rec(x, y, i) * gen.shape_rec(x, y, j) * scalar_jacobian, 0, 1,
                        #         lambda x: 0, lambda x: 1)[0]
                    if temp_i[1] == 0 and temp_j[1] == 0:
                        bound_matrix[lagrange_points.index(temp_i)][lagrange_points.index(temp_j)] += \
                            quad(lambda x: gen.shape_rec(x, 0, i) * gen.shape_rec(x, 0, j) * scalar_jacobian_bound, 0,
                                 1)[0]
                        stiffness_matrix[lagrange_points.index(temp_i)][lagrange_points.index(temp_j)] += \
                            4 * \
                            quad(lambda x: gen.shape_rec(x, 0, i) * gen.shape_rec(x, 0, j) * scalar_jacobian_bound, 0,
                                 1)[0]
                    if temp_i[0] == 1 and temp_j[0] == 1:
                        bound_matrix[lagrange_points.index(temp_i)][lagrange_points.index(temp_j)] += \
                            quad(lambda y: gen.shape_rec(1, y, i) * gen.shape_rec(1, y, j) * scalar_jacobian_bound, 0,
                                 1)[0]
                        stiffness_matrix[lagrange_points.index(temp_i)][lagrange_points.index(temp_j)] += \
                            4 * \
                            quad(lambda y: gen.shape_rec(1, y, i) * gen.shape_rec(1, y, j) * scalar_jacobian_bound, 0,
                                 1)[0]
                lagrange_points = np.array(lagrange_points)
        # print mass_matrix
        stiffness_matrix = np.array(stiffness_matrix)
        bound_matrix = np.array(bound_matrix)

    print('mass and stiffness matrix assembled.')

    # We need to remove the rows and columns of the stiffness matrix which are all zero (these correspond to boundary nodes)
    print(stiffness_matrix.shape)
    print(mass_matrix.shape)
    blank_row = []
    for i in range(0, n ** 2 + 2 * n + 1):
        blank_row.append(0)

    temp_stiffness_matrix = []
    actual_stiffness_matrix = []
    temp_mass_matrix = []
    actual_mass_matrix = []
    temp_mass_grad_matrix = []
    actual_mass_grad_matrix = []
    temp_bound_matrix = []
    actual_bound_matrix = []

    for i in range(0, n ** 2 + 2 * n + 1):
        if not (np.array_equal(stiffness_matrix[i], blank_row)):
            temp_stiffness_matrix.append(stiffness_matrix[i])
            temp_mass_matrix.append(mass_matrix[i])
            temp_mass_grad_matrix.append(mass_grad_matrix[i])
            temp_bound_matrix.append(bound_matrix[i])

    temp_stiffness_matrix = np.transpose(np.array(temp_stiffness_matrix))
    temp_mass_matrix = np.transpose(np.array(temp_mass_matrix))
    temp_mass_grad_matrix = np.transpose(np.array(temp_mass_grad_matrix))
    temp_bound_matrix = np.transpose(np.array(temp_bound_matrix))

    print(temp_stiffness_matrix.shape)
    print(temp_mass_matrix.shape)

    blank_row = []
    for j in range(0, len(temp_stiffness_matrix[0])):
        blank_row.append(0)

    for j in range(0, n ** 2 + 2 * n + 1):
        # print temp_stiffness_matrix[:,j]
        if not (np.array_equal(temp_stiffness_matrix[j], blank_row)):
            actual_stiffness_matrix.append(temp_stiffness_matrix[j])
            actual_mass_matrix.append(temp_mass_matrix[j])
            actual_mass_grad_matrix.append(temp_mass_grad_matrix[j])
            actual_bound_matrix.append(temp_bound_matrix[j])

    actual_mass_matrix = np.array(actual_mass_matrix)
    actual_mass_grad_matrix = np.array(actual_mass_grad_matrix)
    actual_stiffness_matrix = np.array(actual_stiffness_matrix)
    actual_bound_matrix = np.array(actual_bound_matrix)

    print(actual_stiffness_matrix.shape)
    print(actual_mass_matrix.shape)

    print('mass and stiffness matrix processsed.')

    # Now we solve the linear algebra system Ku = Mf

    # Initialize the forcing term f
    forcing = []
    forcing_grad = []
    u0 = []
    forcing_bound = []
    nonzero_lagrange_points = []
    for i in range(0, len(lagrange_points)):
        u0.append(u0_function(lagrange_points[i][1]))
        if lagrange_boundary[i] == 0:
            forcing.append(forcing_function(lagrange_points[i][0], lagrange_points[i][1]))
            forcing_grad.append(forcing_grad_function(lagrange_points[i][1]))
            nonzero_lagrange_points.append(lagrange_points[i])
            forcing_bound.append(0)
        if lagrange_boundary[i] == 2:
            forcing.append(forcing_function(lagrange_points[i][0], lagrange_points[i][1]))
            forcing_grad.append(forcing_grad_function(lagrange_points[i][1]))
            nonzero_lagrange_points.append(lagrange_points[i])
            forcing_bound.append(forcing_bound_function(lagrange_points[i][0], lagrange_points[i][1]))
    lagrange_points = np.array(lagrange_points)
    nonzero_lagrange_points = np.array(nonzero_lagrange_points)

    # print actual_stiffness_matrix

    # Solve for u
    solution = np.linalg.solve(actual_stiffness_matrix, np.dot(actual_mass_matrix, forcing) + \
                               np.dot(actual_mass_grad_matrix, forcing_grad) + np.dot(actual_bound_matrix,
                                                                                      forcing_bound))

    # solution = np.linalg.solve(actual_stiffness_matrix, np.dot(actual_mass_matrix, forcing) + \
    #                             np.dot(actual_bound_matrix, forcing_bound))
    # print solution

    full_solution = []
    very_temp = 0
    for i in range(0, len(lagrange_points)):
        if lagrange_boundary[i] == 1:
            full_solution.append(u0[i])
        else:
            full_solution.append(solution[very_temp] + u0[i])
            very_temp = very_temp + 1
    full_solution = np.array(full_solution)
    del very_temp

    elltwo_error = 0

    for i in range(0, len(lagrange_points)):
        elltwo_error += ((analytic_soln(lagrange_points[i][0], lagrange_points[i][1]) - full_solution[i]) ** 2 / float(
            n ** 2))
    elltwo_error = elltwo_error ** 0.5
    elltwo_error_box.append(elltwo_error)

    # Sup Error
    sup_bowl = []
    sup_error = 0
    for i in range(0, len(nonzero_lagrange_points)):
        sup_bowl.append(abs(analytic_soln(nonzero_lagrange_points[i][0], nonzero_lagrange_points[i][1]) - solution[i]))
    sup_error = max(sup_bowl)
    sup_error_box.append(sup_error)

    # H1 error
    h1_error = 0
    real_grad = np.array(real_grad)
    h1_error += np.sum((real_grad[:, 0] - np.dot(H1_matrix_x, full_solution)) ** 2) / float(n ** 2)
    h1_error += np.sum((real_grad[:, 1] - np.dot(H1_matrix_y, full_solution)) ** 2) / float(n ** 2)
    h1_error = h1_error ** 0.5
    H1_error_box.append(h1_error)
    print('h1 error: {:.6f}, ell2 error: {:.6f}, sup error: {:.6f}'.format(h1_error, elltwo_error, sup_error))

# np.save('rect1_ell2', np.array(elltwo_error_box))
# np.save('rect1_sup', np.array(sup_error_box))
# np.save('rect1_h1', np.array(H1_error_box))



# ## plot the surface plot
X = np.arange(0, 1.00001, 1/n)
Y = np.arange(0, 1.00001, 1/n)
X_2D, Y_2D = np.meshgrid(X, Y)
Z = np.zeros(X_2D.shape)

lagrange_points = lagrange_points.tolist()
X = X.tolist()
Y = Y.tolist()
count = 0
print(len(lagrange_points))
for node in lagrange_points:
    i_ind = X.index(node[0])
    j_ind = Y.index(node[1])
    Z[i_ind, j_ind] = full_solution[count]
    count += 1

ax = Axes3D(fig)

ax.plot_surface(X_2D, Y_2D, Z, rstride=1, cstride=1, cmap = plt.get_cmap('rainbow'))
# ax.contourf(X_2D, Y_2D, Z, zdir='z', cmap='rainbow')
# ax.set_zlim(-1,0)

plt.show()