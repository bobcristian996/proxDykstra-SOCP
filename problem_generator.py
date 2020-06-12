"""
This script can be used to generate conic problems. You must give to the
script the problem structure and where to save it and it will generate a
conic problem for you. You can also specify the problem's format (sparse |
dense).

The problem structure parameters have the following format:
{
    'sol_dim': [min[, max]], # The solution dimension
    'A_dim': [min[, max]], # The number of lines in A
    'Q_dim': [min[, max]], # The number of lines in Q matrices
    'Q_len': [min[, max]], # The number of conic constraints
    'type': 'sparse' | 'dense', # The type of the problem
    'density': 0 <= nb <= 1 # The density of the matrices if type == sparse
}

The list items represent a range (min and max). If only a element is found in
a list, then it will constrain the generator to use that exact number.
"""
import os
import sys
import time
import argparse
import logging
import pprint as p
import json
from enum import Enum
from random import randint, random
import scipy.sparse as ss
import numpy as np
import matrix_ops as mo
from tqdm import trange

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

DEF_PROBLEM_PARAMS = {
    'sol_dim': [10],
    'A_dim': [10],
    'Q_dim': [4],
    'Q_len': [10],
    'type': 'sparse',
    'density': 0.4
}

class ProblemType(Enum):
    """
    This enum represents the Problem Type.
    """
    SPARSE = 'sparse'
    DENSE = 'dense'

def generate_problem(params):
    """
    This function generates the matrices that represents the conic problem.

    The matrices are returned in the defined format.

    Args:
        params:     A dict that contains the problem specifications. It must
                    have the format of DEF_PROBLEM_PARAMS

    Returns:
        A tuple containing the generated matrices

    Example:
        >>> generate_problem(params)
        (b, c, Q_list)
    """
    # Randomize the parameters except for the Q_dim
    rand_params = dict(params)
    for key, value in rand_params.items():
        if key != 'Q_dim' and isinstance(value, list):
            rand_params[key] = randint(*value)

    _logger.info('Auto generating matrices with params:')
    p.pprint(rand_params)

    problem_type = ProblemType(rand_params['type'])

    # Compute the solution, c and A matrices
    if problem_type == ProblemType.DENSE:
        sol = np.random.rand(rand_params['sol_dim'], 1)
        c = np.random.rand(1, rand_params['sol_dim'])

        A = np.random.rand(rand_params['A_dim'], rand_params['sol_dim'])
        Q_creator_fcn = mo.create_dense_Q
    else:
        sol = ss.rand(rand_params['sol_dim'], 1, density=rand_params['density'],
                      format='csr')
        c = ss.rand(1, rand_params['sol_dim'], density=rand_params['density'],
                    format='csc')
        A = ss.rand(rand_params['A_dim'], rand_params['sol_dim'],
                    density=rand_params['density'], format='csr')
        Q_creator_fcn = mo.create_sparse_Q

    # Compute the linear constraints
    b = A.dot(sol)

    # Put the linear constraint A matrix into the defined format
    Q_list = [Q_creator_fcn(A)]

    del A

    # Generate conic constraints and put them into the right format
    for _ in trange(rand_params['Q_len'], mininterval=3):
        dim = randint(*rand_params['Q_dim'])

        if problem_type == ProblemType.DENSE:
            Q = np.random.rand(dim, rand_params['sol_dim'])
            q = np.random.rand(dim, 1)
            f = np.random.rand(1, rand_params['sol_dim'])
            # f = np.zeros((1, rand_params['sol_dim']))
            d = np.linalg.norm(Q.dot(sol) + q, 2) - f.dot(sol) + random()
        else:
            Q = ss.rand(dim, rand_params['sol_dim'],
                        density=rand_params['density'], format='csr')
            q = ss.rand(dim, 1, density=rand_params['density'], format='csr')
            f = ss.rand(1, rand_params['sol_dim'],
                        density=rand_params['density'], format='csc')

            print(f.dot(sol).data)
            print(mo.sparse_vec_norm2(Q.dot(sol) + q, 2))
            aux_var = f.dot(sol).data
            if aux_var:
                d = mo.sparse_vec_norm2(Q.dot(sol) + q, 2) - aux_var[0] + random()
            else:
                d = mo.sparse_vec_norm2(Q.dot(sol) + q, 2) - + random()

        Q_list.append(Q_creator_fcn(Q, q, f, d))

    del Q, q, f, d

    # Put b and c vectors into the defined format
    if problem_type == ProblemType.SPARSE:
        b = ss.vstack([b, ss.csr_matrix([[1]])], format='csr')
        c = ss.hstack([c, ss.csr_matrix([[0]])], format='csc')
    else:
        b = np.vstack((b, np.array([[1]])))
        c = np.hstack((c, np.array([[0]])))

    return sol, b, c, Q_list

def save_matrices(sol, b, c, Q_list, save_dir, problem_type):
    """
    This function saves the solution, b, c and Q matrices into the save_dir
    according to the problem format.

    Args:
        sol:            The solution vector
        b:              The b vector
        c:              The c vector
        Q_list:         The list of constraints
        save_dir:       The path to the directory where to save the prolem
        problem_type:   ProblemType
    """
    # Set the save function according to the problem_type
    save_fcn = ss.save_npz if problem_type == ProblemType.SPARSE else \
        np.savetxt
    # Set the extension according to the problem_type
    extension = '.npz' if problem_type == ProblemType.SPARSE else '.npy'

    if not os.path.isdir(save_dir):
        _logger.info(f"Creating folder: {save_dir} ...")
        os.system(f'mkdir {save_dir}')

    _logger.info("Saving the matrices ...")

    sol_file = os.path.join(save_dir, 'sol' + extension)
    _logger.info(f"Saving sol vector in: {sol_file} ...")
    save_fcn(sol_file, sol)

    b_file = os.path.join(save_dir, 'b' + extension)
    _logger.info(f"Saving b vector in: {b_file} ...")
    save_fcn(b_file, b)

    c_file = os.path.join(save_dir, 'c' + extension)
    _logger.info(f"Saving c vector in: {c_file} ...")
    save_fcn(c_file, c)

    _logger.info(f"Saving constraints into {save_dir}/Q_<idx>.npz ...")
    for i in trange(len(Q_list), mininterval=3):
        Q_file = os.path.join(save_dir, f"Q_{i}" + extension)
        save_fcn(Q_file, Q_list[i])

    _logger.info(f"All matrices were seved successfuly into: {save_dir}!")

def generate_problem_name():
    """
    This function generates a problem name following the format:
        auto_problem_<year><month><day>_<hour>_<min>_<sec>

    Returns:
        A string that represents the generated name
    """
    datetime_format = time.strftime("%Y%m%d_%H_%M_%S")

    return f'auto_problem_{datetime_format}'

if __name__ == '__main__':
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s [%(module)s]: %(message)s',
        level=logging.INFO
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem_folder', required=True,
                        help='The folder in which to store the generated '
                        'problem.')
    parser.add_argument('--problem_name', required=False, default=None,
                        help='The name of the generated problem. This '
                        'represents the folder where the matrix will be '
                        'stored. Default: auto_problem.<timestamp>')
    parser.add_argument('--params', required=True,
                        help='The parameters used to generate the matrix. This '
                        'is the path to a JSON file that needs to have the '
                        'following format: '
                        f'{p.pformat(DEF_PROBLEM_PARAMS, indent=2)}')
    args = parser.parse_args()

    if not os.path.isdir(args.problem_folder):
        _logger.error("The provided problem folder does not exist: "
                      f"{args.problem_folder}!")
        sys.exit(1)

    if not os.path.isfile(args.params):
        _logger.error("The provided params folder does not exist: "
                      f"{args.params}!")
        sys.exit(1)

    with open(args.params, 'r') as f:
        params = json.load(f)

    # Update the given parameters
    for key, value in params.items():
        if isinstance(value, list):
            if not value:
                _logger.error(f"The list for {key} cannot be empty!")
                sys.exit(1)
            elif len(value) > 2:
                _logger.error(f"The list assigned for {key} cannot have more "
                              "than 2 elemets!")
                sys.exit(1)
            elif len(value) == 2 and value[0] > value[1]:
                _logger.error(f"The first value for {key} list cannot be "
                              "greater than the second value!")
                sys.exit(1)
            elif len(value) == 1:
                value *= 2

            if any(not isinstance(elem, int) for elem in value):
                _logger.error(f"The elemet(s) for {key} list must be integers.")
                sys.exit(1)

    if args.problem_name is None:
        problem_name = generate_problem_name()
    else:
        problem_name = args.problem_name

    _logger.info(f"Generating the problem: {problem_name}")

    sol, b, c, Q_list = generate_problem(params)

    save_dir = os.path.join(args.problem_folder, problem_name)

    save_matrices(sol, b, c, Q_list, save_dir, ProblemType(params['type']))
