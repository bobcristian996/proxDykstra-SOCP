#    (C) Copyright 2020 Bob Cristian
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
import numpy as np
from CBFdata import CBFdata
from scipy.sparse import csc_matrix, rand, save_npz, load_npz, coo_matrix, csr_matrix, lil_matrix, vstack, hstack
import matrix_ops as mo
from datetime import datetime
from tqdm import trange
import logging
import re
import cvxpy as cp
from scipy.sparse.linalg import norm
import json
from random import randint
import scipy
import matplotlib.pyplot as plt
from math import sqrt, sin, pi, log

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class ConicProblem:
    def __init__(self, cbf_file=None, np_dir=None, name=None, format='sparse',
                 debug=False):
        """
        The initializer of a ConicProblem object. One of the 'cbf_file' or
        'np_dir' arguments is required. Also, you cannot give both arguments
        to the same problem.

        If a 'cbf_fie' is given, then the format is automatically set to
        'sparse'.

        Kwargs:
            cbf_file:   The path to a CBF format file. Default: None
            np_dir:     The path to a directory that contains numpy/scipy
                        generated matrices. Default: None
            name:       The name of the problem. Default: None
            format:     The format of the problem (sparse | dense).
                        Default: sparse
            debug:      Flag that enables debug for solvers

        Returns:
            ConicProblem object

        Raises:
            AttributeError
        """
        self.name = name

        if cbf_file is not None and np_dir is not None:
            raise AttributeError("You must specify only one of 'cbf_file' and "
                                 "'np_dir'!")

        self.cbf_file = cbf_file
        self.np_dir = np_dir
        self.debug = debug

        if format not in ['dense', 'sparse']:
            raise AttributeError("'format' must be in ['sparse', 'dense']")

        if cbf_file is not None:
            self.format = 'sparse'
            if self.format != format:
                _logger.info("Set the format to: sparse because 'cbf_file' "
                             "was provided!")
        else:
            self.format = format

        # Set the corresponding extract and create methods according to the
        # chosen format
        self.create_Q_block = getattr(mo, f"create_{self.format}_Q")
        self.extract_Q_blocks = getattr(mo, f"get_Q_{self.format}_blocks")

        # Set the corresponding file save and load methods according to the
        # chosen format
        self.file_loader = load_npz if self.format == 'sparse' else \
            np.loadtxt
        self.file_saver = save_npz if self.format == 'sparse' else \
            np.savetxt

        # Set the corresponding norm method methods according to the
        # chosen format
        self.norm2 = mo.sparse_vec_norm2 if self.format == 'sparse' else \
            np.linalg.norm

        # Set the corresponding copy matrix method according to the
        # chosen format
        self.copy_matrix = csr_matrix if self.format == 'sparse' else \
            np.array

        # Extract the problem
        if self.cbf_file is not None:
            self.__cbf_extractor()
        elif self.np_dir is not None:
            self.__file_extractor()
        else:
            raise AttributeError("At least one argument, 'cbf_file' "
                                 "or 'np_dir', must be given")

    def __cbf_extractor(self):
        """
        This method extracts a problem from the given CBF format file.

        Initially, the CBFData object is created and, afterwards, the extracted
        data is put into the supported problem definition format. If the CBF
        file contains other constraints than L= or Q, an exception is raised.

        Raises:
            Exception
        """

        # Extract only the points of interest
        keywords = set(["VAR", "ACOORD",
                        "BCOORD", "OBJACOORD", "CON", "OBJSENSE"])

        # Compute the CBFData object
        obj = next(CBFdata(self.cbf_file, keywords).iterator())

        self.objective = obj.obj
        if self.objective == 'MAX':
            raise Exception("Unsuported problem objective: Maximization!")

        # Extract the c vector and put it into the right format
        c_data = (obj.objaval, ([0] * obj.objannz, obj.objasubj))
        self.c = csc_matrix(c_data, shape=(1, obj.varnum))
        self.c = hstack([self.c, csc_matrix([[0]])], format='csc')

        del c_data
        del obj.objaval
        del obj.objasubj

        # Extract the constraints
        con_constraints = list(zip(obj.mapstackdomain, obj.mapstackdim))

        del obj.mapstackdim
        del obj.mapstackdomain

        # Create the aux sparse A matrix. This matrix contains information
        # regarding both conic and linear constraints
        aux_A_data = (obj.aval, (obj.asubi, obj.asubj))
        aux_A_mat = csr_matrix(aux_A_data,
                               shape=(obj.mapnum, obj.varnum))

        del aux_A_data
        del obj.aval
        del obj.asubi
        del obj.asubj

        # Create the aux sparse b vector. This matrix contains information
        # regarding both conic and linear constraints
        aux_b_data = (obj.bval, (obj.bsubi, [0] * obj.bnnz))
        aux_b_vec = csr_matrix(aux_b_data, shape=(obj.mapnum, 1))

        del aux_b_data
        del obj.bval
        del obj.bsubi

        # The list of conit constraints
        self.Q_list = []
        # The A matrix in the defined format
        A_mat = None
        # The b vector in the defined format
        b_vec = None

        for i in trange(len(con_constraints), mininterval=3):
            con_constraint = con_constraints[i]
            aux_dim = con_constraint[1]

            if con_constraint[0] == "Q":
                # Create the conic constraints in the defined format and append
                # them to the conic constraints list
                self.Q_list.append(self.create_Q_block(
                    Q_blk=aux_A_mat[1:aux_dim],
                    q=aux_b_vec[1:aux_dim],
                    f=aux_A_mat[0],
                    d=aux_b_vec[0, 0]))
            elif con_constraint[0] == "L=":
                # Create the linear constraints in the defined format and
                # add them to the defined A matrix and b vector using the
                # minimal dimension:
                # >>> A1.shape
                #   (4, 5)
                # >>> A2.shape
                #   (2, 5)
                # >>> A = A1 + vstack([A2, np.zeros(4 - 2, 5)])
                if A_mat is None and b_vec is None:
                    A_mat = csr_matrix(aux_A_mat[:aux_dim])
                    b_vec = csr_matrix(aux_b_vec[:aux_dim])
                else:
                    m = A_mat.shape[0]
                    aux_m = aux_dim

                    if m >= aux_m:
                        A_mat[:aux_m] += aux_A_mat[:aux_m]
                        b_vec[:aux_m] += aux_b_vec[:aux_m]
                    else:
                        aux_A_mat[:m] += A_mat
                        aux_b_vec[:m] += b_vec

                        A_mat = csr_matrix(aux_A_mat[:aux_m])
                        b_vec = csr_matrix(aux_b_vec[:aux_m])
            else:
                raise Exception(f"Unsuported constraint: {con_constraint[0]}")

            # Delete the unnecessary elements of aux_A_mat and aux_b_vec
            if aux_dim != aux_A_mat.shape[0]:
                aux_A_mat = aux_A_mat[aux_dim:]
                aux_b_vec = aux_b_vec[aux_dim:]

        del aux_A_mat
        del aux_b_vec
        del con_constraints

        # Create the Q_0 block and add it to the beginning of the Q_list
        self.Q_list.insert(0, self.create_Q_block(
            Q_blk=A_mat
        ))
        del A_mat

        # Put b vector into the defined format
        self.b = vstack([-b_vec, csr_matrix([[1]])], format='csr')
        del b_vec

        del obj

        _logger.info("Extraction completed successfuly!")

    def __file_extractor(self):
        """
        This method extracts a problem from the given 'np_dir' directory
        according to the set format.

        The given 'np_dir' must only contain files that match the following
        regex: (b.*|c.*|Q_\d+.*|sol.*).

        Raises:
            Exception
        """
        if not os.path.isdir(self.np_dir):
            raise Exception(f"The provided 'np_dir': {self.np_dir} was not "
                            "found!")

        _, wd, files = next(os.walk(self.np_dir))

        if wd:
            _logger.error("There are directories inside the provided 'np_dir'"
                          " directory:")
            print(wd)
            raise Exception("Unsupported directory structure!")

        del wd

        if not files:
            _logger.error("There are no files inside the provide 'np_dir': "
                          f"{self.np_dir}!")
            raise FileNotFoundError("No files inside directory")

        _logger.info("Checking for wrong file names (not b.* | c.* | "
                     "Q_<idx>.*|sol.) ...")

        re_pattern = r"(b.*|c.*|Q_\d+.*|sol.*)"
        b_found = False
        c_found = False
        Q_found = False
        sol_exists = False
        for i in trange(len(files)):
            file_name, _ = os.path.splitext(files[i])

            if not re.search(pattern=re_pattern, string=file_name):
                _logger.error(f"\nThe file: {files[i]} did not respect the "
                              f"imposed pattern: {re_pattern}!")
                raise Exception("Unsupported file name")

            if file_name == 'b':
                b_found = True
            elif file_name == 'c':
                c_found = True
            elif not Q_found and file_name.startswith("Q"):
                Q_found = True
            else:
                sol_exists = True

        # Check that the provided directory contains the needed matrices:
        # b, c and at least one Q
        if not all([b_found, c_found, Q_found]):
            _logger.error("There should be at least 1 'b', 'c' and 'Q_idx' "
                          f"files inside {self.np_dir}!")
            msg = []
            if not b_found:
                msg.append('b file')

            if not c_found:
                msg.append('c file')

            if not Q_found:
                msg.append('Q file')

            raise FileNotFoundError(' AND '.join(msg))

        del re_pattern
        del b_found
        del c_found
        del Q_found

        # Load the matrices into self fields
        self.Q_list = [None] * (len(files) - 2 - int(sol_exists))
        _logger.info("Extracting the matrices ...")
        for i in trange(len(files), mininterval=5):
            file_name, _ = os.path.splitext(files[i])
            file_path = os.path.join(self.np_dir, files[i])

            if file_name == 'b':
                self.b = self.file_loader(file_path)
            elif file_name == 'c':
                self.c = self.file_loader(file_path)
            elif file_name.startswith('Q_'):
                idx = int(file_name.split('_')[1])
                self.Q_list[idx] = self.file_loader(file_path)

        if self.format == 'dense':
            self.b = np.reshape(self.b, (self.b.shape[0], 1))
            self.c = np.reshape(self.c, (1, self.c.shape[0]))

        # Safety check
        if any(elem is None for elem in self.Q_list):
            _logger.error("Extraction failed! None was found in Q_list at "
                          f"position {self.Q_list.index(None)}")
            raise Exception("None found in list")

        _logger.info("Extraction completed successfuly!")

    def set_pp_parameters(self, **kwargs):
        """
        This method sets the proximal point parameters.

        This method must be invoked before running the self.pp_solver() method.

        The required Kwards are:

        Kwargs:
            eps_pp:         The error for Proximal Point algorithm
            eps_dj:         The error for Dykstra algorithm
            max_iter_pp:    The maximum number of iterations that Proximal
                            Point algorithm can make
            max_iter_dj:    The maximum number of iterations that Dykstra
                            algorithm can make
            gamm_0:         The penalization used by Proximal Point

        Raises:
            AttributeError
        """
        param_list = ['eps_pp', 'eps_dj', 'max_iter_pp', 'max_iter_dj',
                      'gamma_0']

        for param in param_list:
            if param not in kwargs:
                raise AttributeError(f"Parameter '{param}' can't be found in "
                                     "the provided arguments list")
            if kwargs[param] is None:
                _logger.warning(f"Not setting {param} because it is None!")
            else:
                setattr(self, param, kwargs[param])

    def save_mats(self, save_dir):
        """
        This method is used to save the b, c and Q_list matrices in the given
        directory.

        The provided 'save_dir' directory must exist! The matrices will be
        saved in '.npz' files if the format is set to 'sparse' and in '.npy'
        otherwise.

        Args:
            save_dir:   The path to the save directory

        Raises:
            Exception
        """
        if not os.path.isdir(save_dir):
            raise Exception(f"The provided directory: {save_dir} does not "
                            "exist!")

        extension = '.npz' if self.format == 'sparse' else '.npy'
        b_file = os.path.join(save_dir, 'b' + extension)
        _logger.info(f"Saving b vector in: {b_file} ...")
        self.file_saver(b_file, self.b)

        del b_file

        c_file = os.path.join(save_dir, 'c' + extension)
        _logger.info(f"Saving c vector in: {c_file} ...")
        self.file_saver(c_file, self.c)

        del c_file

        _logger.info(f"Saving constraints into {save_dir}/Q_<idx>.npz ...")
        for i in trange(len(self.Q_list), mininterval=5):
            Q_file = os.path.join(save_dir, f"Q_{i}" + extension)
            self.file_saver(Q_file, self.Q_list[i])

        _logger.info(f"All matrix were seved successfuly into: {save_dir}!")

    def cvx_solver(self):
        '''
        This method solves the defined problem using the CVX solver.

        Returns:
            A dict holding the solve info

        See:
            https://www.cvxpy.org/examples/basic/socp.html

        Example:
            >>> prob.cvx_solver()
                {
                    'sol': solution,
                    'f_val': min value,
                    'iters': number of iterations,
                    'cpu_time': time needed for solving
                }
        '''
        if self.name is not None:
            _logger.info(f"Solving problem {self.name} using CVX ...")
        # Get the A matrix
        A, _, _, _ = self.extract_Q_blocks(self.Q_list[0])
        m, n = A.shape

        # SOC primal solution
        # x = cp.Variable((n, 1))
        x = cp.Variable(n)

        if self.format == 'sparse':
            b = np.reshape(np.array(self.b[0:-1].todense()), m)
        else:
            b = np.reshape(self.b[:-1], m)

        # Setting the equality constraint
        # soc_eq_constr = [A @ x == self.b[0:-1]]
        soc_eq_constr = [A @ x == b]

        del A

        # Setting the SOC inequality constraints
        # https://buildmedia.readthedocs.org/media/pdf/cvxpy/latest/cvxpy.pdf
        soc_ineq_constr = []
        for Q_mat in self.Q_list[1:]:
            Q_blk, q, f, d = self.extract_Q_blocks(Q_mat)
            if self.format == 'sparse':
                f = np.reshape(np.array(f.todense()), n)
                q = np.reshape(np.array(q.todense()), Q_blk.shape[0])
            else:
                f = np.reshape(f, n)
                q = np.reshape(q, Q_blk.shape[0])

            soc_ineq_constr.append(
                cp.SOC(f @ x + d, Q_blk @ x + q)
            )

        del Q_blk
        del q
        del f
        del d

        # Define the whole problem
        prob = cp.Problem(cp.Minimize(self.c[0, 0:-1] @ x), constraints=(
            soc_eq_constr + soc_ineq_constr))

        # Solve the problem
        start_time = datetime.now()
        prob.solve(verbose=True)
        cpu_time = datetime.now() - start_time

        solution = x.value
        iters = prob.solver_stats.num_iters

        return {
            'sol': solution,
            'f_val': prob.value,
            'iters': iters,
            'cpu_time': cpu_time
        }

    def pp_solver(self):
        """
        This method solves the defined problem using the Proximal Point
        algorithm.

        Be aware that before running this method, you have to set the
        parameters for Proximal Point algorithm.

        Returns:
            A dict holding the solve info

        Example:
            >>> prob.pp_solver()
                {
                    'sol': solution,
                    'f_val': min value,
                    'iters': number of iterations,
                    'cpu_time': time needed for solving,
                    'iter_metadata': a list containing metadata used for
                                     debugging
                }
        """
        if self.name is not None:
            _logger.info(f"Solving problem {self.name} using "
                         "Proximal Point ...")

        # Init
        gamma = self.gamma_0
        _, n = self.Q_list[0].shape
        iter_pp = 0
        crt_pp = 1

        # Starting with Full-Zero Y
        # Y will be updated iteration by iteration
        Y = []
        for q in self.Q_list:
            if self.format == 'dense':
                Y.append(np.zeros((q.shape[0], 1)))
            else:
                Y.append(csr_matrix((q.shape[0], 1)))

        L = mo.get_L(self.Q_list)

        # Starting with Full-Zero x
        if self.format == 'dense':
            x = np.zeros((n, 1))
        else:
            x = csr_matrix((n, 1))

        f_val = self.c.dot(x)[0, 0]

        # Metadata list for each iteration used for debugging
        iter_metadata = []

        # Store the transpose of c vector for time-efficiency
        c_transpose = self.c.T

        crt_arr = []
        f_vals = []

        # Start measuring the runtime
        start_time = datetime.now()
        while iter_pp < self.max_iter_pp and crt_pp >= self.eps_pp:
            v = x - gamma * c_transpose

            # Call the Dykstra step
            x, Y, dj_crt_out, dj_cpu_time, dj_iter_metadata = \
                self.__dykstra_step(Y, L, v)

            # Compute the new minimum value
            f_val = self.c.dot(x)[0, 0]

            print("Dykstra epsilon:", dj_crt_out)
            print("Cost function:", f_val)
            # Check that for at least 3 consecutive steps the optimum value
            # doesen't change with more than self.eps_pp
            if len(crt_arr) < 3:
                crt_arr.append(f_val)
            else:
                aux_crt_arr = crt_arr[1:] + [f_val]
                crt_diffs = [abs(elem[1] - elem[0])
                             for elem in zip(crt_arr, aux_crt_arr)]

                if all(crt_dif < self.eps_pp for crt_dif in crt_diffs):
                    crt_pp = max(crt_diffs)

                crt_arr = list(aux_crt_arr)

            if self.debug:
                iter_metadata.append({
                    'iter': iter_pp,
                    'gamma': gamma,
                    'crt_out': crt_pp,
                    'f_val': f_val,
                    'dj_meta': {
                        'cpu_time': str(dj_cpu_time),
                        'crt_out': dj_crt_out,
                        'iter_metadata': dj_iter_metadata
                    }
                })

            # Update values
            iter_pp += 1
            # gamma = 10 ** (1/iter_pp)
            gamma = 2 ** (sqrt(iter_pp))
            # gamma = log(5 ** iter_pp, 2)
            print(gamma)

        # Compute the runtime
        cpu_time = datetime.now() - start_time

        return {
            'sol': x,
            'f_val': f_val,
            'iters': iter_pp,
            'cpu_time': cpu_time,
            'metadata': iter_metadata
        }

    def __dykstra_step(self, Y, L, v):
        """
        This method runs a Dykstra step.

        Args:
            Y:  The dual solution
            L:  The list of max eigenvalue for Q_idx.T * Q_idx
            v:  The x - gamma * c_transpose vector

        Returns:
            A tuple holding the solve info

        Example:
            >>> prob.__dykstra_step(Y, L, v)
              (x, Y, crt_dj (last_computed_criterion), cpu_time, iter_metadata)
        """
        # Init
        crt_dj = 1
        iter_dj = 0

        # Comput the rezidual: rez = sum(Q[i].T * Y[i]) - v
        rez = self.copy_matrix(-v)
        for i in range(len(Y)):
            rez += self.Q_list[i].T.dot(Y[i])

        # Metadata list for each iteration used for debugging
        iter_metadata = []

        # dim = m + 1
        dim = len(L)

        # Start measuring the runtime
        start_time = datetime.now()

        while iter_dj < self.max_iter_dj and crt_dj > self.eps_dj:
            # Select a random index of Q
            idx = randint(0, dim-1)
            delta = self.Q_list[idx].dot(rez)

            # Store the value for L[idx] for little time-efficiency
            # improovement
            eig_idx = L[idx]

            # Comput the step
            step = Y[idx] - 1 / eig_idx * delta

            # Compute the projection
            if idx == 0:
                prj = 1 / eig_idx * self.b
            else:
                prj = 1 / eig_idx * self.__projection(eig_idx * step)

            # Update the rezidual
            Y_new = step - prj
            rez += self.Q_list[idx].T.dot(Y_new - Y[idx])

            # Update Y
            Y[idx] = Y_new

            # Compute the crt_dj once a while because it is computational
            # expensive
            if iter_dj % dim == 0:
                aux_rez = -rez
                crt_1 = 0
                crt_2 = self.norm2(self.Q_list[0].dot(aux_rez) - self.b, ord=2)

                for i in range(1, len(self.Q_list)):
                    Q_blk, q, f, d = self.extract_Q_blocks(self.Q_list[i])
                    t1 = self.norm2(
                        Q_blk.dot(aux_rez[0:-1]) + q * aux_rez[-1, -1], ord=2)
                    t2 = f.dot(aux_rez[0:-1])[0, 0] + d * aux_rez[-1, -1]
                    crt_2 = max(crt_2, max(t1 - t2, 0))

                crt_dj = max(crt_1, crt_2)

            if self.debug:
                iter_metadata.append({
                    'iter': iter_dj,
                    'crt_out': crt_dj,
                    'idx': idx
                })

            iter_dj += 1

        # Create the primal solution
        x = self.copy_matrix(aux_rez)
        # Compute the runtime
        cpu_time = datetime.now() - start_time

        return x, Y, crt_dj, cpu_time, iter_metadata

    def __projection(self, vec):
        """
        This method computes the projection of a vector on a cone.

        Args:
            vec:    The vector

        Returns:
            A numpy or a sparse matrix
        """
        q = vec[0:-1]
        norm_q = self.norm2(q, ord=2)
        r = vec[-1, -1]

        if norm_q <= r:
            return self.copy_matrix(vec)
        elif norm_q <= -r:
            if self.format == 'dense':
                return np.zeros(vec.shape)
            else:
                return csr_matrix(vec.shape)
        else:
            if self.format == 'dense':
                to_return = np.zeros(vec.shape)
            else:
                to_return = lil_matrix(vec.shape)

            to_return[-1, -1] = 1
            to_return[0:-1] = q / norm_q

            to_return *= (norm_q + r) / 2
            if format == 'dense':
                return to_return
            else:
                return csr_matrix(to_return)
