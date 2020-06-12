"""
This script can be used to run multiple generated or CBF problems. You have
to spacify the folder where the problem are stored. You must keep in mind that
inside the problem folder the files are considered to be CBF problems and
the directories are considered generated problems (see problem_generator.py).

This script also generates a HTML report that allows you to better analyze the
results of the run algorithms. Also, it allows you to keep track of the
progress made in optimizing some problems.
"""
import argparse
import logging
import os
import sys
import time
import json
import pprint as p
from conic_problem import ConicProblem
from result_analyzer import create_html_report

logging.basicConfig(
    format='[%(asctime)s] %(levelname)s [%(module)s]: %(message)s',
    level=logging.INFO
)

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

# The used HTML template
HTML_TPL = "report.tpl"

def create_test_dir(wd):
    """
    This function creates a test directory with respect to the following
    format: tests/test_<year><month><day>_<hour>_<min>_<sec>

    Args:
        wd:     The path to the base directory where the test dir is going to
                be created.

    Returns:
        A string that represents the path to the created test directory.
    """
    datetime_format = time.strftime("%Y%m%d_%H_%M_%S")

    base_test_dir = os.path.join(wd, "tests")
    test_dir = os.path.join(base_test_dir, 'test_' + datetime_format)

    os.system(f"mkdir {base_test_dir}")
    os.system(f"mkdir {test_dir}")

    print(f"Regression directory set to: {test_dir}")

    return test_dir

def problem_extractor(problem_path):
    """
    This function looks under the given 'problem_path' and extracts the
    Conic Problem inside of it.

    If the 'problem_path' is the path to a dir, then, according to the
    files extensions, '.npy' or '.npz', it creates a 'dense' or, respecively,
    a 'sparse' problem.

    If the 'problem_path' is the path to a file, then the file is considered to
    be a CBF format problem and it has to end in '.cbf.gz'.

    Args:
        problem_path:   The path to the problem.

    Returns:
        The extracted ConicProblem.
    """

    if os.path.isdir(problem_path):
        # Get all the files inside problem_path
        *_, files = next(os.walk(problem_path))
        if files[0].endswith('.npz'):
            problem_format = 'sparse'
        elif files[0].endswith('.npy'):
            problem_format = 'dense'
        else:
            _logger.error("Unsupported problem type! The files inside the "
                          f"{problem_path} must end in either '.npz' or '.npy'")
            sys.exit(1)

        # The problem name is the folder's name
        problem_name = problem_path.split(os.path.sep)[-1]
        _logger.info(f"Loading problem {problem_name} in {problem_format} "
                     "format ...")
        return ConicProblem(np_dir=problem_path, format=problem_format,
                            name=problem_name)
    else:
        if not problem_path.endswith('.cbf.gz'):
            _logger.error(f"The file {problem_math} is considered a CBF file. "
                         "It must end in '.cbf.gz'.")
            sys.exit(1)

        # The problem name is the file name
        problem_name = problem_path.split(os.path.sep)[-1].replace('.cbf.gz',
                                                                   '')
        _logger.info(f"Loading problem {problem_name} in sparse format...")
        return ConicProblem(cbf_file=problem_path, name=problem_name)

def get_problems(problems_dir):
    """
    This function searches for problems under the given 'problems_dir'.

    This function returns a generator.

    Args:
        problems_dir:   The path to the problem directory.

    Returns:
        A problem generator.
    """
    _logger.info(f"Searching for problems under {problems_dir} ...")
    root, dirs, files = next(os.walk(problems_dir))

    if not dirs and not files:
        _logger.error(f"The {root} directory is empty!")
        sys.exit(1)

    if dirs:
        _logger.info("Detected numpy (scipy) generated problems under "
                     f"{root} folder")
        for dir in dirs:
            yield problem_extractor(os.path.join(root, dir))

    if files:
        _logger.info(f"Detected CBF files under {root} folder")

        for file in files:
            yield problem_extractor(os.path.join(root, file))

    _logger.info("Problem set ended!")

def solve_problems(problems_dir, solvers, pp_params=None):
    """
    This function solves the problems under 'problems_dir' using the given
    'solvers'. If the 'pp' solver is one of the given 'solvers', then for each
    problem the pp_params are set.

    Args:
        problems_dir:   The path to the problems directory.
        solvers:        The list of solvers.
        pp_params:      The Proximal Point parameters. Default: None.

    Returns:
        A list of collected data for each of the found problems.
    """
    problems = get_problems(os.path.abspath(problems_dir))

    problems_results = []

    for prob in problems:
        problem_data = {
            'name': prob.name,
            'algorithms': []
        }

        if 'pp' in solvers:
            prob.set_pp_parameters(**pp_params)

        for solver_pref in solvers:
            algo = {'name': solver_pref}
            solver = getattr(prob, f"{solver_pref}_solver")

            algo.update(solver())
            problem_data['algorithms'].append(algo)

        del prob

        problems_results.append(problem_data)

    return problems_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problems_dir', required=True,
                        help='The path to the problems directory. This '
                        'directory can contain a single problem or multiple '
                        'problems, each problem stored in a different '
                        'directory.')
    parser.add_argument('--solvers', required=True, nargs='+',
                        choices=['pp', 'cvx'], help='This argument represents '
                        'the solvers that are going to be used for solving '
                        'the ConicProblem. You can use either CVX, '
                        'ProximalPoint  (pp) or both.')
    parser.add_argument('--pp_params', required=False, default=None,
                        help='This is the path to a JSON file that contains '
                        'the required parameters for the Proximal Point '
                        'solver. It is not required if only cvx is used! '
                        'Default: None.')
    parser.add_argument("--work_dir", required=False, default='.',
                        help="The path to the directory where to save the "
                        "results. Default: . (current directory)")
    args = parser.parse_args()

    if not os.path.isdir(args.problems_dir):
        _logger.error(f"The provided 'problems_dir': {args.problems_dir} does "
                      "not exist!")
        sys.exit(1)

    if not os.path.isdir(args.work_dir):
        _logger.error(f"The provided 'work_dir': {args.work_dir} does "
                      "not exist!")
        sys.exit(1)

    if 'pp' in args.solvers and args.pp_params is None:
        _logger.error("You have to provide Proximal Point parameters with "
                      "--pp_params if you want to solve the Conic Problem "
                      "with it!")
        sys.exit(1)
    elif 'pp' in args.solvers:
        with open(args.pp_params, 'r') as f:
            pp_params = json.load(f)
    else:
        pp_params = None

    results = solve_problems(args.problems_dir, args.solvers, pp_params)

    test_dir = create_test_dir(args.work_dir)

    create_html_report(HTML_TPL, test_dir, results, pp_params)
