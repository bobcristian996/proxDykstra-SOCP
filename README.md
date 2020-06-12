# proxDykstra-SOCP

This is the implementation of the *proxDykstra* algorithm in `Python 3.6`.

The *proxDykstra-SOCP* algorithm was designed by *I. Necoara, O. Ferqoc* and it meant to solve a *Second-Order Cone Program* optimization problem like:


<p style="text-align: center;"><a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\large&space;\begin{aligned}&space;\min_{x}&space;\quad&space;&&space;c^Tx\\&space;\textrm{s.t.}&space;\quad&space;&&space;Ax&space;=&space;b\\&space;&\left\Vert&space;Qx&space;&plus;&space;q&space;\right\Vert^2_2&space;\leq&space;fx&space;&plus;&space;d&space;\\&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\large&space;\begin{aligned}&space;\min_{x}&space;\quad&space;&&space;c^Tx\\&space;\textrm{s.t.}&space;\quad&space;&&space;Ax&space;=&space;b\\&space;&\left\Vert&space;Qx&space;&plus;&space;q&space;\right\Vert^2_2&space;\leq&space;fx&space;&plus;&space;d&space;\\&space;\end{aligned}" title="\large \begin{aligned} \min_{x} \quad & c^Tx\\ \textrm{s.t.} \quad & Ax = b\\ &\left\Vert Qx + q \right\Vert^2_2 \leq fx + d \\ \end{aligned}" /></a> </p>

> *The theoretical documentation will follow-up shortly*

The implementation offers the possibility to run multiple `ConicProblems` with
one or both of the solvers: *proxDykstra* or [*CVX*][4] and compare their
performances in a nice-looking HTML table.

The provided problems must be of SOCP type. The problems can be either generated
in `numpy.dense` or `numpy.sparse` formats or they can be provided in a CBF
format like the problems in the [CBLIB - The Conic Benchmark Library][5].

The repo contains the following files and folders:
- [cblib](cblib) - this folder contains the [CBFData](cblib/scripts/data/CBFdata.py)
 class that was implemented by
the Department of Optimization at [Zuse Institute Berlin][1] and it was taken
from this [repo][2]
- [example_report](example_report) - this folder contains an example HTML report
- [test_problems](test_problems) - a folder containing test auto-generated problems
on which the algorithm was run
- [conic_problem.py](conic_problem.py) - this module contains the implementation for a
`ConicProblem`
- [generation_params.json](generation_params.json) - these are the parameters used by the [problem_generator.py](problem_generator.py) script to generate one of the problems under [test_problems](test_problems)
- [inputs.json](inputs.json) - these are the parameters that were set for the
Proximal Point solver inside the `ConicProblem`
- [matrix_ops.py](matrix_ops.py) - this module contains functions that were used to operate on
sparse and dense matrices
- [problem_generator.py](problem_generator.py) - this script can be used to generate SOCP problems
- [report.tpl](report.tpl) - the [Mako][3] template used for report generation
- [py_requirements.txt](py_requirements.txt) - the `Python` requirements
- [result_analyzer.py](result_analyzer.py) - this module contains functions that are used to
interpret the [runner.py](runner.py) results
- [runner.py](runner.py) - this script runs multiple problems and, at the end, it creates
a report from the algorithms performances

## Installation
The python required modules for this implementation are in the **py_requirements.txt**
file and they can be installed with the following `pip` command:

```
pip install --U -r py_requirements.txt
```

You have to also include the path to the [CBFData.py](cblib/scripts/data/CBFdata.py)
module to the `PYTHONPATH` Environment Variable.

## Generating Problems
You can generate a problem using the `problem_generator.py` script. You have to
provide to this script the path to the directory where to create the problem with
`--problem_folder` and the generation JSON parameters with `--params` and the
script will generate for you a SOCP problem with an automatically generated
name.

The JSON parameters that are provided to this script must respect the following
format:

```
{
    'sol_dim': [min[, max]], # The solution dimension
    'A_dim': [min[, max]], # The number of lines in A
    'Q_dim': [min[, max]], # The number of lines in Q matrices
    'Q_len': [min[, max]], # The number of conic constraints
    'type': 'sparse' | 'dense', # The type of the problem
    'density': 0 <= nb <= 1 # The density of the matrices if type == sparse
}
```

The list items represent a range (min and max). If only a element is found in
a list, then it will constrain the generator to use that exact number.

## Solve the problems
In order to solve a set of problems, you must first include them in a directory
(like [test_problems](test_problems) directory). This directory may have the
following structure:

```
.
+-- problem_cbf.cbf.gz
+-- numpy_dense_problem
|   +-- b.npy
|   +-- c.npy
|   +-- Q_0.npy
...
|   +-- Q_n.npy
+-- scipy_sparse_problem
|   +-- b.npz
|   +-- c.npz
|   +-- Q_0.npz
...
|   +-- Q_n.npz
```

This folder must be passed to the `runner.py` script which will take each of the
found problems inside the directory, load it into a `ConicProblem` object and then
solve it using the provided solvers. You can pass this to the script using the
`--problems_dir` argument.

In the current implementation, you have 2 available solvers that can be used:
`pp` (*proxDykstra*) or `cvx`. If you decide that you want to try the `pp` solver,
please keep in mind that you also have to provide to the `runner.py` script the
proxDykstra JSON parameters (like [inputs.json](inputs.json)) with the
`--pp_params` argument.

After the problems are resolved, the script will generate a HTML report under
a generated *test directory* in the provided `--work_dir` (Default: *current working
directory*).

## Interpreting the HTML results
The HTML table contains 3 parts:
* Short introduction
* *proxDykstra* used parameters (this one can miss if only the `cvx` is used)
* Algorithm Performances

The later part is the most important of all. It is represented by a HTML table
which contains the following headers:
* **\#** - The index of the problem
* **Problem Name** - the name of the problem
* **Algorithm Name** - the name of the used algorithm
* **Iters.** - The iterations that the algorithm needed to compute the solution
* **Min. Value** - the found optimal value by the algorithm
* **Time (s)** - the time (in seconds) needed by the algorithm to compute the
solution

An example of such a report can be found in [example_report/report.html](example_report/report.html).

## Reference
- I. Necoara, O. Ferqoc, *Linear convergence of dual coordinate descent on non-polyhedral convex problems*,  Arxiv preprint:1911.06014, 2019
- CBLIB - The Conic Benchmark Library

### Credits
- [Ion Necoara](https://acse.pub.ro/person/ion-necoara/) - Coordinator
- [Bob Cristian](mailto:b.cristian.cb@gmail.com) - Python implementation
- [Zuse Institute Berlin][1] - CBF format extraction

### Contributions are welcome

[1]: http://www.zib.de/
[2]: https://github.com/HFriberg/cblib-base/
[3]: https://www.makotemplates.org/
[4]: https://www.cvxpy.org/
[5]: https://cblib.zib.de/
