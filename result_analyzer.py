import os
import logging
import matplotlib.pyplot as plt
from mako import exceptions
from mako.template import Template

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

def dj_analyzer(dj_analyzer_dir, dj_metadata, eps_dj, invoc):
    """
    This function uses the data provided by a Dykstra step and plots the
    results.

    Args:
        dj_analyzer_dir:    The directory where the results are going to be
                            plotted.
        dj_metadata:        A list of dicts containing multiple Dykstra steps
                            metadata.
        eps_dj:             The epsilon used for Dykstra algorithm
        invoc:              The invocation number of the Dykstra step.
    """
    if not dj_metadata['iter_metadata']:
        _logger.warn("No metadata for Dijkstra algorithm! Not ploting ...")
        return

    if not os.path.isdir(dj_analyzer_dir):
        _logger.info(f"Creating Dijkstra Analyses Dir: {dj_analyzer_dir} ...")
        os.system(f'mkdir -p {dj_analyzer_dir}')

    _logger.info("Analyzing Dijkstra algorithm invocation ...")

    iters = [metadata['iter'] for metadata in dj_metadata['iter_metadata']]
    crt_outs = [metadata['crt_out'] for metadata in dj_metadata[
        'iter_metadata']]

    plt.plot(iters, crt_outs, label='crt_out')
    plt.plot([iters[0], iters[-1]], [eps_dj, eps_dj], '--r', label='eps')

    plt.title(f"Dijkstra crt_out {invoc}")
    plt.ylabel("crt_out")
    plt.xlabel("iter")
    plt.legend()

    plt_file = os.path.join(dj_analyzer_dir, f"invoc_{invoc}.png")
    _logger.info(f"Saving Dijkstra analysis into {plt_file} ...")
    plt.savefig(plt_file)

    plt.clf()
    plt.cla()
    plt.close()


def result_analyzer(result_folder, iter_metadata, params):
    """
    This function uses the data provided by the Proximal Point algorithm and
    plots the results.

    Args:
        result_folder:      The directory where the results are going to be
                            plotted.
        iter_metadata:      A list of dicts containing multiple PP steps
                            metadata.
        params:             The parameters used for Proximal Point
    """
    if not iter_metadata:
        _logger.warn("No iteration metadata provided! Not ploting ...")
        return

    if not os.path.isdir(result_folder):
        _logger.info(f"Creating plots directory {result_folder} ...")
        os.system(f'mkdir -p {result_folder}')

    dj_analyzer_dir = os.path.join(result_folder, "dijkstra_plots")
    gammas = []
    iters = []
    crt_outs = []
    f_vals = []
    eps_pp = params['eps_pp']
    for metadata in iter_metadata:
        iters.append(metadata['iter'])
        gammas.append(metadata['gamma'])
        crt_outs.append(metadata['crt_out'])
        f_vals.append(metadata['f_val'])
        dj_analyzer(
            dj_analyzer_dir=dj_analyzer_dir,
            dj_metadata=metadata.get('dj_meta', {}),
            eps_dj=params['eps_dj'],
            invoc=iters[-1]
        )

    # Gamma Evolution plot
    plt.plot(iters, gammas)

    plt.title(f"Proximal Point gamma evo")
    plt.ylabel("gamma")
    plt.xlabel("iter")

    plt_file = os.path.join(result_folder, "pp_gamma_evo.png")
    _logger.info(f"Saving Gamma analysis into {plt_file} ...")
    plt.savefig(plt_file)

    plt.clf()
    plt.cla()
    plt.close()

    # Proximal Point Crt Out plot
    plt.plot(iters, crt_outs, label="crt_out")
    plt.plot([iters[0], iters[-1]], [eps_pp, eps_pp], '--r', label='eps')
    plt.title(f"Proximal Point crt_out evo")
    plt.ylabel("crt_out")
    plt.xlabel("iter")

    plt_file = os.path.join(result_folder, "pp_crt_out_evo.png")
    _logger.info(f"Saving Crt Out analysis into {plt_file} ...")
    plt.savefig(plt_file)

    plt.clf()
    plt.cla()
    plt.close()

    # Objective Function values plot
    plt.plot(iters, f_vals)

    plt.title(f"Proximal Point Objective Function evo")
    plt.ylabel("f")
    plt.xlabel("iter")

    plt_file = os.path.join(result_folder, "pp_obj_evo.png")
    _logger.info(f"Saving Objective Function analysis into {plt_file} ...")
    plt.savefig(plt_file)

    plt.clf()
    plt.cla()
    plt.close()


def create_html_report(tpl_file, results_folder, problems, params=None):
    """
    This function is used to generate a HTML report using mako templating.

    Args:
        tpl_file:       The file to the Mako template.
        results_folder: The path to where to save the HTML report.
        problems:       The list of problems.
        params:         The dict of used parameters to solve the problem.
    """
    if not os.path.isfile(tpl_file):
        _logger.error("The provided template file does not exist: "
                      f"{os.path.abspath(tpl_file)}")
        return

    if not problems:
        _logger.error(f"No problems were provided in 'problems'!")
        return

    if not os.path.isdir(results_folder):
        _logger.error("The provided results_folder does not exist: "
                      f"{result_folder}")

    report_file = os.path.join(results_folder, "report.html")
    _logger.info(f"Generating HTML ...")
    try:
        html_content = Template(
            filename=tpl_file, strict_undefined=True
        ).render(
            problems=problems,
            params=params
        )
    except Exception as ex:
        _logger.error("There was an error while generating the HTML report!")
        html_content = str(exceptions.html_error_template().render())
        print(str(ex))

    _logger.info(f"Saving HTML content in: {report_file} ...")
    with open(report_file, 'w') as f:
        f.write(html_content)
