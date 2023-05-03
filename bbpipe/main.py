import os
import yaml
import sys
import parsl
import argparse
from . import Pipeline, PipelineStage
from . import sites
import time
import shutil

# Add the current dir to the path - often very useful
sys.path.append(os.getcwd())

parser = argparse.ArgumentParser(description='Run a Ceci pipeline from a configuration file')
parser.add_argument('pipeline_config', help='Pipeline configuration file in YAML format.')
parser.add_argument('--export-cwl', type=str, help='Exports pipeline in CWL format to provided path and exits')
parser.add_argument('--dry-run', action='store_true', help='Just print out the commands the pipeline would run without running them')
parser.add_argument('--python-cmd', type=str, default='python3', help='Command that calls the python interpreter')

def run(pipeline_config_filename, dry_run=False, pycmd='python3'):
    """
    Runs the pipeline
    """

    # Get current time in Unix milliseconds to define log directory
    init_time_ms = int(time.time()*1e3)

    # YAML input file.
    pipe_config = yaml.safe_load(open(pipeline_config_filename))
    output_dir = pipe_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Log directory should not already exist
    log_dir = f'{output_dir}/run_{str(init_time_ms)}'
    os.makedirs(log_dir, exist_ok=False)

    # Copy the main config files into the log directory
    shutil.copyfile(pipeline_config_filename,f'{log_dir}/pipeline_config.yml')
    stages_config = pipe_config['config']
    shutil.copyfile(stages_config,f'{log_dir}/stages_config.yml')

    # Optional logging of pipeline infrastructure to
    # file.
    log_file = f'{log_dir}/pipeline_log.txt'

    if log_file:
        parsl.set_file_logger(log_file)

    # Required configuration information
    # List of stage names, must be imported somewhere
    stages = pipe_config['stages']

    # Python modules in which to search for pipeline stages
    modules = pipe_config['modules'].split()

    # parsl execution/launcher configuration information
    launcher = pipe_config.get("launcher", "local")
    if launcher == "local":
        launcher_config = sites.local.make_launcher(stages)
    elif launcher == "cori":
        launcher_config = sites.cori.make_launcher(stages)
    elif launcher == "cori-interactive":
        launcher_config = sites.cori_interactive.make_launcher(stages)
    else:
        raise ValueError(f"Unknown launcher {launcher}")
    # 
    # launcher_config = pipe_config['launcher']

    # Inputs and outputs
    inputs = pipe_config['inputs']
    resume = pipe_config['resume']


    for module in modules:
        __import__(module)

    # Create and run pipeline
    pipeline = Pipeline(launcher_config, stages, log_dir, pycmd=pycmd)

    if dry_run:
        pipeline.dry_run(inputs, output_dir, stages_config)
    else:
        pipeline.run(inputs, output_dir, resume, stages_config)

def export_cwl(args):
    """
    Function exports pipeline or pipeline stages into CWL format.
    """
    path = args.export_cwl
    # YAML input file.
    config = yaml.load(open(args.pipeline_config))

    # Python modules in which to search for pipeline stages
    modules = config['modules'].split()
    for module in modules:
        __import__(module)

    # Export each pipeline stage as a CWL app
    for k in PipelineStage.pipeline_stages:
        tool = PipelineStage.pipeline_stages[k][0].generate_cwl()
        tool.export(f'{path}/{k}.cwl')

    stages = config['stages']

    # Exports the pipeline itself
    launcher = config.get("launcher", "local")
    if launcher == "local":
        launcher_config = sites.local.make_launcher(stages)
    elif launcher == "cori":
        launcher_config = sites.cori.make_launcher(stages)
    else:
        raise ValueError(f"Unknown launcher {launcher}")

    inputs = config['inputs']

    pipeline = Pipeline(launcher_config, stages)
    cwl_wf = pipeline.generate_cwl(inputs)
    cwl_wf.export(f'{path}/pipeline.cwl')


def main():
    args = parser.parse_args()
    if args.export_cwl is not None:
        export_cwl(args)
    else:
        run(args.pipeline_config, dry_run=args.dry_run, pycmd=args.python_cmd)

if __name__ == '__main__':
    main()
