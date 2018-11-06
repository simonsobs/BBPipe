import os
import yaml
import sys
import parsl
import argparse
from . import Pipeline, PipelineStage
from . import sites

# Add the current dir to the path - often very useful
sys.path.append(os.getcwd())

parser = argparse.ArgumentParser(description='Run a Ceci pipeline from a configuration file')
parser.add_argument('pipeline_config', help='Pipeline configuration file in YAML format.')

def run(pipeline_config_filename):
    """
    Runs the pipeline
    """
    # YAML input file.
    pipe_config = yaml.load(open(pipeline_config_filename))

    # Optional logging of pipeline infrastructure to
    # file.
    log_file = pipe_config.get('pipeline_log')
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
    output_dir = pipe_config['output_dir']
    inputs = pipe_config['inputs']
    log_dir = pipe_config['log_dir']
    resume = pipe_config['resume']

    stages_config = pipe_config['config']

    for module in modules:
        __import__(module)

    # Create and run pipeline
    pipeline = Pipeline(launcher_config, stages)
    pipeline.run(inputs, output_dir, log_dir, resume, stages_config)

def main():
    args = parser.parse_args()
    run(args.pipeline_config)

if __name__ == '__main__':
    main()
