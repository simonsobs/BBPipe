B-modes pipeline constructor
----------------------------
A framework for creating B-mode pipelines for the Simons Observatory. Documentation is still under development, but details about how to use this package to create new pipelines can be found below and in [CONTRIBUTING.md](CONTRIBUTING.md).

## Installation
To install `BBPipe`, just clone this repository and run
```bash
python setup.py install
```

Once installed, you can test the installation by running it on the current [test power-spectrum pipeline](bbpower_test) (mostly made out of placeholders). To do so, type:
```bash
bbpipe test/test.yml
```

## Creating a pipeline
To create a new pipeline, you must create its corresponding pipeline stages, and link them together through a [yaml](http://yaml.org/) file.

Creating a new pipeline stage involves creating a python module. Note that this module doesn't have to live in this repo, it just has to be accessible by `bbpipe` when you run it. The new repo must:

- Have an `__init__.py` file that imports from `.` all the stages used by your pipeline.
- Have a `__main__.py` file with the same contents as those from the example `bbpower_test` [directory](bbpower_test).
- Each stage is defined by a class which must inherit from `bbpipe.PipelineStage`. Each class must have its own `name`, `inputs` and `outputs` attributes (essentially the names of the expected input and output data), and a `run` method that executes the stage.
- The `run` method should use the parent methods from `PipelineStage` to get its inputs and outputs etc.

To create the yaml file that puts your pipeline together, have a look at the [test file](test/test.yml). This file should contain:
- A list of modules where the different pipeline stages are to be found.
- The launcher type (to be used by PARSL to launch each stage). Currently the only defined launcher type is the `local` one (i.e. launch jobs serially in your machine), but more will be defined. They will be located in [`bbpipe/sites`](bbpipe/sites).
- The list of stages that define your pipeline. Note that this list is not related to the order in which the different stages will be executed. This order is automatically determined from the inputs and outputs of each pipeline stage.
- The overall inputs of the pipeline (accessible to all pipeline stages).
- A path to another yaml file (`config`) containing configuration options for each individual pipeline stage. Have a look at [`test/config.yml`](test/config.yml) to see an example for our test power spectrum pipeline.
- A value for the `resume` parameter, which determines whether a given stage is run if its outputs already exist.
- An output directory where the pipeline outputs will be stored.


## Credit
`BBPipe` is heavily inspired by `ceci`, a pipeline constructor designed within the LSST DESC by Joe Zuntz, Francois Lanusse and others.
`BBPipe` uses [PARSL](http://parsl-project.org/).
