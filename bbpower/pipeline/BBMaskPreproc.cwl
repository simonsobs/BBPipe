#!/usr/bin/env cwl-runner

arguments:
- {loadContents: false, position: -1, separate: true, shellQuote: true, valueFrom: -mbbpower}
- {loadContents: false, position: 0, separate: true, shellQuote: true, valueFrom: BBMaskPreproc}
baseCommand: python3
class: CommandLineTool
cwlVersion: v1.0
doc: "\n    Template for a map pre-processing stage\n    "
id: BBMaskPreproc
inputs:
  aposize_edges:
    default: 1.0
    doc: Some documentation about this parameter
    inputBinding: {loadContents: false, prefix: --aposize_edges=, separate: false,
      shellQuote: true}
    label: aposize_edges
    type: float
  aposize_srcs:
    default: 0.1
    doc: Some documentation about this parameter
    inputBinding: {loadContents: false, prefix: --aposize_srcs=, separate: false,
      shellQuote: true}
    label: aposize_srcs
    type: float
  apotype_edges:
    default: C1
    doc: Some documentation about this parameter
    inputBinding: {loadContents: false, prefix: --apotype_edges=, separate: false,
      shellQuote: true}
    label: apotype_edges
    type: string
  apotype_srcs:
    default: C1
    doc: Some documentation about this parameter
    inputBinding: {loadContents: false, prefix: --apotype_srcs=, separate: false,
      shellQuote: true}
    label: apotype_srcs
    type: string
  binary_mask:
    doc: Some documentation about the input
    format: FitsFile
    inputBinding: {loadContents: false, prefix: --binary_mask, separate: true, shellQuote: true}
    label: binary_mask
    type: File
  config:
    doc: Configuration file
    format: YamlFile
    inputBinding: {loadContents: false, prefix: --config, separate: true, shellQuote: true}
    label: config
    type: File
  source_data:
    doc: Some documentation about the input
    format: TextFile
    inputBinding: {loadContents: false, prefix: --source_data, separate: true, shellQuote: true}
    label: source_data
    type: File
label: BBMaskPreproc
outputs:
  window_function:
    doc: Some results produced by the pipeline element
    format: FitsFile
    label: window_function
    outputBinding: {glob: window_function.fits}
    type: File
