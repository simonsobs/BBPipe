#!/usr/bin/env cwl-runner

arguments:
- {loadContents: false, position: -1, separate: true, shellQuote: true, valueFrom: -mbbpower}
- {loadContents: false, position: 0, separate: true, shellQuote: true, valueFrom: BBMapsPreproc}
baseCommand: python3
class: CommandLineTool
cwlVersion: v1.0
doc: "\n    Template for a map pre-processing stage\n    "
id: BBMapsPreproc
inputs:
  config:
    doc: Configuration file
    format: YamlFile
    inputBinding: {loadContents: false, prefix: --config, separate: true, shellQuote: true}
    label: config
    type: File
  purify_b:
    doc: Some documentation about this parameter
    inputBinding: {loadContents: false, prefix: --purify_b, separate: true, shellQuote: true}
    label: purify_b
    type: boolean
  splits_info:
    doc: Some documentation about the input
    format: YamlFile
    inputBinding: {loadContents: false, prefix: --splits_info, separate: true, shellQuote: true}
    label: splits_info
    type: File
  window_function:
    doc: Some documentation about the input
    format: FitsFile
    inputBinding: {loadContents: false, prefix: --window_function, separate: true,
      shellQuote: true}
    label: window_function
    type: File
label: BBMapsPreproc
outputs:
  nmt_fields:
    doc: Some results produced by the pipeline element
    format: DummyFile
    label: nmt_fields
    outputBinding: {glob: nmt_fields.dum}
    type: File
