#!/usr/bin/env cwl-runner

arguments:
- {loadContents: false, position: -1, separate: true, shellQuote: true, valueFrom: -mbbpower}
- {loadContents: false, position: 0, separate: true, shellQuote: true, valueFrom: BBCompSep}
baseCommand: python3
class: CommandLineTool
cwlVersion: v1.0
doc: "\n    Template for a component separation stage\n    "
id: BBCompSep
inputs:
  config:
    doc: Configuration file
    format: YamlFile
    inputBinding: {loadContents: false, prefix: --config, separate: true, shellQuote: true}
    label: config
    type: File
  covariance_matrix:
    doc: Some documentation about the input
    format: DummyFile
    inputBinding: {loadContents: false, prefix: --covariance_matrix, separate: true,
      shellQuote: true}
    label: covariance_matrix
    type: File
  power_spectra_splits:
    doc: Some documentation about the input
    format: DummyFile
    inputBinding: {loadContents: false, prefix: --power_spectra_splits, separate: true,
      shellQuote: true}
    label: power_spectra_splits
    type: File
  splits_info:
    doc: Some documentation about the input
    format: YamlFile
    inputBinding: {loadContents: false, prefix: --splits_info, separate: true, shellQuote: true}
    label: splits_info
    type: File
label: BBCompSep
outputs:
  param_chains:
    doc: Some results produced by the pipeline element
    format: DummyFile
    label: param_chains
    outputBinding: {glob: param_chains.dum}
    type: File
