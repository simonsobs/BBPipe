#!/usr/bin/env cwl-runner

arguments:
- {loadContents: false, position: -1, separate: true, shellQuote: true, valueFrom: -mbbpower}
- {loadContents: false, position: 0, separate: true, shellQuote: true, valueFrom: BBCovFeFe}
baseCommand: python3
class: CommandLineTool
cwlVersion: v1.0
doc: "\n    Template for a covariance matrix stage\n    "
id: BBCovFeFe
inputs:
  beam_correct:
    default: true
    doc: Some documentation about this parameter
    inputBinding: {loadContents: false, prefix: --beam_correct, separate: true, shellQuote: true}
    label: beam_correct
    type: boolean
  bpw_edges:
    default: [2, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290,
      350, 1000]
    doc: Some documentation about this parameter
    inputBinding: {itemSeparator: ',', loadContents: false, prefix: --bpw_edges=,
      separate: false, shellQuote: true}
    label: bpw_edges
    type: {items: int, type: array}
  config:
    doc: Configuration file
    format: YamlFile
    inputBinding: {loadContents: false, prefix: --config, separate: true, shellQuote: true}
    label: config
    type: File
  mode_coupling_matrix:
    doc: Some documentation about the input
    format: DummyFile
    inputBinding: {loadContents: false, prefix: --mode_coupling_matrix, separate: true,
      shellQuote: true}
    label: mode_coupling_matrix
    type: File
  simulation_info:
    doc: Some documentation about the input
    format: YamlFile
    inputBinding: {loadContents: false, prefix: --simulation_info, separate: true,
      shellQuote: true}
    label: simulation_info
    type: File
  splits_info:
    doc: Some documentation about the input
    format: YamlFile
    inputBinding: {loadContents: false, prefix: --splits_info, separate: true, shellQuote: true}
    label: splits_info
    type: File
label: BBCovFeFe
outputs:
  covariance_matrix:
    doc: Some results produced by the pipeline element
    format: DummyFile
    label: covariance_matrix
    outputBinding: {glob: covariance_matrix.dum}
    type: File
  sims_powspec_list:
    doc: Some results produced by the pipeline element
    format: TextFile
    label: sims_powspec_list
    outputBinding: {glob: sims_powspec_list.txt}
    type: File
