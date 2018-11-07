#!/usr/bin/env cwl-runner

arguments:
- {loadContents: false, position: -1, separate: true, shellQuote: true, valueFrom: -mbbpower}
- {loadContents: false, position: 0, separate: true, shellQuote: true, valueFrom: BBPowerSpecter}
baseCommand: python3
class: CommandLineTool
cwlVersion: v1.0
doc: "\n    Template for a power spectrum stage\n    "
id: BBPowerSpecter
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
  nmt_fields:
    doc: Some documentation about the input
    format: DummyFile
    inputBinding: {loadContents: false, prefix: --nmt_fields, separate: true, shellQuote: true}
    label: nmt_fields
    type: File
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
label: BBPowerSpecter
outputs:
  mode_coupling_matrix:
    doc: Some results produced by the pipeline element
    format: DummyFile
    label: mode_coupling_matrix
    outputBinding: {glob: mode_coupling_matrix.dum}
    type: File
  power_spectra_splits:
    doc: Some results produced by the pipeline element
    format: DummyFile
    label: power_spectra_splits
    outputBinding: {glob: power_spectra_splits.dum}
    type: File
