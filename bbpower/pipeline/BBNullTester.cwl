#!/usr/bin/env cwl-runner

arguments:
- {loadContents: false, position: -1, separate: true, shellQuote: true, valueFrom: -mbbpower}
- {loadContents: false, position: 0, separate: true, shellQuote: true, valueFrom: BBNullTester}
baseCommand: python3
class: CommandLineTool
cwlVersion: v1.0
doc: "\n    Template for a null testing stage\n    "
id: BBNullTester
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
label: BBNullTester
outputs:
  null_spectra:
    doc: Some results produced by the pipeline element
    format: DummyFile
    label: null_spectra
    outputBinding: {glob: null_spectra.dum}
    type: File
