#!/usr/bin/env cwl-runner

class: Workflow
cwlVersion: v1.0
inputs:
  aposize_edges@BBMaskPreproc: {default: 1.0, id: aposize_edges@BBMaskPreproc, label: aposize_edges,
    type: float}
  aposize_srcs@BBMaskPreproc: {default: 0.1, id: aposize_srcs@BBMaskPreproc, label: aposize_srcs,
    type: float}
  apotype_edges@BBMaskPreproc: {default: C1, id: apotype_edges@BBMaskPreproc, label: apotype_edges,
    type: string}
  apotype_srcs@BBMaskPreproc: {default: C1, id: apotype_srcs@BBMaskPreproc, label: apotype_srcs,
    type: string}
  beam_correct@BBCovFeFe: {default: true, id: beam_correct@BBCovFeFe, label: beam_correct,
    type: boolean}
  beam_correct@BBNullTester: {default: true, id: beam_correct@BBNullTester, label: beam_correct,
    type: boolean}
  beam_correct@BBPowerSpecter: {default: true, id: beam_correct@BBPowerSpecter, label: beam_correct,
    type: boolean}
  binary_mask: {format: FitsFile, id: binary_mask, label: binary_mask, type: File}
  bpw_edges@BBCovFeFe:
    default: [2, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290,
      350, 1000]
    id: bpw_edges@BBCovFeFe
    label: bpw_edges
    type: {items: int, type: array}
  bpw_edges@BBNullTester:
    default: [2, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290,
      350, 1000]
    id: bpw_edges@BBNullTester
    label: bpw_edges
    type: {items: int, type: array}
  bpw_edges@BBPowerSpecter:
    default: [2, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290,
      350, 1000]
    id: bpw_edges@BBPowerSpecter
    label: bpw_edges
    type: {items: int, type: array}
  config: {format: YamlFile, id: config, label: config, type: File}
  purify_b@BBMapsPreproc: {id: purify_b@BBMapsPreproc, label: purify_b, type: boolean}
  simulation_info: {format: YamlFile, id: simulation_info, label: simulation_info,
    type: File}
  source_data: {format: TextFile, id: source_data, label: source_data, type: File}
  splits_info: {format: YamlFile, id: splits_info, label: splits_info, type: File}
outputs:
  null_spectra: {format: DummyFile, id: null_spectra, label: null_spectra, outputSource: BBNullTester/null_spectra,
    type: File}
steps:
  BBCompSep:
    id: BBCompSep
    in: {config: config, covariance_matrix: BBCovFeFe/covariance_matrix, power_spectra_splits: BBPowerSpecter/power_spectra_splits,
      splits_info: splits_info}
    out: [param_chains]
    run: BBCompSep.cwl
  BBCovFeFe:
    id: BBCovFeFe
    in: {beam_correct: beam_correct@BBCovFeFe, bpw_edges: bpw_edges@BBCovFeFe, config: config,
      mode_coupling_matrix: BBPowerSpecter/mode_coupling_matrix, simulation_info: simulation_info,
      splits_info: splits_info}
    out: [sims_powspec_list, covariance_matrix]
    run: BBCovFeFe.cwl
  BBMapsPreproc:
    id: BBMapsPreproc
    in: {config: config, purify_b: purify_b@BBMapsPreproc, splits_info: splits_info,
      window_function: BBMaskPreproc/window_function}
    out: [nmt_fields]
    run: BBMapsPreproc.cwl
  BBMaskPreproc:
    id: BBMaskPreproc
    in: {aposize_edges: aposize_edges@BBMaskPreproc, aposize_srcs: aposize_srcs@BBMaskPreproc,
      apotype_edges: apotype_edges@BBMaskPreproc, apotype_srcs: apotype_srcs@BBMaskPreproc,
      binary_mask: binary_mask, config: config, source_data: source_data}
    out: [window_function]
    run: BBMaskPreproc.cwl
  BBNullTester:
    id: BBNullTester
    in: {beam_correct: beam_correct@BBNullTester, bpw_edges: bpw_edges@BBNullTester,
      config: config, power_spectra_splits: BBPowerSpecter/power_spectra_splits, splits_info: splits_info}
    out: [null_spectra]
    run: BBNullTester.cwl
  BBPowerSpecter:
    id: BBPowerSpecter
    in: {beam_correct: beam_correct@BBPowerSpecter, bpw_edges: bpw_edges@BBPowerSpecter,
      config: config, nmt_fields: BBMapsPreproc/nmt_fields, splits_info: splits_info,
      window_function: BBMaskPreproc/window_function}
    out: [power_spectra_splits, mode_coupling_matrix]
    run: BBPowerSpecter.cwl
