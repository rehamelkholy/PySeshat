# Public API

## pyseshat.__init__

## pyseshat.analysis.__init__

## pyseshat.analysis.binary_consistency

### Classes
- BinaryConsistencyResult

### Functions
- compute_mass_consistency()
- normalized_difference()
- evolutionary_total_mass()
- compute_luminosity_consistency()
- build_binary_consistency_result()
- build_binary_consistency_from_elements()
- build_binary_consistency_from_fit_results()

## pyseshat.analysis.bolometric

### Functions
- luminosity_to_mbol()
- mbol_to_luminosity()
- distance_modulus()
- absolute_to_apparent()
- apparent_to_absolute()

## pyseshat.analysis.dynamics

### Functions
- separation_vs_time()
- separation_table()

## pyseshat.analysis.error_propagation

### Functions
- sigma_log10()
- sigma_log_temperature()
- sigma_log_radius()
- sigma_log_luminosity()

## pyseshat.analysis.luminosity

### Functions
- compute_luminosity()
- compute_log_luminosity()
- component_luminosities()
- combined_luminosity()

## pyseshat.analysis.masses

### Functions
- compute_total_mass()
- total_mass_from_elements()
- compute_mass_function()
- compute_component_masses()
- component_masses_from_elements()
- minimum_secondary_mass()
- minimum_secondary_mass_from_elements()

## pyseshat.analysis.precession

### Functions
- compute_precession_rate()
- apply_theta_precession()
- remove_theta_precession()

## pyseshat.analysis.report

### Functions
- build_fit_report()
- report_to_text()

## pyseshat.analysis.statistics

### Functions
- chi_square()
- reduced_chi_square()
- rms_residual()
- parameter_comparison()
- print_parameter_comparison()
- fit_summary()
- combined_fit_summary()
- print_fit_summary()

## pyseshat.ephemeris.__init__

## pyseshat.ephemeris.ephemeris

### Functions
- generate_visual_ephemeris()
- visual_ephemeris_table()
- generate_rv_ephemeris()
- rv_ephemeris_table()
- generate_combined_ephemeris()
- combined_ephemeris_table()

## pyseshat.ephemeris.ephemeris_result

### Dataclasses
- EphemerisResult

### Classes
- EphemerisResult

## pyseshat.evolution.__init__

## pyseshat.evolution.age

### Classes
- TrackFitResult
- CommonAgeFitResult

### Functions
- fit_track()
- track_residuals()
- track_chi2()
- fit_common_age()

## pyseshat.evolution.binary

### Classes
- BinaryTrackFitResult

### Functions
- fit_binary_tracks()
- fit_binary_evolution_from_orbit()

## pyseshat.evolution.grids

### Functions
- load_model()
- load_parsec()

## pyseshat.evolution.hrdiagram

### Classes
- HRDiagramData

### Functions
- build_binary_hrdiagram_data()

## pyseshat.evolution.interpolate

### Classes
- EvolutionSample

### Functions
- contains_age()
- contains_mass()
- sample_track()
- sample_isochrone()

## pyseshat.evolution.isochrone

### Classes
- IsochroneFitResult

### Functions
- fit_isochrone()

## pyseshat.evolution.model_fit

### Classes
- EvolutionModelFitResult

### Functions
- fit_evolution_model()

## pyseshat.evolution.models

### Classes
- EvolutionTrack
- Isochrone
- EvolutionModel

## pyseshat.evolution.observables

### Classes
- StellarObservables

## pyseshat.evolution.parsers.__init__

## pyseshat.evolution.parsers.parsec

### Functions
- load_parsec_track()
- load_parsec_isochrone()

## pyseshat.evolution.residuals

### Classes
- StellarResiduals

### Functions
- compute_residuals()

## pyseshat.evolution.search

### Functions
- nearest_track()
- nearest_isochrone()

## pyseshat.evolution.solution

### Classes
- BinaryEvolutionSolution

## pyseshat.experiments.__init__

## pyseshat.experiments.experiment

### Abstract Base Classes
- Experiment

### Classes
- Experiment

## pyseshat.experiments.experiment_result

### Dataclasses
- ExperimentResult

### Classes
- ExperimentResult

## pyseshat.experiments.experiment_runner

### Classes
- ExperimentRunner

## pyseshat.experiments.monte_carlo

### Dataclasses
- MonteCarloExperiment

### Classes
- MonteCarloExperiment

## pyseshat.experiments.observation_set

## pyseshat.experiments.synthetic

### Functions
- perturb_values()
- perturb_xy()

## pyseshat.inference.__init__

## pyseshat.inference.base_fit_result

### Dataclasses
- BaseFitResult

### Classes
- BaseFitResult

## pyseshat.inference.context

### Classes
- LikelihoodContext

## pyseshat.inference.core

### Functions
- update_elements()

## pyseshat.inference.covariance

### Functions
- estimate_covariance()
- parameter_uncertainties()

## pyseshat.inference.diagnostics

### Functions
- chi_square()
- reduced_chi_square()
- rms_residual()
- fit_summary()

## pyseshat.inference.evolution_likelihood

### Functions
- evolution_log_likelihood()
- log_likelihood()

## pyseshat.inference.fit_result

### Dataclasses
- FitResult

### Classes
- FitResult

## pyseshat.inference.joint_likelihood

### Functions
- joint_log_likelihood()
- joint_log_likelihood_from_parameters()

## pyseshat.inference.joint_parameters

### Classes
- JointParameters

## pyseshat.inference.joint_posterior

### Functions
- joint_log_posterior()

## pyseshat.inference.joint_problem

### Classes
- JointInferenceProblem

## pyseshat.inference.likelihood

### Functions
- combined_residuals()
- chi_square()
- reduced_chi_square()
- log_likelihood()
- log_likelihood_from_context()
- log_likelihood_from_problem()

## pyseshat.inference.mcmc_engine

### Functions
- initialize_walkers()
- run_emcee()

## pyseshat.inference.mcmc_result

### Dataclasses
- MCMCResult

### Classes
- MCMCResult

## pyseshat.inference.node_disambiguation

### Functions
- node_flip()
- node_assignment_chi2()
- select_best_node_assignment()

## pyseshat.inference.observation_wrappers

### Functions
- fit_visual_orbit_from_observations()

## pyseshat.inference.optimizer

### Functions
- build_parameter_bounds()
- fit_visual_orbit()
- fit_visual_orbit_from_problem()
- fit_visual_orbit_mcmc_from_problem()
- fit_visual_orbit_mcmc()

## pyseshat.inference.optimizer_engine

### Functions
- run_least_squares()

## pyseshat.inference.photometric_context

### Classes
- PhotometricContext

## pyseshat.inference.photometric_likelihood

### Functions
- photometric_residuals_from_problem()
- photometric_residuals()
- chi_square()
- reduced_chi_square()
- photometric_log_likelihood_from_problem()
- log_likelihood()

## pyseshat.inference.photometric_mcmc

### Functions
- sample_photometry_from_problem()
- sample_photometry()

## pyseshat.inference.photometric_optimizer

### Functions
- build_parameter_bounds()
- fit_photometry_from_problem()
- fit_photometry()

## pyseshat.inference.photometric_posterior

### Functions
- photometric_log_posterior_from_problem()
- log_posterior()

## pyseshat.inference.photometric_problem

### Classes
- PhotometricInferenceProblem

## pyseshat.inference.posterior

### Functions
- log_posterior()
- log_posterior_from_problem()

## pyseshat.inference.priors

### Functions
- log_uniform_prior()
- log_jeffreys_prior()
- log_isotropic_prior()
- log_prior()
- log_photometric_prior()

## pyseshat.inference.problem

### Classes
- InferenceProblem

## pyseshat.inference.solver

### Functions
- solve()

## pyseshat.inference.solver_engine

### Functions
- solve_problem()

## pyseshat.inference.terms.__init__

## pyseshat.inference.terms.joint

### Functions
- joint_residuals()
- joint_residuals_from_context()

## pyseshat.inference.terms.radial_velocity

### Functions
- rv_residuals()

## pyseshat.inference.terms.visual

### Functions
- visual_xy_residuals()

## pyseshat.io.__init__

## pyseshat.io.coordinates

### Functions
- parse_ra()
- parse_dec()

## pyseshat.io.fit_result_io

### Functions
- orbital_elements_to_dict()
- fit_result_to_dict()
- save_fit_result()
- dict_to_orbital_elements()
- load_fit_result()

## pyseshat.io.orbit_parser

### Functions
- jd_to_decimal_year()
- normalize_time()
- parse_orbit_file()

## pyseshat.io.report_writer

### Functions
- write_text_report()

## pyseshat.observations.__init__

## pyseshat.observations.adapters

### Functions
- visual_arrays()

## pyseshat.observations.radial_velocity

### Classes
- RVObservationSet

## pyseshat.observations.system

### Classes
- SystemObservations

## pyseshat.observations.visual

### Classes
- VisualObservationSet

## pyseshat.orbit.__init__

## pyseshat.orbit.orbit_model

### Functions
- solve_kepler()
- thiele_innes()
- compute_visual_orbit()
- sample_visual_orbit()

## pyseshat.orbit.orbital_elements

### Dataclasses
- OrbitalElements

### Classes
- OrbitalElements

## pyseshat.orbit.phase

### Functions
- orbital_phase()

## pyseshat.orbit.rv_model

### Functions
- compute_rv_orbit()
- compute_rv_model()

## pyseshat.photometry.__init__

## pyseshat.photometry.evolution_adapter

### Functions
- stellar_observables_from_photometric_parameters()
- stellar_observables_from_solution()
- stellar_observables_from_fit_result()

## pyseshat.photometry.extinction

### Functions
- apply_extinction()
- ccm89_optical_extinction()

## pyseshat.photometry.filters

### Dataclasses
- PhotometricPassband

### Classes
- PhotometricPassband

### Functions
- load_passband()

## pyseshat.photometry.model

### Functions
- synthetic_binary_magnitudes()
- binary_photometric_chi_square()

## pyseshat.photometry.observations

### Classes
- PhotometricMeasurement

## pyseshat.photometry.parameters

### Classes
- PhotometricParameters

### Functions
- update_parameters()

## pyseshat.photometry.residuals

### Functions
- photometric_residual()
- photometric_chi_square()

## pyseshat.photometry.synthetic

### Functions
- synthetic_band_flux()
- pogson_magnitude()

## pyseshat.stellar.__init__

## pyseshat.stellar.atmospheres

### Dataclasses
- AtmosphereModel

### Classes
- AtmosphereModel
- CK04CatalogEntry
- AtmosphereGrid

### Functions
- load_ck04_fits()
- load_ck04_grid()
- load_ck04_catalog()

## pyseshat.stellar.sed_model

### Functions
- scale_surface_flux()
- composite_binary_sed()

## pyseshat.triple.__init__

## pyseshat.triple.triple_bounds

### Functions
- build_triple_bounds()

## pyseshat.triple.triple_diagnostics

### Functions
- chi_square()
- reduced_chi_square()
- rms_residual()
- triple_fit_summary()

## pyseshat.triple.triple_elements

### Dataclasses
- TripleOrbitalElements

### Classes
- TripleOrbitalElements

## pyseshat.triple.triple_fit

### Functions
- triple_to_array()
- array_to_triple()
- update_triple_elements()
- triple_fitted_parameters()
- update_triple_fitted_parameters()
- triple_fit_flag_residuals()
- triple_fit_residuals()
- triple_B_residuals()
- triple_a2_residuals()
- fit_triple_orbit()

## pyseshat.triple.triple_fit_result

### Dataclasses
- TripleFitResult

### Classes
- TripleFitResult

## pyseshat.triple.triple_identifiability

### Functions
- covariance_condition_number()
- covariance_eigenvalues()
- covariance_rank()
- is_degenerate()

## pyseshat.triple.triple_observation

### Dataclasses
- TripleObservation

### Classes
- TripleObservation

## pyseshat.triple.triple_orbit

### Functions
- compute_triple_orbit()

## pyseshat.triple.triple_parameters

### Functions
- triple_to_array()
- array_to_triple()

## pyseshat.triple.triple_parser

### Functions
- parse_triple_file()

## pyseshat.triple.triple_pca

### Functions
- covariance_pca()
- best_constrained_mode()
- worst_constrained_mode()
- variance_ratio()

## pyseshat.triple.triple_residuals

### Functions
- triple_visual_residuals()

## pyseshat.triple.triple_statistics

### Functions
- correlation_matrix()
- print_correlation_matrix()

## pyseshat.validation.__init__

## pyseshat.validation.metrics

### Functions
- parameter_bias()
- absolute_error()
- relative_error()
- normalized_error()

## pyseshat.visualization.__init__

## pyseshat.visualization.dynamics_plot

### Functions
- plot_separation_vs_time()

## pyseshat.visualization.hrdiagram_plot

### Functions
- plot_binary_hrdiagram()

## pyseshat.visualization.orbit_plot

### Functions
- plot_visual_orbit()
- plot_residuals()
- plot_rv_orbit()

