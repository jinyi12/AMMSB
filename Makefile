PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest
RUFF ?= ruff

.PHONY: help setup install-local install-csp install-skills lint check repo-health hotspots test test-harness test-tran-eval test-csp smoke-tran-eval smoke-csp

help:
	@printf "Available targets:\n"
	@printf "  setup              - provision the full repo environment via make_venv.sh\n"
	@printf "  install-local      - install local Python dependencies into the current environment\n"
	@printf "  install-csp        - install the base repo environment plus optional CSP Python dependencies\n"
	@printf "  install-skills     - install repo-local Codex skills into the global discovery directory\n"
	@printf "  lint               - run Ruff on active code surfaces\n"
	@printf "  check              - run the legacy validation script\n"
	@printf "  repo-health        - validate harness surfaces and doc links\n"
	@printf "  hotspots           - report large active files and repeated helper names\n"
	@printf "  test               - run the full test suite\n"
	@printf "  test-harness       - run harness and repo-health tests\n"
	@printf "  test-tran-eval     - run focused Tran-evaluation support tests\n"
	@printf "  test-csp           - run CSP tests when optional dependencies are available\n"
	@printf "  smoke-tran-eval    - run CLI help checks for the main Tran-evaluation entrypoints\n"
	@printf "  smoke-csp          - run CLI help checks for the main CSP entrypoints\n"

setup:
	bash make_venv.sh

install-local:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

install-csp: install-local
	$(PYTHON) -m pip install -e ".[csp]"

install-skills:
	$(PYTHON) scripts/install_repo_skills.py

lint:
	$(RUFF) check mmsfm/ csp/ scripts/ tests/

check:
	bash scripts/check.sh

repo-health:
	$(PYTHON) scripts/repo_health.py

hotspots:
	$(PYTHON) scripts/refactor_hotspots.py

test:
	$(PYTEST) -q tests/

test-harness:
	$(PYTEST) -q tests/test_harness_inventory.py tests/test_repo_health.py tests/test_install_repo_skills.py

test-tran-eval:
	$(PYTEST) -q \
		tests/test_tran_evaluation_run_support.py \
		tests/test_tran_evaluation_conditional_support.py \
		tests/test_tran_evaluation_conditional_metrics.py \
		tests/test_tran_evaluation_latent_msbm_runtime.py \
		tests/test_tran_evaluation_statistics.py \
		tests/test_latent_geometry_model_comparison.py \
		tests/test_latent_geometry.py

test-csp:
	$(PYTEST) -q tests/test_csp.py tests/test_csp_runtime.py tests/test_csp_token_dit.py tests/test_csp_token_runtime.py

smoke-tran-eval:
	$(PYTHON) scripts/fae/tran_evaluation/evaluate.py --help >/dev/null
	$(PYTHON) scripts/fae/tran_evaluation/generate.py --help >/dev/null
	$(PYTHON) scripts/fae/tran_evaluation/evaluate_conditional.py --help >/dev/null
	$(PYTHON) scripts/fae/tran_evaluation/evaluate_conditional_diagnostic.py --help >/dev/null
	$(PYTHON) scripts/fae/tran_evaluation/evaluate_postfiltered_consistency.py --help >/dev/null
	$(PYTHON) scripts/fae/tran_evaluation/compare_latent_geometry_models.py --help >/dev/null
	$(PYTHON) scripts/fae/tran_evaluation/encode_corpus.py --help >/dev/null
	$(PYTHON) scripts/fae/tran_evaluation/visualize_latent_msbm_manifold.py --help >/dev/null
	$(PYTHON) scripts/fae/tran_evaluation/visualize_conditional_latent_projections.py --help >/dev/null

smoke-csp:
	$(PYTHON) scripts/csp/encode_fae_latents.py --help >/dev/null
	$(PYTHON) scripts/csp/encode_fae_token_latents.py --help >/dev/null
	$(PYTHON) scripts/csp/train_csp.py --help >/dev/null
	$(PYTHON) scripts/csp/train_csp_from_fae.py --help >/dev/null
	$(PYTHON) scripts/csp/train_csp_token_dit.py --help >/dev/null
	$(PYTHON) scripts/csp/train_csp_token_dit_from_fae.py --help >/dev/null
	$(PYTHON) scripts/csp/evaluate_csp.py --help >/dev/null
	$(PYTHON) scripts/csp/evaluate_csp_conditional.py --help >/dev/null
	$(PYTHON) scripts/csp/build_eval_cache.py --help >/dev/null
	$(PYTHON) scripts/csp/evaluate_csp_token_dit.py --help >/dev/null
	$(PYTHON) scripts/csp/evaluate_csp_token_dit_conditional.py --help >/dev/null
	$(PYTHON) scripts/csp/build_eval_cache_token_dit.py --help >/dev/null
	$(PYTHON) scripts/csp/train_csp_benchmark.py --help >/dev/null
	$(PYTHON) scripts/csp/plot_csp_training.py --help >/dev/null
