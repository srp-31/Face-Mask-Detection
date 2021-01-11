.PHONY: clean clean-build clean-pyc clean-test clean-venv  format help lint test-coverage .MAKEVENV .MAKEPOETRY
.DEFAULT_GOAL := help
#Variables available to all targets
SAMPLE_INPUT :=--header "Content-Type: application/json" --request POST --data '[{"fixed acidity": 7.4, "citric acid": 0, "sulphates": 0.56, "alcohol": 9.4}]'
DETECTED_OS := $(shell uname -s)
#------------------------------------------------------------------------------------------------------------------------------

help:  ## Print this help text.
	@echo "Please provide a make target."
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install-dependencies: ## Install production dependencies locally using poetry.
	@poetry install --no-dev

install-dev-dependencies: ## Install development dependencies(testing, linting etc.) locally using poetry.
	@poetry install

poetry-lock: ## Lock dependencies.
	@poetry lock

run-app: ## Run the deployment Flask app locally
	@poetry run python ./deploy/app.py

install-pre-commit-hooks: ## Install pre-commit hooks for the current git repo.
	@pre-commit install

lint:install-pre-commit-hooks ## Run code linters.
	SKIP=black,isort poetry run pre-commit run --all-files

format-code:   ## Auto-format all Python code to project standard.
	@poetry run isort deploy tests
	@poetry run black deploy tests

test: export PYTHONPATH=deploy
test: ## Run the tests using pytest.
	poetry run pytest

#Docker Commands
build-image: ## Build the docker image from the current folder using the file sin deploy folder.
	@docker build --no-cache -t $(DOCKER_IMAGE_NAME) -f ./Dockerfile .


push-image:   ##Push the docker image to AWS ECR.
	@docker push $(DOCKER_IMAGE_NAME)

sample-request-docker: ## Test the local docker container on the host PC with a sample request.
	@curl $(SAMPLE_INPUT) http://127.0.0.1:5000/predict

