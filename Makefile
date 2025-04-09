.PHONY: clean lint test docs jupyter install release

# Default target
all: clean lint test

# Install development requirements
install:
	pip install -e .
	pip install -r requirements.txt

# Clean up build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete

# Run linting checks
lint:
	flake8 src tests
	black --check src tests

# Format code automatically
format:
	black src tests

# Run tests
test:
	pytest tests/

# Run only unit tests
unit-test:
	pytest tests/ -m unit

# Generate documentation
docs:
	$(MAKE) -C docs html

# Start Jupyter notebook server
jupyter:
	jupyter notebook notebooks/

# Create a release
release: clean
	python setup.py sdist bdist_wheel

# Run the full data processing pipeline
pipeline:
	cd notebooks && jupyter nbconvert --to notebook --execute 01_data_exploration.ipynb
	cd notebooks && jupyter nbconvert --to notebook --execute 02_data_preprocessing.ipynb
	cd notebooks && jupyter nbconvert --to notebook --execute 03_modeling_evaluation.ipynb
