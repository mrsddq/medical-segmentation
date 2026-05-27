.PHONY: install test compile clean

install:
	python -m pip install -r requirements.txt

test:
	python -m unittest discover -s tests

compile:
	python -m compileall -q .

clean:
	python -c "import shutil; from pathlib import Path; [shutil.rmtree(p, ignore_errors=True) for p in Path('.').rglob('__pycache__')]"
