.PHONY: all clean

all: setup

setup:
	python -m venv venv
	./venv/Scripts/activate && pip install -r requirements.txt

clean:
	rm -rf venv __pycache__
