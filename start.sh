#!/bin/bash

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macOS detected. Using Python 3.10.14 virtual environment..."

    if ! command -v pyenv &> /dev/null; then
        echo "pyenv not found. Install it first."
        exit 1
    fi

    pyenv shell 3.10.14

    PYTHON=$(pyenv which python)

    if [ ! -d "venv" ]; then
        $PYTHON -m venv venv
    fi

    source venv/bin/activate

    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
    python app.py
else
    echo "Non-macOS detected. Setting up virtual environment..."

    pip install --upgrade pip
    pip install -r requirements.txt

    python app.py
fi
