#!/bin/bash
source venv/bin/activate
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_sm
