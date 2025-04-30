#!/bin/bash

# Start FastWhisperAPI service
# Activate conda environment first: conda activate howdy310
cd FastWhisperAPI
uvicorn main:app --reload
