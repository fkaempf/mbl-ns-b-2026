#!/bin/bash
# Double-click in Finder (or run in a terminal) to open the napari bounce selector.
# Uses the nsb_fly conda env and runs from the code dir so the imports/data resolve.
cd /Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/fly/code || exit 1
exec /Users/fkampf/miniforge3/envs/nsb_fly/bin/python bounce_select.py
