"""Ganglion-level calcium activity analysis for leech nerve-cord recordings.

Three stages, each independently runnable:
  1. signals          - load a suite2p run, build background images, ROI dF/F (pure)
  2. annotate_ganglia - napari tool to draw one polygon per ganglion (GUI)
  3. ganglion_activity- correlation / raster / lag / events from region traces (pure math + figures)
"""
