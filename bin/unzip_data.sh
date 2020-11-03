#!/bin/bash
for f in /datadrive/stream_data/*.zip; do
	7z x $f -o/datadrive/stream_data
done
