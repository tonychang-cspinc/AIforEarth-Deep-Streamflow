#!/bin/bash
IMAGENAME=cspinc/deep-streamflow:latest

nvidia-docker run -it --rm \
	--name jupyter \
	-p 8080:8080 \
	-v $(pwd):/content \
	-w /content \
	-v /datadrive:/datadrive \
	$IMAGENAME \
	jupyter notebook --port 8080 --ip 0.0.0.0 --no-browser --allow-root
	
