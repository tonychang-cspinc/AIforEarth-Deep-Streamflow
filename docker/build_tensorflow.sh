#!/bin/bash
TAG=latest
IMAGENAME=cspinc/deep-streamflow
docker build --no-cache \
	-t $IMAGENAME:$TAG -f Dockerfile-tensorflow2 .
