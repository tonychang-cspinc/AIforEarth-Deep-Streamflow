#!/bin/bash

docker run \
	-it --rm \
	-v $(pwd):/content \
	-w /content \
	cspinc/r-base:latest 
	/bin/bash
