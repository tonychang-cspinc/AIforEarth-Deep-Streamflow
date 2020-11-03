#!/bin/bash
IMAGENAME=cspinc/rstreamflow:latest
DOCKERFILE=Dockerfile-Rstreamflow
docker build -t $IMAGENAME -f $DOCKERFILE .
