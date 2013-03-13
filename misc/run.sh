#!/bin/bash
rm samples.txt;
cd images;
files=`ls`;
for f in $files
	do
	../SamplesGathering $f ../samples.txt;
done
