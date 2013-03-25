#!/bin/bash
outfile="../samples.txt"
if [ $# -gt 0  ]
	then
	outfile=$1
fi
rm $outfile;
cd images;
files=`ls`;
cd ..
for f in $files
	do
	./SamplesGathering images/$f $outfile;
done
