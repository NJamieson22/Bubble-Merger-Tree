#!/bin/sh
mkdir -p bubble_tree


echo Starting...
args="$*"
cmd="./tree $args"

$cmd
