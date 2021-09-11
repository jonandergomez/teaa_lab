#!/bin/bash

master="teaa-master-cluster.dsicv.upv.es"

firefox ${master}:50070 ${master}:8088 ${master}:8080  &
