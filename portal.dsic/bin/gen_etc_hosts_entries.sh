#!/bin/bash

for n in {0..30}
do
    if [ $n -eq 0 ]
    then
        worker_name=teaa-master-cluster.dsicv.upv.es
	else
        worker_name=$(printf "teaa-worker%02d-cluster.dsicv.upv.es" $n)
    fi

    address_line=$(host ${worker_name} | grep address)

    hostname=$(echo $address_line | awk '{ print $1 }')
    ipaddr=$(echo $address_line | awk '{ print $NF }')

    echo $ipaddr $(echo ${worker_name} | cut -f1 -d'.') $(echo ${hostname} | cut -f1 -d'.') ${hostname} ${worker_name}
    #ping -c 4 $ipaddr
done
