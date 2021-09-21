#!/bin/bash -eu

#server="ds-teaa-002.dsicv.upv.es"
server="teaa-master-cluster.dsicv.upv.es"

no_force="n"
additional_options=""

sense="none"

case $0 in
    *get_from_master-cluster.sh)
        sense="from-master"
        ;;
    *to_master-cluster.sh)
        sense="to-master"
        ;;
esac

while [ $# -ge 1 ]
do
    case $1 in
        --force)
            no_force=""
            ;;
        --delete)
            additional_options="${additional_options} --delete"
            ;;
        *)
            ;;
    esac
    shift
done


mkdir -p examples spark hadoop bin docs scripts

for dir in bin docs scripts
do
    echo ""
    echo "######################################### ${dir}"
    if [ ${sense} = "from-master" ]
    then
        rsync -avHC${no_force} ${additional_options} ubuntu@${server}:teaa/${dir} .
    elif [ ${sense} = "to-master" ]
    then
        rsync -avHC${no_force} ${additional_options} ${dir} ubuntu@${server}:teaa/
    else
        echo "PANIC"
        exit 1
    fi
done

pushd examples

for dir in python scripts
do
    echo ""
    echo "######################################### ${dir}"
    if [ ${sense} = "from-master" ]
    then
        rsync -avHC${no_force} ${additional_options} ubuntu@${server}:teaa/examples/${dir} .
    elif [ ${sense} = "to-master" ]
    then
        rsync -avHC${no_force} ${additional_options} ${dir} ubuntu@${server}:teaa/examples/
    else
        echo "PANIC"
        exit 1
    fi
done


popd

pushd spark

for dir in conf
do
    echo ""
    echo "######################################### ${dir}"
    if [ ${sense} = "from-master" ]
    then
        rsync -avHC${no_force} ${additional_options} ubuntu@${server}:teaa/cluster/spark/${dir} .
    elif [ ${sense} = "to-master" ]
    then
        rsync -avHC${no_force} ${additional_options} ${dir} ubuntu@${server}:teaa/cluster/spark/
    else
        echo "PANIC"
        exit 1
    fi
done

popd

pushd hadoop

for dir in etc
do
    echo ""
    echo "######################################### ${dir}"
    if [ ${sense} = "from-master" ]
    then
        rsync -avHC${no_force} ${additional_options} ubuntu@${server}:teaa/cluster/hadoop/${dir} .
    elif [ ${sense} = "to-master" ]
    then
        rsync -avHC${no_force} ${additional_options} ${dir} ubuntu@${server}:teaa/cluster/hadoop/
    else
        echo "PANIC"
        exit 1
    fi
done

popd
