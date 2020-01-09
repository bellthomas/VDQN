#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
while read h; do
    ADDR="$h"
    mkdir -p $DIR/data/$ADDR
    rsync -avzhq -e "ssh -i ~/Desktop/hc-Default.pem" ubuntu@$ADDR:~/VDQN/logs/ $DIR/data/$ADDR
    echo $ADDR
done <hosts.txt
