#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
while read h; do
    ADDR="$h"
    ssh -i ~/Desktop/hc-Default.pem ubuntu@$ADDR "cd ~/VDQN; git pull; ./utils/restart-remote.sh"
    echo $ADDR
done <hosts.txt
