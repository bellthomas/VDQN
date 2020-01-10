#!/bin/bash
while read h; do
    ADDR="$h"
    ssh -q -i ~/Desktop/hc-Default.pem ubuntu@$ADDR "cd ~/VDQN; rm -rf logs; git pull; ./utils/restart-remote.sh" >>/dev/null 2>&1 & #& echo $ADDR &
done <hosts.txt
