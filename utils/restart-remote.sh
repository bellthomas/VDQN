#!/bin/bash
cd ~/VDQN
tmux kill-server
git pull origin master
tmux new-session -d -s evaluation
tmux send -t evaluation "source env/bin/activate; python3 drive.py" ENTER