export XDG_RUNTIME_DIR="/run/user/$UID"
export DBUS_SESSION_BUS_ADDRESS="unix:path=${XDG_RUNTIME_DIR}/bus"

systemd-run --user --unit=pt-train-0 -p StandardOutput=journal -p StandardError=journal /home/james/miniconda3/envs/a4c3d/bin/python /home/james/a4c3d/train.py --local_rank 0 --ngpu 4
systemd-run --user --unit=pt-train-1 -p StandardOutput=journal -p StandardError=journal /home/james/miniconda3/envs/a4c3d/bin/python /home/james/a4c3d/train.py --local_rank 1 --ngpu 4
systemd-run --user --unit=pt-train-2 -p StandardOutput=journal -p StandardError=journal /home/james/miniconda3/envs/a4c3d/bin/python /home/james/a4c3d/train.py --local_rank 2 --ngpu 4
systemd-run --user --unit=pt-train-3 -p StandardOutput=journal -p StandardError=journal /home/james/miniconda3/envs/a4c3d/bin/python /home/james/a4c3d/train.py --local_rank 3 --ngpu 4
