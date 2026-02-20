python Run.py DGEV_order_100 FED --device cuda:0 --num_clients 100 --fedavg --divide metis --active_mode adptpolu &
python Run.py DGEV_kt_100 FED --device cuda:0 --num_clients 100 --fedavg --divide metis --active_mode adptpolu 
python Run.py BJEV_order_100 FED --device cuda:0 --num_clients 100 --fedavg --divide metis --active_mode adptpolu &
python Run.py BJEV_kt_100 FED --device cuda:0 --num_clients 100 --fedavg --divide metis --active_mode adptpolu &
python Run.py HZEV_order_100 FED --device cuda:0 --num_clients 100 --fedavg --divide metis --active_mode adptpolu &
python Run.py HZEV_kt_100 FED --device cuda:0 --num_clients 100 --fedavg --divide metis --active_mode adptpolu &
wait

echo "所有任务均已完成！"
