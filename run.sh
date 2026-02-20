# python Run.py DGEV_order FED --device cuda:0 --num_clients 480 --divide metis --embed_dim 6 --num_layers 1 --active_mode adptpolu &
# python Run.py DGEV_kt FED --device cuda:1 --num_clients 480 --divide metis --embed_dim 6 --num_layers 1 --active_mode adptpolu &
# python Run.py BJEV_order FED --device cuda:2 --num_clients 329 --divide metis --embed_dim 6 --num_layers 1 --active_mode adptpolu &
# python Run.py BJEV_kt FED --device cuda:3 --num_clients 329 --divide metis --embed_dim 6 --num_layers 1 --active_mode adptpolu &
# python Run.py HZEV_order FED --device cuda:4 --num_clients 519 --divide metis --embed_dim 6 --num_layers 1 --active_mode adptpolu &
# python Run.py HZEV_kt FED --device cuda:5 --num_clients 519 --divide metis --embed_dim 6 --num_layers 1 --active_mode adptpolu &
# python Run.py HZEV_order FED --device cuda:6 --num_clients 519 --divide metis --embed_dim 8 --num_layers 1 --active_mode adptpolu &
# python Run.py HZEV_kt FED --device cuda:7 --num_clients 519 --divide metis --embed_dim 6 --num_layers 3 --active_mode adptpolu &
# wait

# echo "所有任务均已完成！"

python Run.py DGEV_order_100 FED --device cuda:0 --num_clients 100 --divide metis --embed_dim 6 --num_layers 1 --active_mode adptpolu --epochs 1 &
# python Run.py DGEV_kt_100 FED --device cuda:0 --num_clients 100 --fedavg --divide metis --embed_dim 6 --num_layers 2 --active_mode adptpolu --mask_ratio 0.5 &
# python Run.py BJEV_order_100 FED --device cuda:0 --num_clients 100 --divide metis --embed_dim 6 --num_layers 1 --active_mode adptpolu &
# python Run.py BJEV_kt_100 FED --device cuda:0 --num_clients 100 --divide metis --embed_dim 6 --num_layers 1 --active_mode adptpolu &
# python Run.py HZEV_order_100 FED --device cuda:0 --num_clients 100 --divide metis --embed_dim 6 --num_layers 1 --active_mode adptpolu &
# python Run.py HZEV_kt_100 FED --device cuda:0 --num_clients 100 --divide metis --embed_dim 6 --num_layers 1 --active_mode adptpolu &
# wait

# echo "所有任务均已完成！"
