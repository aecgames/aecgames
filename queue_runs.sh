for i in 1 2 3
do
	nohup python parameterSharingPursuit.py PPO > "unpruned_pursuit_PPO_${i}.out" &
	process_id_0=$!
	wait $process_id_0
done


