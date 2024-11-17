task=("track_zigzag" "track_poly")
quadrotor=("train_random")

seed=(0 1 2)
short=(1 5 10)
long=(10 25 50)

for t in "${task[@]}"; do
for s in "${seed[@]}"; do
for quad in "${quadrotor[@]}"; do

for n in "${short[@]}"; do
python run_quadrotor.py controller=ours quadrotor=$quad task=$t wind.enable=true seed=$s wandb.project=torchctrl controller.obs.short_history_len=$n
done

for n in "${long[@]}"; do
python run_quadrotor.py controller=ours quadrotor=$quad task=$t wind.enable=true seed=$s wandb.project=torchctrl controller.obs.long_history_len=$n
done


done
done
done