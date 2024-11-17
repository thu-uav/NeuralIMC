tasks=("track_poly" "track_zigzag")
ctrl=("ppo" "datt" "rma" "ours" "mppi_resrl")
wind=(true)
quadrotor=("train_random")
seed=(0 1 2 3 4)

for s in "${seed[@]}"; do
for quad in "${quadrotor[@]}"; do
for w in "${wind[@]}"; do
for c in "${ctrl[@]}"; do
for t in "${tasks[@]}"; do

python run_quadrotor.py controller=$c quadrotor=$quad task=$t wind.enable=$w seed=$s wandb.project=torchctrl

done
done
done
done
done