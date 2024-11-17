task=("track_zigzag" "track_poly")
c="neuralimc"
wind=(true)
quadrotor=("train_random")

seed=(0 1 2)
noise=(0.02 0.04 0.08 0.16)

for t in "${task[@]}"; do
for s in "${seed[@]}"; do
for quad in "${quadrotor[@]}"; do
for n in "${noise[@]}"; do

python run_quadrotor.py controller=neuralimc quadrotor=$quad task=$t wind.enable=true seed=$s wandb.project=torchctrl_recycle_ablation_sim pred_model_noise=$n

done
done
done
done