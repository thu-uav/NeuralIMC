# Neural Internal Model Control

## <font><div align='center' > [[📜 arXiv Paper](https://github.com/thu-uav/NeuralIMC)]  [[📹 Supplementary Video](https://www.youtube.com/watch?v=7MChzWLqbZk&ab_channel=FengGao)] </div> </font>

![Overview of Neural-IMC](assets/overview.png)

---
## TODO

- [ ] Clean codes and update README
- [ ] Codes for experiments on quadrupeds

## Installation


```bash
conda create -n torchctrl python=3.9
cd torch_control
conda activate torchctrl
pip install -e .
```

## Trajectory examples used for experiments

| Trajectory Type | Description | 3D Trajectory | Per-axis Trajectory | Implementation |
|----------------|-------------|------------|------------|------------|
| Circle         | Circular trajectory | ![Circle](torch_control/tasks/trajectory/figs/circle.png) | ![Circle](torch_control/tasks/trajectory/figs/circle_xyz.png) | [circle.py](torch_control/tasks/trajectory/circle.py)
| Poly     | Chained polynomial trajectories | ![Poly](torch_control/tasks/trajectory/figs/chainedpoly.png) | ![Poly](torch_control/tasks/trajectory/figs/chainedpoly_xyz.png) | [chained_polynomial.py](torch_control/tasks/trajectory/chained_polynomial.py)
| Star          | 5-pointed star trajectory | ![Star](torch_control/tasks/trajectory/figs/star.png) | ![Star](torch_control/tasks/trajectory/figs/star_xyz.png) | [pointed_star.py](torch_control/tasks/trajectory/pointed_star.py)
| Zigzag          | Zigzag trajectory | ![Zigzag](torch_control/tasks/trajectory/figs/zigzag.png) | ![Zigzag](torch_control/tasks/trajectory/figs/zigzag_xyz.png) | [zigzag.py](torch_control/tasks/trajectory/zigzag.py)


## Usage

### Tuning the controller parameters

```bash
conda activate torchctrl && cd scripts
# By default, it will use wandb to log data, please make sure you have set WANDB_API_KEY in your environment variables.
python run_quadrotor.py
# To run without wandb logging, use the following command:
python run_quadrotor.py wandb.mode=disabled
```
