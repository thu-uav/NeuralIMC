# Neural Internal Model Control

## <font><div align='center' > [[📖 Website](https://github.com/thu-uav/NeuralIMC)]  [[📜 arXiv Paper](https://github.com/thu-uav/NeuralIMC)] </div> </font>

![Overview of Neural-IMC](assets/overview.png)

---

## TODO

- [ ] Website and arXiv paper
- [ ] Clean codes and update README
- [ ] Codes for experiments on quadrupeds

## Installation


```bash
conda create -n torchctrl python=3.9
conda activate torchctrl
pip install -e .
```

## Usage

### Tuning the controller parameters

```bash
conda activate torchctrl && cd scripts
# By default, it will use wandb to log data, please make sure you have set WANDB_API_KEY in your environment variables.
python run_quadrotor.py
# To run without wandb logging, use the following command:
python run_quadrotor.py wandb.mode=disabled
```
