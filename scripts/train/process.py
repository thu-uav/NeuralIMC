# Let's load the uploaded file to examine its contents and extract the necessary data
# for creating the LaTeX table as described.

file_path = 'sss_datt_circle.log'
algo_name = 'datt' #file_path.split('.')[0].split('_')[1]

# Reading the file contents
with open(file_path, 'r') as file:
    content = file.readlines()

import re
from collections import defaultdict
import numpy as np

# Dictionary to store parsed data in the form {(algo, task): {metric: [values across seeds]}}
data = defaultdict(lambda: defaultdict(list))

# Pattern to match the header line with algo, task, and seed information
header_pattern = re.compile(r"(\w+)_track_(\w+)_seed(\d+)")

# Iterate over lines in content
current_algo, current_task, current_seed = None, None, None
for line in content:
    line = line.strip()  # Remove any surrounding whitespace/newlines

    if not line:  # Empty line signifies end of a block
        current_algo, current_task, current_seed = None, None, None
        continue

    # Check if the line is a header line
    header_match = header_pattern.match(line)
    if header_match:
        current_algo, current_task, current_seed = header_match.groups()
        continue

    # Parse metric lines
    if current_algo and current_task:
        parts = line.split()
        metric_name = parts[0]  # First part is the metric name
        value = float(parts[-1])  # Last part is the numeric value

        # Store the value under the respective metric for the (algo, task) pair
        data[(current_algo, current_task)][metric_name].append(value)

# Calculate averages across seeds for each metric for each (algo, task) pair
averages = {}
for (algo, task), metrics in data.items():
    averages[(algo, task)] = {metric: np.mean(values) for metric, values in metrics.items()}
    print(task, averages[(algo, task)])

# exit()

# Preparing the LaTeX table string
latex_table = "\\begin{table}[h!]\n\\centering\n\\begin{tabular}{|c|c|c|c|c|c|c|}\n\\hline\n"
latex_table += "Algorithm & Task & Train Error & Train L1AC Error & Train NoWind Error & Eval Random Error & Eval Random L1AC Error & Eval Random NoWind Error \\\\\n\\hline\n"

train_line = []
eval_line = []

print(averages.keys())

# for (algo, task), avg_metrics in averages.items():
    # Insert row for each (algo, task) with the computed averages
    # latex_table += f"{algo} & {task} & {avg_metrics.get('train', 0):.6f} & {avg_metrics.get('train_l1ac', 0):.6f} & "
    # latex_table += f"{avg_metrics.get('train_nowind', 0):.6f} & {avg_metrics.get('eval_random', 0):.6f} & "
    # latex_table += f"{avg_metrics.get('eval_random_l1ac', 0):.6f} & {avg_metrics.get('eval_random_nowind', 0):.6f} \\\\\n"

all_tasks = ["circle", "poly", "star", "zigzag"]
# all_tasks = ['n0', 'n002', 'n004', 'n008', 'n016']
for ttt in all_tasks:
    temp_line = []
    # try:
    nowind = float("{:.3f}".format(averages[(algo_name, ttt)].get('train_nowind_l1ac', 0)))
    wind = float("{:.3f}".format(averages[(algo_name, ttt)].get('train_l1ac', 0)))
    pert = (wind - nowind) / nowind * 100.
    train_line += [f"{nowind:.3f}", "\\cellcolor{cyan!10}" + f"{wind:.3f} (+{pert:.2f}\%)"]
    
    nowind = float("{:.3f}".format(averages[(algo_name, ttt)].get('eval_random_nowind_l1ac', 0)))
    wind = float("{:.3f}".format(averages[(algo_name, ttt)].get('eval_random_l1ac', 0)))
    pert = (wind - nowind) / nowind * 100.
    eval_line += [f"{nowind:.3f}", "\\cellcolor{cyan!10}" + f"{wind:.3f} (+{pert:.2f}\%)"]
    # except:
    #     print(ttt, averages[(algo_name, ttt)])
    #     import pdb; pdb.set_trace()
    print(" &".join(temp_line))

latex_table += "\\hline\n\\end{tabular}\n\\caption{Average Error Metrics for Each Algorithm and Task Pair}\n\\end{table}"

# latex_table
print(" &".join(train_line))
print(" &".join(eval_line))
