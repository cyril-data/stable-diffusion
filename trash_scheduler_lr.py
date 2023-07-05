import torch
# Import your choice of scheduler here
import torch.optim.lr_scheduler as lr_scheduler

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

LEARNING_RATE = 1e-6
EPOCHS = 100
STEPS_IN_EPOCH = 25

print("STEPS_IN_EPOCH", STEPS_IN_EPOCH)

# Set model and optimizer
model = torch.nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Define your scheduler here as described above

base_lr = LEARNING_RATE/10
max_lr = LEARNING_RATE

scheduler = lr_scheduler.CyclicLR(optimizer,
                                  base_lr=base_lr,  # Initial learning rate which is the lower boundary in the cycle for each parameter group
                                  max_lr=max_lr,  # Upper learning rate boundaries in the cycle for each parameter group
                                  # Number of training iterations in the increasing half of a cycle
                                  step_size_up=STEPS_IN_EPOCH * 5,
                                  mode="triangular2")

# Get learning rates as each training step
steps = []
learning_rates = []

for i in range(EPOCHS*STEPS_IN_EPOCH):
    optimizer.step()
    if i % 10 == 0:
        steps.append(i)
        learning_rates.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

# Visualize learinig rate scheduler
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(steps,
        learning_rates,
        marker='o',
        color='black')
ax.set_xlim([0, EPOCHS*STEPS_IN_EPOCH])
ax.set_ylim([0, LEARNING_RATE])
ax.set_xlabel('Steps')
ax.set_ylabel('Learning Rate')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_major_locator(MultipleLocator(STEPS_IN_EPOCH))
ax.xaxis.set_minor_locator(MultipleLocator(1))
plt.show()
