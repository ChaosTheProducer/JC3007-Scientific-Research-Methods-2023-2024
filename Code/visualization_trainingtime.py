import matplotlib.pyplot as plt

# Data
methods = ["LoRA", "Prefix Tuning"]
training_times = [16162.1726, 22147.6752]

# Plot
plt.figure(figsize=(8, 6))
plt.bar(methods, training_times, color=['skyblue', 'orange'], width=0.5)

# Labels and Title
plt.ylabel("Training Time (seconds)", fontsize=12)
plt.xlabel("Fine-Tuning Methods", fontsize=12)
plt.title("Training Time Comparison between LoRA and Prefix Tuning*", fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Annotate values on top of bars
for i, time in enumerate(training_times):
    plt.text(i, time + 500, f"{time:.1f}s", ha='center', fontsize=10)

# Show plot
plt.tight_layout()
plt.show()
