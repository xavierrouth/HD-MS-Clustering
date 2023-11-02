from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

true_labels = []
predicted_labels = []

# Read the true and predicted labels from file
with open("out.txt", "r") as f:
    for line in f:
        true_label, predicted_label = map(int, line.strip().split())
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

# Calculate Normalized Mutual Information Score
nmi_score = normalized_mutual_info_score(true_labels, predicted_labels)
print(f"Normalized Mutual Information Score: {nmi_score}")

# Calculate Adjusted Mutual Information Score
ami_score = adjusted_mutual_info_score(true_labels, predicted_labels)
print(f"Adjusted Mutual Information Score: {ami_score}")