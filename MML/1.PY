import csv

# Load data from CSV
with open('enjoysport.csv', 'r') as csvfile:
    data = list(csv.reader(csvfile))

# Initialize the hypothesis
num_attributes = len(data[0]) - 1
hypothesis = ['0'] * num_attributes

# Update the hypothesis based on the training data
for instance in data:
    if instance[num_attributes] == 'yes':
        for j in range(num_attributes):
            hypothesis[j] = instance[j] if hypothesis[j] in ['0', instance[j]] else '?'

# Print the results
print("Data:", data)
print("Total instances:", len(data))
print("Final hypothesis:", hypothesis)
