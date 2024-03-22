import pickle
with open('data.pickle', 'rb') as f:
    data = pickle.load(f)

# Now you can use the data object as it was originally created
print(data)