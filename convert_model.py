
# Load đúng model từ .pkl
import pickle
with open('best_model.pkl', 'rb') as file:
    best_model = pickle.load(file)

# best_model là keras model, save sang h5
best_model.save('best_model.h5')

