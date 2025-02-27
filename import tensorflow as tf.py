import tensorflow as tf

# ✅ Assuming your trained model is stored in 'model'
model_path = "/Users/mattgutierrez80/Desktop/UDA_Notes/image_selection_model.keras"

# ✅ Save the model in the correct location
model.save(model_path)

print(f"✅ Model saved at: {model_path}")
