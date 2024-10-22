import streamlit as st
import joblib
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F

# Set up image size and class mapping
image_size = 224
cat_to_name = {
    '1': 'Ant',
    '2': 'Bee',
    '3': 'Bettle',
    '4': 'Butterfly',
    '5': 'Dragonfly',
    '6': 'Fly',
    '7': 'Grasshoper',
    '8': 'Ladybug',
    '9': 'Mosquito',
    '10': 'Spider',
    '11': 'Wasp'
}

# Load model
restnet_model = models.resnet18().cpu()
restnet_model.fc = torch.nn.Linear(512, 11)
restnet_model.load_state_dict(torch.load("restnet_trained.pth", map_location=torch.device('cpu')))
restnet_model.eval()

# App layout
st.set_page_config(page_title="Insect Classification Web")

st.title("Insect Classification Web")
st.subheader("Upload an image of an insect and let our model predict its species!")

st.sidebar.title("Instructions")
st.sidebar.write("1. Upload a JPG, PNG, or JPEG image.")
st.sidebar.write("2. The model will predict the insect species.")
st.sidebar.write("""3. Species that the model can classify: 
                Ant, Bee, Bettle, Butterfly, Dragonfly, Fly, Grasshoper, Ladybug, Mosquito, Spider, Wasp.""")

# File uploader
img = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if img is not None:
    st.image(img, caption="Uploaded Image", use_column_width=True, clamp=True)
    
    # Process the image
    img = Image.open(img)
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    img_tensor = preprocess(img).unsqueeze(0).cpu()

    # Make prediction
    with torch.no_grad():
        output = restnet_model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        percentages = probabilities * 100
        predicted_class = torch.argmax(probabilities).item()
        predicted_label = cat_to_name[str(predicted_class + 1)]  

    # Display results
    st.markdown("### Prediction Result")
    st.success(f"This is a **{predicted_label}**!")
    st.write(f"Confidence: **{percentages[0][predicted_class]:.2f}%**")
    
    # Display percentages for all classes
    st.markdown("### Class Probabilities")
    for i in range(len(cat_to_name)):
        st.write(f"{cat_to_name[str(i + 1)]}: **{percentages[0][i].item():.2f}%**")

# To run the localhost:
# python -m streamlit run app.py
