import streamlit as st
import cv2
import numpy as np

# Function to load and preprocess the image
def load_image(image_file):
    img = cv2.imread("C:/Users/HP/Pictures/IMG_6707.JPG")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    return img_rgb

# Function to detect faces and draw rectangles
def detect_faces(image, scaleFactor, minNeighbors, rectangle_color):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

    for (x, y, w, h) in faces:
        # Convert rectangle_color from HTML color format (#RRGGBB) to BGR
        bgr_color = tuple(int(rectangle_color[i:i+2], 16) for i in (1, 3, 5))
        cv2.rectangle(image, (x, y), (x+w, y+h), bgr_color, 2)
    
    return image

# Streamlit app
def main():
    st.title("Vic Face Detection App with Viola-Jones Algorithm")

    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert the file to an opencv image
        image = load_image(uploaded_file)
        
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Parameters for face detection
        scaleFactor = st.slider("Scale Factor", 1.1, 2.0, 1.2, 0.1)
        minNeighbors = st.slider("Min Neighbors", 1, 10, 5)
        rectangle_color = st.color_picker("Rectangle Color", "#00ff00")  # Default color is green
        
        # Detect faces on button click
        if st.button("Detect Faces"):
            detected_image = detect_faces(np.copy(image), scaleFactor, minNeighbors, rectangle_color)
            st.image(detected_image, caption="Detected Faces", use_column_width=True)
        
        # Save image with detected faces
        if st.button("Save Image"):
            file_name = "detected_faces_image.jpg"
            cv2.imwrite(file_name, cv2.cvtColor(detected_image, cv2.COLOR_RGB2BGR))
            st.success(f"Image saved as {file_name}")

if __name__ == "__main__":
    main()
