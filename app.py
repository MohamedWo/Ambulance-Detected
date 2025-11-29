# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image

# # -----------------------------
# # Load Model
# # -----------------------------
# @st.cache_resource
# def load_model():
#     model = tf.keras.models.load_model("ambulance_cnn_small.h5")
#     return model

# model = load_model()

# # -----------------------------
# # Streamlit UI
# # -----------------------------
# st.title("🚑 Ambulance Detection App")
# st.write("ارفع صورة وسوف يخبرك النموذج هل هي سيارة إسعاف أم لا.")

# uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # عرض الصورة
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # تجهيز الصورة للنموذج
#     img = image.resize((64, 64))
#     img_array = np.array(img)
#     img_array = img_array / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     # التنبؤ
#     prediction = model.predict(img_array)[0][0]

#     # -----------------------------
#     # النتيجة
#     # -----------------------------
#     st.subheader("🔍 Result:")

#     if prediction > 0.5:
#         st.success("🚑 **Ambulance Detected!**")
#     else:
#         st.error("🚗 **Not an Ambulance**")



# # python -m streamlit run app.py



import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("ambulance_cnn_small.h5")
    return model

model = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🚑 Ambulance Detection App")
st.write("ارفع صورة وسوف يخبرك النموذج هل هي سيارة إسعاف أم لا.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # عرض الصورة
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # تجهيز الصورة للنموذج
    img = image.convert("RGB")  # force 3 channels
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # debug: عرض الأبعاد للتأكد
    st.write("Model input shape:", model.input_shape)
    st.write("Image array shape:", img_array.shape)

    # التنبؤ
    prediction = model.predict(img_array)[0][0]

    # -----------------------------
    # النتيجة
    # -----------------------------
    st.subheader("🔍 Result:")

    if prediction > 0.5:
        st.success("🚑 **Ambulance Detected!**")
    else:
        st.error("🚗 **Not an Ambulance**")
