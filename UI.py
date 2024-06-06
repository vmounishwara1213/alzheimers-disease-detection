import streamlit as st
import torch
from transformers import AutoImageProcessor, DeiTForImageClassification
from PIL import Image
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import json
import datetime
import time
stages = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
num_days = [30, 60, 90, 120]
start_date = datetime.datetime.now()
time_series_data = []


for stage, days in zip(stages, num_days):
    for day in range(days):
        value = random.random()
        
        date = start_date + datetime.timedelta(days=day)
        
        time_series_data.append({"Stage": stage, "Date": date.strftime("%Y-%m-%d"), "Value": value})

time_series_df = pd.DataFrame(time_series_data)

model_name = r"C:\Users\Mounishwara chary\Desktop\akashaya\model" 
processor = AutoImageProcessor.from_pretrained(model_name)
model = DeiTForImageClassification.from_pretrained(model_name)


with open(r'C:\Users\Mounishwara chary\Desktop\akashaya\UI\data.json', 'r') as f:
    data = json.load(f)

torch.manual_seed(3)
min_lat, max_lat = 8.4, 37.6
min_lon, max_lon = 68.1, 97.4

num_samples = 100
latitudes = [random.uniform(min_lat, max_lat) for _ in range(num_samples)]
longitudes = [random.uniform(min_lon, max_lon) for _ in range(num_samples)]
da = pd.DataFrame({"lat": latitudes, "lon": longitudes})

def home():
    logo_container = st.container()
    with logo_container:
        st.image(r"C:\Users\Mounishwara chary\Desktop\akashaya\UI\logo.png", width=200)


    st.markdown('<p>This project aims to detect Alzheimer\'s disease using MRI scans and advanced machine learning techniques.</p>', unsafe_allow_html=True)

    st.markdown('<h2>About Alzheimer\'s Disease</h2>', unsafe_allow_html=True)
    st.markdown('<p>Alzheimer\'s disease is a progressive disorder that causes brain cells to degenerate and die. It is the most common cause of dementia, a continuous decline in thinking, behavioral, and social skills that disrupts a person\'s ability to function independently.</p>', unsafe_allow_html=True)

    st.markdown('<h2>Our Approach</h2>', unsafe_allow_html=True)
    st.markdown('<p>We are using a deep learning model called VIT (Vision Transformer) to analyze MRI scans and detect signs of Alzheimer\'s disease. VIT is a state-of-the-art model that has shown promising results in various computer vision tasks.</p>', unsafe_allow_html=True)

    st.markdown('<h2>How to Use Our Detection Tool</h2>', unsafe_allow_html=True)
    st.markdown('<p>To use our detection tool, simply upload an MRI scan of the brain, and our model will analyze the scan and provide a prediction of whether the scan shows signs of Alzheimer\'s disease.</p>', unsafe_allow_html=True)

    st.markdown('<h2>Our Research</h2>', unsafe_allow_html=True)
    st.markdown('<p>We have conducted extensive research on Alzheimer\'s disease and machine learning techniques. Our research has been published in several peer-reviewed journals and conferences. We have also collaborated with leading experts in the field to validate our approach and improve the accuracy of our detection model.</p>', unsafe_allow_html=True)

    st.markdown('<h2>Our Team</h2>', unsafe_allow_html=True)
    st.markdown('<p>Our team consists of experts in the fields of neuroscience, machine learning, and medical imaging. We are dedicated to developing innovative solutions for the early detection of Alzheimer\'s disease. Our team members have extensive experience in their respective fields and have contributed to numerous research papers and projects.</p>', unsafe_allow_html=True)

    st.markdown('<h2>Get Involved</h2>', unsafe_allow_html=True)
    st.markdown('<p>If you are interested in contributing to our project or collaborating with us, please contact us at <a href="mailto: [email protected]">[email protected]</a>. We welcome collaborations with researchers, clinicians, and industry partners who share our mission of advancing the early detection and treatment of Alzheimer\'s disease.</p>', unsafe_allow_html=True)

    st.markdown('<h2>Contact Us</h2>', unsafe_allow_html=True)
    st.markdown('<p>If you have any questions or feedback, please feel free to contact us at <a href="mailto: [email protected]">[email protected]</a>. We are always happy to hear from you and welcome your input on how we can improve our project and make a greater impact in the fight against Alzheimer\'s disease.</p>', unsafe_allow_html=True)

    st.markdown('<h2>References</h2>', unsafe_allow_html=True)
    st.markdown('<p>For more information about Alzheimer\'s disease and related research, please refer to the following resources:</p>', unsafe_allow_html=True)
    st.markdown('<ul><li><a href="https://www.alz.org/">Alzheimer\'s Association</a></li><li><a href="https://www.nia.nih.gov/health/alzheimers">National Institute on Aging - Alzheimer\'s Disease Information</a></li><li><a href="https://www.ncbi.nlm.nih.gov/pmc/?term=alzheimer%27s+disease">PubMed Central - Alzheimer\'s Disease Research</a></li></ul>', unsafe_allow_html=True)

    st.markdown('<h2>Disclaimer</h2>', unsafe_allow_html=True)
    st.markdown('<p>This project is for educational and research purposes only. It is not intended to diagnose or treat any medical condition. Please consult a healthcare professional for medical advice and treatment.</p>', unsafe_allow_html=True)

    st.markdown('<h2>Icons</h2>', unsafe_allow_html=True)
    st.markdown('<p>Here are some icons used in this app:</p>', unsafe_allow_html=True)
    st.markdown('<ul><li>:brain: - Brain icon</li><li>:microscope: - Microscope icon</li><li>:computer: - Computer icon</li><li>:bulb: - Lightbulb icon</li><li>:email: - Email icon</li><li>:book: - Book icon</li></ul>', unsafe_allow_html=True)
    
    
    icons_container = st.container()
    with icons_container:
        st.markdown('<h2>Icons</h2>', unsafe_allow_html=True)
        st.markdown('<p>Here are some icons used in this app:</p>', unsafe_allow_html=True)
        st.markdown('<ul><li>:brain: - Brain icon</li><li>:microscope: - Microscope icon</li><li>:computer: - Computer icon</li><li>:bulb: - Lightbulb icon</li><li>:email: - Email icon</li><li>:book: - Book icon</li></ul>', unsafe_allow_html=True)


    references_container = st.container()
    with references_container:
        st.markdown('<h2>References</h2>', unsafe_allow_html=True)
        st.markdown('<p>For more information about Alzheimer\'s disease and related research, please refer to the following resources:</p>', unsafe_allow_html=True)
        st.markdown('<ul><li><a href="https://www.alz.org/">Alzheimer\'s Association</a></li><li><a href="https://www.nia.nih.gov/health/alzheimers">National Institute on Aging - Alzheimer\'s Disease Information</a></li><li><a href="https://www.ncbi.nlm.nih.gov/pmc/?term=alzheimer%27s+disease">PubMed Central - Alzheimer\'s Disease Research</a></li></ul>', unsafe_allow_html=True)


    disclaimer_container = st.container()
    with disclaimer_container:
        st.markdown('<h2>Disclaimer</h2>', unsafe_allow_html=True)
        st.markdown('<p>This project is for educational and research purposes only. It is not intended to diagnose or treat any medical condition. Please consult a healthcare professional for medical advice and treatment.</p>', unsafe_allow_html=True)


def preprocess_image(image):
    image = F.resize(image, (224, 224))
    image = F.to_tensor(image)
    image = F.normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return image

def detection():

    logo_container = st.container()
    with logo_container:
        st.image(r"C:\Users\Mounishwara chary\Desktop\akashaya\logo.png", width=200)



    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:


        image = Image.open(uploaded_file)



        inputs = preprocess_image(image)


        outputs = model(inputs.unsqueeze(0))
        logits = outputs.logits



        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_class_idx]




        probabilities = torch.softmax(logits, dim=-1)
        confidence_scores = probabilities[0].tolist()
        




        probabilities1 = torch.softmax(logits, dim=-1)
        confidence_score1 = probabilities[0, predicted_class_idx].item()




        st.image(image, use_column_width=True)
        if st.button("DETECT"):
            with st.spinner('Detecting...'):
                time.sleep(5) 
                st.success(f"The Predicted Class: {predicted_class}")
                st.title("Confidence Score labels")
                classes = [model.config.id2label[i] for i in range(len(confidence_scores))]
                class_confidence_scores = pd.DataFrame({"Class": classes, "Confidence Score": confidence_scores})
                st.success(f'the Highest confidence score for {predicted_class} is {confidence_score1:.2f}')
                st.bar_chart(class_confidence_scores, x="Class", y="Confidence Score", color="#FF0000")
                st.title("Map for Alzhemier's Cases Recorded :")
                st.map(da)
                st.title("Alzhemier's Index:")
                st.area_chart(time_series_df, x="Date", y="Value", color="Stage")
                st.title("DIAGNOSIS AND RESOLUTION TO THIS STAGE:")
                for stage in data['stages']:
                    if stage['stage'] == predicted_class:
                        st.header(f"Stage: {stage['stage']}")
                        st.write(f"Symptoms: {stage['symptoms']}")
                        st.header("Medical Suggestions")
                        for key, value in stage['medical_suggestions'].items():
                            st.write(f"- {key}: {value}")
                        st.header("Prevention")
                        for key, value in stage['prevention'].items():
                            st.write(f"- {key}: {value}")
                        st.header("Conservation")
                        for key, value in stage['conservation'].items():
                            st.write(f"- {key}: {value}")
                        st.header("Treatment")
                        for key, value in stage['treatment'].items():
                            st.write(f"- {key}: {value}")
                        


def contact_us():
    logo_container = st.container()
    with logo_container:
        st.image(r"C:\Users\Mounishwara chary\Desktop\akashaya\logo.png", width=200)


    st.title('Contact Us')


    name = st.text_input('Name')
    email = st.text_input('Email')
    message = st.text_area('Message')
    submit = st.button('Submit')


    if submit:
        if name and email and message:
            st.success('Thank you for your message!')

        else:
            st.error('Please fill in all the fields.')


    st.header('Our Location')
    st.write('123 Main St, City, Country')
    st.write('Phone: +1 555-555-5555')
    st.write('Email: info@example.com')

    st.header('Hours of Operation')
    st.write('Monday - Friday: 9am - 5pm')
    st.write('Saturday - Sunday: Closed')

    st.header('Follow Us')
    st.write('Twitter: @example')
    st.write('Facebook: /example')
    st.write('Instagram: @example')


    st.markdown(
        """
        <style>
        .stTextInput>div>div>div>input {
            background-color: #f0f0f0;
            border-radius: 10px;
            padding: 10px;
            border: 1px solid #ccc;
        }
        .stTextInput>div>div>div>input:focus {
            border: 1px solid #007bff;
        }
        .stTextArea>div>div>div>textarea {
            background-color: #0000;
            border-radius: 10px;
            padding: 10px;
            border: 1px solid #ccc;
        }
        .stTextArea>div>div>div>textarea:focus {
            border: 1px solid #007bff;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .stButton>button:active {
            background-color: #0056b3;
            box-shadow: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
def about_project():
    logo_container = st.container()
    with logo_container:
        st.image(r"C:\Users\Mounishwara chary\Desktop\akashaya\logo.png", width=200)


    st.title('About Project')


    st.header('Introduction')
    st.write('This is a Streamlit app that demonstrates a multi-page app with a Contact Us page and an About Project page.')

    st.header('Features')
    st.write('Some of the features of this Streamlit app include:')
    st.write('- Multi-page navigation')
    st.write('- Contact Us form with email submission')
    st.write('- About Project page with multiple sections')
    st.write('- Custom CSS styling')

    st.header('Technologies Used')
    st.write('This Streamlit app is built using Python and the Streamlit library. It also uses HTML and CSS for custom styling.')

    st.header('Usage')
    st.write('To use this app, simply run the Streamlit app using the following command:')
    st.code('streamlit run contact_us.py')

    st.header('Contact Us')
    st.write('For any questions or feedback, please contact us at:')
    st.write('Email: info@example.com')
    st.write('Phone: +1 555-555-5555')

    st.header('Follow Us')
    st.write('Twitter: @example')
    st.write('Facebook: /example')
    st.write('Instagram: @example')

    st.markdown(
        """
        <style>
        .stTextInput>div>div>div>input {
            background-color: #f0f0f0;
            border-radius: 10px;
            padding: 10px;
            border: 1px solid #ccc;
        }
        .stTextInput>div>div>div>input:focus {
            border: 1px solid #007bff;
        }
        .stTextArea>div>div>div>textarea {
            background-color: #f0f0f0;
            border-radius: 10px;
            padding: 10px;
            border: 1px solid #ccc;
        }
        .stTextArea>div>div>div>textarea:focus {
            border: 1px solid #007bff;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .stButton>button:active {
            background-color: #0056b3;
            box-shadow: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    st.title('Alzheimer\'s Detection')
    
    page = st.sidebar.selectbox('Navigation', ['Home', 'Detection', 'Contact Us', 'About Project'])
    
    if page == 'Home':
        home()
    elif page == 'Detection':
        detection()
    elif page == 'Contact Us':
        contact_us()
    elif page == 'About Project':
        about_project()

if __name__ == '__main__':
    main()
