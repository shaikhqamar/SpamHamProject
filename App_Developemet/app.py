import streamlit as st
import pickle




class MyApp:
    # Create a custom CSS style
    custom_css = """
    <style>
    .container {
    display: flex;
    justify-content: center;
    align-items: center;
    }
    </style>
    """
    # Funtion to preprocess the Text
    def preprocess(text):

        import re
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()

        text = re.sub(r'\bSubject\b', '', text, flags=re.IGNORECASE)

        # replacing urls with empty str
        text = re.sub(r'www.\S+|https?://\S+', ' ', text)

        # Removing special characters and digits
        text = re.sub("[^a-zA-Z]", " ", text)

        # Remove extra spaces (optional)
        text = re.sub(r'\s+', ' ', text).strip()

        # change sentence to Lower case
        text = text.lower()

        # tokenize into words
        tokens = text.split()

        # remove stop words
        clean_tokens = [lemmatizer.lemmatize(token) for token in tokens if not token in stopwords.words("english")]
        return " ".join(clean_tokens)

    def __init__(self):

        self.user_input = ''
        st.image(r"E:\PyCharmProjects\nlp_app\Static\logo\innomatics_logo.webp", caption="")
        st.title('NLP Project App')
        self.menu()

    def menu(self):
        st.markdown("This is a small project app besed on NLP which perform some basic task such as Spam/Ham Detection, Extraction of valueable information from txt/pdf files and analysizing about public mood/trends/openion on social meadia about different topics")
        st.write('''
        1. **Spam/Ham Prediction:** This project involves developing a classification model to identify whether a given text message or email is spam (unwanted) or ham (legitimate).The model can effectively distinguish between the two categories, helping users filter out unwanted communications.
        \n2. **Named Entity Extraction:** This project focuses on identifying and categorizing key entities within a text, such as names of people, organizations, locations, dates, and more. The goal is to extract valuable information for applications such as information retrieval, knowledge graphs, and content summarization.
        \n3. **Sentiment Analysis:** This project aims to determine the sentiment expressed in a piece of text—whether it is positive, negative, or neutral. The project analyzes user opinions, reviews, or social media posts to gauge public sentiment and inform decision-making processes.
        ''')

        self.option = st.selectbox(
            "**Choose Your Task Here**",
            ['Click To Choose',"Spam/Ham Prediction", 'Named Entity Extraction', 'Sentiment Analysis'],
            index=0,
            format_func=lambda x: x.replace("_", " ")
        )
        if self.option == 'Click To Choose':
            pass
        elif self.option == "Spam/Ham Prediction":
            self.data_input_method(self.user_input)
        elif self.option == "Named Entity Extraction":
            self.fileanalyzer = TxtPdfAnalyzer()
            st.write('Project is under work')
        elif self.option == 'Sentiment Analysis':
            st.write('The Project is in progress: You can choose Spam/Ham Prediction.')





    def data_input_method(self,user_input=''):

        self.text_input = st.text_area(
            value=None,
            label="",
            placeholder="Type or Paste your Message or Email here to check whether it's a Spam/Ham",
            )

        # Inject the CSS with st.markdown
        st.markdown(MyApp.custom_css, unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="container">', unsafe_allow_html=True)
            st.subheader('OR')
            st.markdown('</div>',unsafe_allow_html=True)


        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is None:
            st.write('Please upload a image file')
            pass
        else:
            import time
            with st.spinner(text='Fetching text from image...'):
                time.sleep(2)
            st.success("Fetched")
            from PIL import  Image
            import pytesseract

            img  = Image.open(uploaded_file)

            # Use Tesseract to do OCR on the image
            self.text_input = pytesseract.image_to_string(img)
        if st.button('Submit'):
            self.process_text(self.text_input)
#            clear_inputs(self.text_input,uploaded_file)






    def process_text(self,text):
        if text:
            st.write(f"Text received ")

            processed_text = MyApp.preprocess(text)

            import time
            with st.spinner(text='Processing your text...'):
                time.sleep(3)
            st.success("Done!")

            # Load the CountVectorizer
            with open(r'E:\PyCharmProjects\nlp_app\CountVectorizer.pkl', 'rb') as file:
                cv = pickle.load(file)

            vector_inputs = cv.transform([processed_text])  # Transform text to feature vector
            self.spam_ham(vector_inputs)  # Call the spam/ham prediction method
    def spam_ham(self, vector_inputs):
        # Load the trained model for spam/ham prediction
        with open(r'E:\PyCharmProjects\nlp_app\randomforest.pkl', 'rb') as file:
            model = pickle.load(file)
            result = model.predict(vector_inputs)[0]  # Predict the class
            # Display the result based on the prediction
            if result == 0:
                st.markdown('<span style="font-size: 24px;">This message/email is:</span> <span style="color: red; font-size: 24px;">SPAM</span>',unsafe_allow_html=True)

            else:
                st.markdown('<span style="font-size: 24px;">This message/email is:</span> <span style="color: green; font-size: 24px;">HAM</span>',unsafe_allow_html=True)





class TxtPdfAnalyzer:
    def __init__(self):
        st.header('this is Text pdf analyzer class')

if __name__ == "__main__":
    app = MyApp()
