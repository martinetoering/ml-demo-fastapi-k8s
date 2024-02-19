import requests
import streamlit as st


def main():
    """Simple streamlit UI that uses FastAPI for video prediction."""
    st.title("Video Prediction on Kinetics-400")
    uploaded_file = st.file_uploader("Choose a video", type=["mp4"])

    if uploaded_file:
        st.video(uploaded_file, format="video/mp4")

        if st.button("Predict Action"):
            # Make a request to the FastAPI app
            files = {"file": uploaded_file}
            response = requests.post(
                "http://127.0.0.1:8000/predict/",
                files=files,
                timeout=10,
            )

            if response.status_code == 200:
                prediction = response.json()
                text = f"{prediction.get('label')} with score {prediction.get('score')}"
                st.markdown(text.capitalize())
            else:
                st.error("Error performing video prediction.")


if __name__ == "__main__":
    main()
