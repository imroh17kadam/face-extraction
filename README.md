# Face Extraction

This project is designed using Streamlit where users can upload an image. After processing, the application extracts the face from the image, providing it with a transparent background.

## Demo

Video

## Screenshots

![App Screenshot](images/screenshots/1.png)
![App Screenshot](images/screenshots/2.png)
![App Screenshot](images/screenshots/3.png)
![App Screenshot](images/screenshots/4.png)

# How to run?
### Steps:

Clone the repository

```bash
https://github.com/imroh17kadam/face-extraction
```
### Step 1 - Create a conda environment after opening the repository
#### Using Conda

```bash
conda create -n extraction python=3.9 -y
```

```bash
conda activate extraction
```

#### Using Python

```bash
python3.9 -m venv extraction
```

```bash
.\extraction\Scripts\activate
```

### Step 2 - install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
streamlit run app.py
```