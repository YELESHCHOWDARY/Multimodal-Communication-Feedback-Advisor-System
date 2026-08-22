# Multimodal Communication Feedback Advisor

An AI-powered public speaking assessment system that analyzes verbal and non-verbal communication from recorded videos and provides personalized feedback to help users improve their communication skills.

## Project Overview

The Multimodal Communication Feedback Advisor evaluates public speaking performance using audio and video analysis.

The system analyzes:
- Speech and audio characteristics
- Eye contact
- Posture
- Hand gestures
- Facial expressions
- Speech fluency
- Speaking pace
- Tone
- Confidence

The results are combined to generate an overall communication score and personalized improvement suggestions.

## Features

- User registration and login
- JWT-based authentication
- Browser-based video recording
- Video analysis
- Audio and video preprocessing
- HuBERT-based speech analysis
- MediaPipe-based facial, pose, and hand analysis
- Speech fluency and pace analysis
- Tone and emotion analysis
- Confidence estimation
- Overall communication score
- Personalized suggestions
- Analysis history
- PDF report generation
- MongoDB Atlas integration

## System Workflow

```text
Recorded Video
      |
      v
FFmpeg Processing
      |
  +---+---+
  |       |
Audio   Video
  |       |
  v       v
HuBERT  MediaPipe
          |
     +----+----+
     |    |    |
    Face Pose Hands
     |    |    |
     +----+----+
          |
          v
 Metric Calculation
          |
          v
 Multimodal Scoring
          |
          v
   Overall Score
          |
          v
Personalized Feedback
```

## Technologies Used

### Backend
- Python 3.11
- Flask
- Flask-JWT-Extended
- Flask-CORS

### AI / Machine Learning
- PyTorch
- HuBERT
- Transformers
- MediaPipe
- OpenCV
- Librosa
- SpeechRecognition

### Database
- MongoDB Atlas
- PyMongo

### Media Processing
- FFmpeg
- MoviePy
- Pydub

### Frontend
- HTML5
- CSS3
- JavaScript
- Bootstrap

### Tools
- Git
- GitHub
- VS Code
- ReportLab

## Project Structure

```text
FYP/
├── app.py
├── analysis_pipeline.py
├── requirements.txt
├── .gitignore
├── .env.example
├── best_fluency_model.pt
├── public_speaking_model.pkl
├── Total_model(finalyr_prjct).ipynb
├── playground-1.mongodb.js
├── static/
│   ├── auth_style.css
│   ├── script.js
│   └── style.css
├── templates/
│   ├── history.html
│   ├── home.html
│   ├── index.html
│   ├── login.html
│   ├── practice.html
│   └── signup.html
└── uploads/
```

## Requirements

- Python 3.11
- FFmpeg
- MongoDB Atlas
- Git

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YELESHCHOWDARY/Multimodal-Communication-Feedback-Advisor-System.git
```

### 2. Enter the project directory

```bash
cd Multimodal-Communication-Feedback-Advisor-System
```

### 3. Install dependencies

```bash
py -3.11 -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
MONGODB_URI=mongodb+srv://USERNAME:PASSWORD@YOUR_CLUSTER.mongodb.net/publicspeakingdb
PORT=5000
SECRET_KEY=your_secret_key
```

Never upload `.env` to GitHub. It is excluded using `.gitignore`.

### 5. Verify FFmpeg

```bash
ffmpeg -version
```

### 6. Run the application

```bash
py -3.11 app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Communication Metrics

| Metric | Description |
|---|---|
| Eye Contact | Evaluates eye contact consistency |
| Posture | Evaluates posture stability |
| Hand Gestures | Analyzes hand movement and gestures |
| Emotion | Analyzes facial/emotional characteristics |
| Fluency | Evaluates speech fluency |
| Tone | Analyzes vocal characteristics |
| Pace | Measures speaking rate |
| Confidence | Estimates speaking confidence |
| Overall Score | Combined communication performance |

## Security

Sensitive information is stored in environment variables.

The following files are excluded from Git:

```text
.env
__pycache__/
*.pyc
uploads/
.vscode/
```

Never commit database passwords, API keys, tokens, or other credentials to the repository.

## Objective

The objective of this project is to provide an AI-based platform that helps users improve public speaking and communication skills by automatically analyzing verbal and non-verbal communication.

## Future Enhancements

- Real-time communication feedback
- Improved emotion recognition
- Advanced speech analysis
- Personalized communication training
- Improved multimodal fusion
- Cloud deployment
- Mobile application
- Advanced performance analytics

## Author

**YELESH CHOWDARY**

B.Tech - Artificial Intelligence and Data Science

## License

This project is developed for academic and educational purposes.
