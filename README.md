# SCREAM 

This project is a machine learning-based application designed to detect screams in real-time audio streams. Leveraging MFCC (Mel Frequency Cepstral Coefficients) for feature extraction and models like SVM and MLPClassifier, the system accurately classifies scream and non-scream sounds. The application includes a modern, intuitive UI built with Kivy, offering an alert mechanism that pops up and sends notifications upon detecting screams. This tool is ideal for enhancing security and emergency response systems.

Features:

Real-time scream detection using live microphone input.
High accuracy prediction models trained on scream and non-scream datasets.
Attractive and modern UI with intuitive controls (Kivy framework).
Automated alerts: Pop-up notifications and SMS alerts via integrated services.
Custom model training: Includes SVM and MLP models for flexibility.
Comprehensive testing and validation for accuracy and reliability.
Technology Stack:

Programming Language: Python
Frameworks and Libraries: Scikit-learn, Kivy, Librosa, Twilio (for alerts)
Algorithms: SVM, MLP, MFCC for feature extraction
How to Run:

Clone the repository.
Install required dependencies from requirements.txt.
Run main.py to launch the application.
