# Model-Predict-API
Step to use code
1. Clone and Ctrl+Shift+P to setup environment Python, choose venv and then choose requirements.txt to install library
2. Drop your model to folder models and change model_path to your model in main.py
3. Change class_label to the array of all the class your model predict
4. Run file main.py to start server
5. Use postman to test api, default api to the predict api will be: http://localhost:5000/predict
6. Choose Rest API POST, pass your photo in the body with form-data type, key will name 'photo', variable will be the image you want to predict
7. Click Send and it will return the best class it predict
8. Use ngrok for other computer to access your server
9. Done
