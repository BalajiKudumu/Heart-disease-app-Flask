import requests
url='http://localhost:5000/predict_api'
r=requests.post(url,json={"age":2,"sex":9,'cp':6,'trestbps':145,'chol':233,'fbs':1,'restecg':0,'thalach':150,'exang':0,'oldpeak':2.3,'slope':0,'ca':0,'thal':1})
print(r.json())