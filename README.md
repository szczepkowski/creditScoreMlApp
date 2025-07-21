ML SCORING MODEL

FAQ : To start app api : 

pip install -r requirements.txt

uvicorn main:app --reload  

<img width="1383" height="715" alt="image" src="https://github.com/user-attachments/assets/a09eab7b-808b-4e73-bf7f-be0d74ced056" />



Valuable Features was found.

Data was cleaned and prepared for ml model.

LinearRegression gave only ROC 0.75. as first attempt.

XgBOOST gave ROC = 0.85 ROC.

AUC: 0.8596877992618432
Confusion Matrix:
 [[25158  6429]
 [  535  1755]]
Classification Report:
               precision    recall  f1-score   support

           0       0.98      0.80      0.88     31587
           1       0.21      0.77      0.34      2290

    accuracy                           0.79     33877
   macro avg       0.60      0.78      0.61     33877
weighted avg       0.93      0.79      0.84     33877

<img width="744" height="457" alt="image" src="https://github.com/user-attachments/assets/f8f4edd4-8892-48de-bc8c-c49a0082e205" />


cid:4EB7ED0A-C034-4EB6-A66D-E61370162F06<img width="850" height="455" alt="image" src="https://github.com/user-attachments/assets/4faa4c50-749a-4a90-999b-0da33da8abca" />
