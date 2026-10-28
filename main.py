import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field, RootModel
import pandas as pd
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware

model = joblib.load('Mental_Health_Model.pkl')
app = FastAPI()

app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


class StudentData(BaseModel):       #validation model for the input data using Pydantic BaseModel
    age                         : int = Field(..., gt=10, le=100)
    gender                      : Literal['Male', 'Female']
    country                     : str
    academic_level              : Literal['Undergraduate', 'Graduate', 'High School']
    most_used_platform          : Literal['Facebook', 'LinkedIn', 'Instagram', 'Snapchat', 'Twitter','YouTube', 'TikTok', 'LINE', 'KakaoTalk', 'VKontakte', 'WhatsApp', 'WeChat']
    purpose_of_use              : Literal['Networking', 'Education', 'Entertainment', 'News']
    avg_daily_usage_hours       : float = Field(..., ge=0, le=24)
    daily_unlocks               : int = Field(..., ge=0)
    study_hours                 : float = Field(..., ge=0, le=24)
    physical_activity_hours     : float = Field(..., ge=0, le=24)
    sleep_hours_per_night       : float = Field(..., ge=4, le=24)
    stress_level                : Literal['Medium', 'Low', 'Very High', 'High']


class PredictionResponse(BaseModel):  #validation model for the output data using Pydantic BaseModel
    predicted_mental_health_score: float


# class PredictionResponse(RootModel[float]):  #validation model for the output data using Pydantic BaseModel
#     root: float = Field(..., gt= 0, le=10, description="Predicted mental health score")



top_countries = ['Other','India','USA','Canada','Australia','UK','Germany','Mexico','Turkey','France']



@app.get("/")
def greet():
    return {"Welcome bro !!"}

@app.post("/predict", response_model=PredictionResponse)  #response_model is used to validate the output data
def predict(data: StudentData):     #data is an object of StudentData class
    country_group = data.country if data.country in top_countries else "Other"
    input_row = pd.DataFrame([{
        'Age'                       : data.age,
        'Gender'                    : data.gender,
        'Country'                   : data.country,
        'Academic_Level'            : data.academic_level,
        'Most_Used_Platform'        : data.most_used_platform,
        'Purpose_Of_Use'            : data.purpose_of_use,
        'Avg_Daily_Usage_Hours'     : data.avg_daily_usage_hours,
        'Daily_Unlocks'             : data.daily_unlocks,
        'Study_Hours'               : data.study_hours,
        'Physical_Activity_Hours'   : data.physical_activity_hours,
        'Sleep_Hours_Per_Night'     : data.sleep_hours_per_night,
        'Stress_Level'              : data.stress_level,
        'Grouped_country'           : country_group

    }])

    prediction = model.predict(input_row)[0]  #[0] because it returns a list of predictions, we want the first one
    return PredictionResponse(predicted_mental_health_score=round(float(prediction), 2))  #returning the prediction as a Pydantic BaseModel
    # response = PredictionResponse(prediction) ---|
                                                #  |---- as per Pydantic RootModel
    # return response.root ------------------------|


@app.get("/health")
def health_check():
    return {"status": "healthy"}