from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.preprocessing import StandardScaler 
import pickle

scaler = joblib.load('model/scaler.pkl')
model = pickle.load(open("model/model.pkl",'rb'))
with open("model/cluster_results.pkl", "rb") as f:
    loaded_data = pickle.load(f)


cluster_targets = loaded_data["cluster_targets"]
career_targets_dict = loaded_data["career_targets_dict"]
cluster_career_map = loaded_data["cluster_career_map"]

jobs_name = ['Lawyer', 'Doctor', 'Government Officer', 'Artist', 'Software Engineer', 'Teacher',
                          'Business Owner', 'Scientist', 'Banker', 'Writer', 'Accountant', 'Designer', 'Construction Engineer',
                          'Game Developer', 'Stock Investor', 'Real Estate Developer']


def recommendations(gender, part_time_job, absence_days, extracurricular_activities, weekly_self_study_hours, math_score, history_score,
                    physics_score, chemistry_score, biology_score, english_score, geography_score):
  # Encode Input Data
  gender_encode = 0
  if gender == "male":
    gender_encode = 0
  else:
    gender_encode = 1

  part_time_job_encode = 0
  if part_time_job == "True":
    part_time_job_encode = 1
  else:
    part_time_job_encode = 0

  extracurricular_activities_encode = 0
  if extracurricular_activities == "True":
    extracurricular_activities_encode = 1
  else:
    extracurricular_activities_encode = 0

  average_score = (math_score + history_score + physics_score + chemistry_score + biology_score + english_score + geography_score)/7

  input_array = [gender_encode, part_time_job_encode, absence_days, extracurricular_activities_encode, weekly_self_study_hours, math_score, history_score,
                physics_score, chemistry_score, biology_score, english_score, geography_score, average_score]
  # Scale Data from 0-1
  
  scaled_input = scaler.transform([input_array])

  # Predict and take top 5 probability results
  prediction = model.predict_proba(scaled_input)
  top_poss_index = np.argsort(prediction[0])[-5:]
  top_poss_probs = [(jobs_name[idx], f"{prediction[0][idx] * 100:.2f}%") for idx in top_poss_index]
  return top_poss_probs


def get_subject_profile_by_job(job_name, cluster_targets, career_targets_dict, jobs_name):
    if job_name not in jobs_name:
        return f"Job name '{job_name}' not found."

    cluster_id = jobs_name.index(job_name)

    # Look up the subject profile from the dict
    subject_profile = cluster_career_map.get(cluster_id)

    if subject_profile is None:
        return f"No subject profile found for cluster ID {cluster_id}."
    else:
        subject_profile = career_targets_dict.get(subject_profile)      

    return {
        "job_name": job_name,
        "cluster_id": cluster_id,
        "subject_profile": subject_profile
    }
    
def target_gap(cluster_career_map, career_targets_dict, target, math_score, history_score, physics_score, chemistry_score, biology_score, english_score, geography_score):
    input_array = [math_score, history_score, physics_score, chemistry_score, biology_score, english_score, geography_score]
    cluster_id = jobs_name.index(target)
    subject_profile_key = cluster_career_map.get(cluster_id)
    subject_profile = career_targets_dict.get(subject_profile_key)
    
    
    result_table = []
    for mark, (subject, target_score) in zip(input_array, subject_profile.items()):
        status = "Good enough" if mark >= target_score else f"Improve by {target_score - mark:.2f}"
        result_table.append({
            'subject': subject,
            'score': mark,
            'target': target_score,
            'status': status
        })
    return result_table



app = Flask (__name__)
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/input')
def input():
    return render_template('input.html')
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        gender = request.form['gender']
        part_time_job = request.form['part_time_job'] == 'true'
        absence_days = int(request.form['absence_days'])
        extracurricular_activities = request.form['extracurricular_activities'] == 'true'
        weekly_self_study_hours = int(request.form['weekly_self_study_hours'])
        math_score = float(request.form['math_score'])
        history_score = float(request.form['history_score'])
        physics_score = float(request.form['physics_score'])
        chemistry_score = float(request.form['chemistry_score'])
        biology_score = float(request.form['biology_score'])
        english_score = float(request.form['english_score'])
        geography_score = float(request.form['geography_score'])
        
        example = recommendations(gender, part_time_job, absence_days,
                          extracurricular_activities, weekly_self_study_hours,
                          math_score, history_score, physics_score, chemistry_score,
                          biology_score, english_score, geography_score)        
        return render_template('predict.html', recommendations=example)
    return render_template('input.html')
  
@app.route('/target', methods = ['GET','POST'])
def target ():
  if request.method == 'POST':
    if 'target_career' not in request.form:
            return "Missing 'target_career' in form data", 400
          
    target_career = str(request.form['target_career'])
    math_score = float(request.form['math_score'])
    history_score = float(request.form['history_score'])
    physics_score = float(request.form['physics_score'])
    chemistry_score = float(request.form['chemistry_score'])
    biology_score = float(request.form['biology_score'])
    english_score = float(request.form['english_score'])
    geography_score = float(request.form['geography_score'])
    
    job_profile = get_subject_profile_by_job(target_career, cluster_targets, career_targets_dict, jobs_name)
    
    result = target_gap(cluster_career_map, career_targets_dict, target_career, math_score, history_score,
               physics_score, chemistry_score, biology_score, english_score, geography_score)
    return render_template('target.html', result=result, career=target_career)
    
  return render_template('target.html', result=None)
if __name__ == '__main__':
    app.run(debug= True)