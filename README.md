# Optimising Student Retention and Educational Outcomes
- Early Identification and Tailored Support in Virtual Learning Using Data Science
  (Presentation Link: https://youtu.be/S7-zdv5pEBM)
  
## Project Overview

This project uses predictive analytics and machine learning to identify at-risk students early on, leveraging the Open University Learning Analytics dataset (OULAD). By examining demographics, virtual learning engagement, and assessment data within the first four weeks, we aim to inform targeted interventions.

This project's value extends beyond higher education to domains like corporate training and lifelong learning, where identifying learners in need of support can boost success and satisfaction rates.

## Business Context

The shift toward online learning, accelerated by advancements in technology and the COVID-19 pandemic, has transformed education. While offering unparalleled flexibility, it has also created challenges for institutions to retain students and ensure academic success. Financial pressures and the growing need for personalised support have highlighted the importance of early intervention strategies.

By applying predictive models, institutions can better understand student behaviours and risk factors, enabling proactive measures to keep students engaged and on track. For the Open University and similar institutions, improving student retention and success not only enhances educational outcomes but also strengthens financial stability and institutional reputation in an increasingly competitive landscape.

## Data Science Approach

Please see below a simple workflow/flowchart outlining our data science pipeline.

- Data Extraction
- Preprocessing and Feature Engineering
- Clustering
- Feature Selection
- Model Training and Evaluation
- Deployment Considerations

<img src="https://github.com/user-attachments/assets/36a4dda3-2bd1-4332-b572-8ec07f049c73" alt="image" width="700"/>

## Dataset Overview  

The dataset used in this project is the Open University Learning Analytics Dataset ([OULAD](https://analyse.kmi.open.ac.uk/open_dataset#description)), which provides detailed information about student interactions with their courses. It contains data from approximately 32,000 students across multiple features, including:  

- **Student Demographics**: Gender, disability status, highest education level, age group, and region.  
- **Virtual Learning Environment (VLE) Interactions**: Detailed logs of students’ online activities, such as clicks on course content, forums, and other learning resources during the first four weeks of their course.  
- **Assessments**: Scores and weights of early assignments, capturing academic performance during the critical initial stages of the course.  
<img width="700" src="https://github.com/user-attachments/assets/708e2229-5fc3-4186-988a-104db243c5ff" alt="image">

The dataset integrates multiple relational tables (e.g., `studentInfo`, `studentVLE`, `assessments`) to maintain granularity at the module-presentation-student level. A binary target variable (`at_risk`) was derived, categorising students into “at-risk” (Withdrawn or Fail) and “not at-risk” (Pass or Distinction) based on their final outcomes.  
<img width="700" src="https://github.com/user-attachments/assets/11b70a0c-c00e-459e-b99a-02c48068d92c" alt="image">

## Exploratory Data Analysis (EDA)

The EDA process provided valuable insights into student engagement patterns, demographics, and academic performance, helping to understand the data and guide feature engineering / selection. Highlights include:

### Target Variable Distribution
The binary target variable (`at_risk`) shows that:
- **At-risk students (Fail or Withdrawn)**: 12,168  
- **Not at-risk students (Pass or Distinction)**: 15,385  
This indicates a slight class imbalance, addressed during model training through techniques like class weighting.

### Demographic Insights
- A significant proportion of at-risk students have lower education levels ("Lower Than A Level").
  <img width="500" src="https://github.com/user-attachments/assets/499e86e0-f4e0-4e92-87e5-b32701cb403e" alt="image">
  
- Gender distribution shows slight variations in engagement and outcomes, with no extreme imbalance.
  <img width="500" src="https://github.com/user-attachments/assets/24f1745e-fc6b-44af-b20b-553ebe7871df" alt="image">

### VLE Engagement
- Features like `vle_avg_engagement_f4w` and `weekly_avg_click_f4w` show considerable variation across students, highlighting differences in online activity levels.
  
  <img width="500" src="https://github.com/user-attachments/assets/539cede3-5190-491c-a084-8fba931c86be" alt="image">
 
- Engagement with specific VLE components, such as forums (`forumng`) and course content (`oucontent`), correlates strongly with performance, reinforcing their importance as predictors.
  
  <img width="400" src="https://github.com/user-attachments/assets/bfad2855-5bb8-40c6-ad9e-64d862d3a409" alt="image">
  
## Feature Engineering

To maximise the predictive power of the model, several feature engineering techniques were applied, focusing on transforming raw data into meaningful predictors. 

### Key Feature Engineering Techniques:
1. **Table Merging**: 
   - Combined multiple tables (e.g., `studentInfo`, `studentVLE`, `studentAssessment`) using primary and foreign keys to ensure all data aligned at the `module-presentation-student` level, maintaining the correct granularity.

2. **Aggregation and Normalisation**:
   - **Engagement Features**: Aggregated VLE data over the first four weeks to calculate metrics such as `vle_avg_engagement_f4w` (active days divided by course length) and `weekly_avg_click_f4w` (average weekly clicks), to capture early signs of potential struggles.
    <img width="500" src="https://github.com/user-attachments/assets/c6c5498f-5625-4f95-8be9-723da620f468" alt="image"> 

   - **Proportions**: Created proportion-based features for each VLE activity (e.g., `oucontent_prop`, `forumng_prop`) by dividing activity clicks by total clicks within the first four weeks.
    <img width="500" src="https://github.com/user-attachments/assets/ff6cfb61-33ce-45b2-a525-0629cffba28f" alt="image"> 

   - **Assessment Scores**: Engineered `assessment1_weighted_score` by multiplying raw scores by assessment weights and normalising to a 0-100 scale.
    <img width="500" src="https://github.com/user-attachments/assets/543cc514-7714-4a4b-8e4d-f316a139e997" alt="image"> 

3. **Handling Categorical Data**:
   - Applied one-hot encoding to categorical variables like `highest_education` and `imd_band`, ensuring compatibility with machine learning models.

4. **Outlier Capping**:
   - Adjusted continuous features such as `num_of_prev_attempts` to reduce the impact of extreme values by capping them at a reasonable threshold.

5. **Clustering**:
   - Introduced cluster membership features (`cluster_1`, `cluster_2`, etc.) from K-Means clustering, reflecting students’ groupings based on demographics, engagement, and performance patterns.
   <img width="700" src="https://github.com/user-attachments/assets/8f6a1707-4efd-480d-bfff-644e8c3396cc" alt="image">
  
  - Elbow method was used to determine the number of clusters:
   <img width="500" alt="image" src="https://github.com/user-attachments/assets/2b8983a2-8c25-4d0a-8607-66d33dd94e6a">

  - Data used include only the static demographic and the first 4 wks’s students data, and the target is not used during the clustering process to prevent data leakage.  
   <img width="700" src="https://github.com/user-attachments/assets/9432f137-0b65-4394-85ec-966d9ea43e8d" alt="image"> 


### Purpose and Value of Clustering:
Clustering features enhance the model’s ability to recommend personalised interventions for students with similar characteristics.
<img width="700" src="https://github.com/user-attachments/assets/927d7130-8e2b-4917-8242-ed9b11619f3a" alt="image">

<img width="700" src="https://github.com/user-attachments/assets/1bd16705-7920-4e73-8ce4-d031ddccef74" alt="image"> 

## Modelling and Evaluation

### Feature Selection
Based on business understanding, trial-and-error, then LASSO regression, the top 25 features were identified for modelling:
#### Course/Assessment Related Features
- assessment1_weight
- assessment1_weighted_score
- cma_tma_weight_ratio

#### VLE Engagement Metrics (Online Activities)
- forumng_prop
- frequent_activity_variety_f4w
- homepage_prop
- oucollaborate_prop
- oucontent_prop
- ouwiki_prop
- quiz_prop
- resource_prop
- subpage_prop
- url_prop
- vle_avg_engagement_f4w
- weekly_avg_click_f4w

#### Student Demographic Information
- cluster_1
- cluster_2
- disability
- gender
- highest_education_grouped_HE Qualification
- highest_education_grouped_Lower Than A Level
- imd_band_group_over 50%
- imd_band_group_up to 50%
- num_of_prev_attempts_capped
- studied_credits

### Model Selection
To address the binary classification problem of identifying at-risk students, four models were selected based on their unique strengths in predictive accuracy, interpretability, and scalability:

1. **Logistic Regression**: A baseline model offering simplicity and interpretability, used as a benchmark.
2. **Decision Tree**: Intuitive and transparent, enabling straightforward analysis of decision paths while performing well on smaller datasets.
3. **LightGBM**: A high-performance gradient boosting model that handles imbalanced datasets effectively and provides robust predictions.
4. **Neural Network (with 2 hidden layers)**: A flexible model capable of capturing complex patterns within the data, particularly useful for nuanced relationships.

### Hyperparameter Tuning
 <img width="500" alt="image" src="https://github.com/user-attachments/assets/8d029114-42fe-4ff6-bb25-33273b760b86">

- A **5-fold cross-validation** strategy was employed to optimise hyperparameters for all models. 
- The evaluation metric prioritised was **F1 Score for class 1 (at-risk)**, as reducing false negatives and false positives was critical to the project’s goals.
- After tuning, the best parameters were used to retrain the models on the full training set, followed by final evaluation on the test set.

### Performance Metrics
The models were assessed using a combination of:
- **F1 Score** (prioritised): Balances precision and recall to capture the model’s effectiveness in identifying at-risk students.
- **Recall**: Measures the ability to identify true at-risk students, crucial for early interventions.
- **Precision**: Evaluates the accuracy of at-risk predictions, ensuring minimal false alarms.
- **ROC AUC**: Reflects the overall ability to distinguish between classes.
- **Training Time**: Assessed for practical deployment considerations.

### Results
The figures below summarise the performance of our four models:

<img width="700" src="https://github.com/user-attachments/assets/a796ece4-b234-4fa0-8405-b35aee684a9a" alt="image"> 

<img width="770" src="https://github.com/user-attachments/assets/e8c3b273-cf5f-43be-8679-daffd2a47daa" alt="image"> 


### Insights and Model Selection
#### Model Insights
- **Logistic Regression (LR)**: Achieves stable generalisation with consistent training and validation scores, fast training times, and linear scalability, making it a reliable baseline with excellent interpretability and ease of deployment.

- **Decision Tree (DT)**: Shows reasonable performance but overfits the training data, with slower scalability compared to LR, making it less suited for generalisation despite its simplicity.

- **LightGBM (LGBM)**: Offers the best overall predictive performance with the highest F1 score and robust handling of increasing data size, despite slightly fluctuating training times. Its ability to improve generalisation as training size grows makes it ideal for production.

- **Neural Network (NN)**: Balances training and validation performance well and captures complex patterns but requires significantly more training time, making it less practical for scenarios with tight computational constraints. 

#### Recommendation
**LightGBM** is the best choice for high performance, while **Logistic Regression** offers a simple and interpretable alternative. Both models balance key priorities of F1 score, scalability, and practicality for early prediction and intervention. 

## Key Findings and Feature Importance
<img width="700" src="https://github.com/user-attachments/assets/30d26a44-37b2-40b9-8aa0-1b7fb4014b05" alt="image"> 

The top features identified by LightGBM are grouped into three categories, providing insights into student success and risk factors:

**Course/Assessment-Related Features:**

- Early assessment performance (`assessment1_weighted_score`, `assessment1_weight`) is critical, with students struggling in initial assessments needing support.
- The balance between computer-marked and tutor-marked assessments (`cma_tma_weight_ratio`) may impact outcomes, suggesting the need for more personalised feedback in automated grading.

**VLE Engagement (Online Activities):**

- Consistent VLE engagement within the first four weeks (`vle_avg_engagement_f4w`, `weekly_avg_click_f4w`) correlates with positive outcomes.
- Engagement with specific VLE activities (`oucontent_prop`, `subpage_prop`, etc.) is important for outcome prediction, suggesting a deeper connection to the learning process.

**Student Demographics:**

- Workload (`studied_credits`) is associated with outcomes, with students needing tailored support to manage their workload effectively.
- Students with lower education levels (`highest_education_grouped_Lower Than A Level`) may require additional academic assistance.

### Key Takeaways
- Early assessment performance and VLE engagement are key predictors, highlighting the importance of monitoring these aspects during the initial weeks of a course.
- Feature importance results guide institutions in identifying and supporting at-risk students through targeted interventions and personalised learning strategies.

These findings demonstrate how data-driven insights can improve educational outcomes by aligning resources with students' needs.

## Implementation and Future Work
<img width="500" src="https://github.com/user-attachments/assets/dcb97017-3450-4bc2-a649-c79929561684" alt="image"> 

### Deployment Considerations
1. **Automated Data Pipeline:**  
   - Set up weekly updates for VLE activity, assessment scores, and demographics to keep predictions timely and actionable.

2. **Dashboard Integration:**  
   - Develop a dashboard to visualise predictions, key feature insights, and clustering profiles, enabling personalised student support strategies.

3. **Model Monitoring:**  
   - Regularly evaluate and retrain models to maintain accuracy and adapt to new data trends while ensuring compliance with data privacy regulations.

### Future Work
1. **Broader Application:**  
   - Explore extending the model to secondary, primary, and early childhood education to support individualised learning plans.

2. **Refined Features:**  
   - Investigate deeper insights from VLE patterns and feature interactions to enhance prediction accuracy.

3. **Long-Term Impact:**  
   - Analyse the effectiveness of interventions driven by the model to improve retention and success rates over time.

## Conclusion

This project highlights the potential of predictive analytics to improve student retention in online learning. Using early-term data, our models achieved an F1 score of 68%, exceeding benchmarks while identifying key predictors like assessment performance and VLE engagement.  

With an automated pipeline, visual dashboards, and model monitoring, the solution is scalable and practical for deployment. Future work will expand its reach to other education levels, ensuring greater impact. This initiative demonstrates the value of data science in creating personalised, effective support systems for students and institutions.

---
## Repository Contents
- **Jupyter Notebooks**:  
  - `oulad_01_preprocess.ipynb`  
  - `oulad_02_model.ipynb`  

- **Datasets**:  
  - **Original Files**:  
    - Six csv files in the 'data_ou' folder (the `studentVle.csv` is not included due to large file size, but can be accessed from Open University's webpage: https://analyse.kmi.open.ac.uk/open_dataset#description)
      
  - **Processed File**:  
    - `oulad_01.csv`

- **Presentation Slides**:
  - `capstone_project_ai_education_2024.11.pptx`
    
- **Technical Documentation**:
  - `Capstone Project Document _ Nov 2024 EMH.docx`

  
---
## About the Author
Emily Huang - Data Science enthusiast with a passion for turning data into actionable insights.

Feel free to connect with me via my [LinkedIn Profile](https://www.linkedin.com/in/emily-huang-3021212a)
