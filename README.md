Machine Learning Driven Heart Disease Detection in Patients

Jack Doughty
4 December 2024

     	Abstract
The diagnosis of heart disease often relies on invasive procedures that can be costly, time-consuming, and prone to inaccuracies. Identifying a more accurate and non-invasive diagnostic approach is vital for improving healthcare outcomes. This study leverages publicly available heart disease patient data to explore machine learning as a tool for heart disease detection. Three supervised learning models—Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Naive Bayes—were trained to classify the presence or absence of heart disease. Optuna-based hyperparameter tuning was applied to optimize model performance, with KNN emerging as the best-performing model, achieving 85% accuracy and a high recall of 93%. Additionally, feature importance was analyzed using SHAP values, highlighting key predictors such as chest pain type and maximum heart rate achieved. The study also included three unsupervised learning models—DBSCAN, K-Means Clustering, and t-SNE—to identify patterns within the data. K-Means demonstrated the highest clustering efficiency with a silhouette score of 0.76. Results indicate that machine learning models can provide an effective, non-invasive alternative for early heart disease detection. By prioritizing interpretability and accuracy, this research underscores the potential for integrating machine learning into clinical decision-making to enhance diagnostic accuracy and accessibility.
    
Introduction
Heart disease is one of the leading causes of mortality worldwide, contributing to millions of deaths annually. This condition arises from various interrelated factors, including the consumption of alcohol and tobacco products, unhealthy dietary habits, and a lack of physical activity [1]. As a complex disease with multifaceted origins, heart disease often presents diverse symptoms that range from chest pain and shortness of breath to less obvious signs such as fatigue and dizziness. These symptoms frequently overlap with those of less severe conditions, making early and accurate diagnosis particularly challenging. Traditional diagnostic methods, such as clinical evaluations, stress tests, and invasive procedures like angiography, are indispensable but often suffer from limitations. These methods can be time-consuming, costly, and dependent on significant medical expertise, potentially delaying intervention for high-risk patients. Consequently, the need for more accessible, accurate, and non-invasive diagnostic tools is critical to improving healthcare outcomes [2]. Motivated by these challenges, this project explores the application of machine learning (ML) techniques to develop a robust, efficient, and non-invasive diagnostic solution for heart disease. By analyzing patient health data, including features such as age, cholesterol levels, blood pressure, and other clinical indicators, the project seeks to classify individuals based on their likelihood of having heart disease. Supervised learning models, including Support Vector Machines (SVM), K-Nearest Neighbors (KNN), and Naive Bayes, are employed to achieve this classification. The models are trained and evaluated using k-fold cross-validation to ensure accuracy, minimize overfitting, and enhance generalizability. Additionally, hyperparameter tuning through Optuna is applied to optimize the performance of each model, identifying the most accurate and reliable approach for early diagnosis. To complement these supervised techniques, the project integrates unsupervised learning algorithms, such as DBSCAN, K-Means clustering, and t-SNE, to uncover hidden patterns in patient data. These models are designed to group patients with similar risk profiles, detect anomalies indicative of rare or severe cases, and provide insights into correlations between patient metrics. The combined use of supervised and unsupervised approaches not only enhances predictive capabilities but also contributes to a deeper understanding of the underlying structure of the data. Ultimately, this comprehensive application of machine learning aims to revolutionize the diagnostic process for heart disease.

Literature Review
While mortality rates from heart disease have significantly declined over the past three decades, it remains the leading cause of death among adults globally and is projected to continue as the primary cause of death and disability in the Western world throughout the 21st century [3].
The application of machine learning in cardiovascular diagnostics has been a transformative development in heart disease healthcare, providing significant improvements in early disease detection, risk, and assessment in recent years. Over the last decade there has been significant development, research, and proven performance of ML-based models for heart disease diagnosis. For example, the authors in [4] introduced a hybrid model combining machine learning and deep learning techniques to predict cardiovascular diseases. Using a public heart disease dataset, their approach achieved high accuracy, highlighting the benefits of integrating deep learning with traditional ML models for robust predictions. Similarly, another study investigated the application of artificial intelligence (AI) in diagnosing cardiovascular conditions. Their study emphasized AI’s role in augmenting diagnostic workflows, enhancing disease stratification, and enabling precise treatment planning. These capabilities position AI as a critical tool for improving clinical decision-making [5]. There have been other studies [6] that have developed a deep learning model using a one-dimensional convolutional neural network (1D-CNN) to classify patients based on heart disease risk. Their approach achieved an impressive 96% accuracy on test datasets, demonstrating the potential of deep learning to enhance diagnostic accuracy and reduce false-positive rates. The authors in [7] further advanced the field by proposing an ensemble framework combining multiple ML algorithms. Their model outperformed individual classifiers, achieving a diagnostic accuracy of 92.34%, reinforcing the value of ensemble learning in healthcare. Despite these advances, challenges persist, including data imbalance, generalizability across diverse populations, and the need for seamless integration into clinical workflows. Future research should focus on enhancing model interpretability, addressing ethical concerns, and validating models in real-world clinical settings to maximize their impact. By integrating advanced ML techniques with clinical expertise, researchers can significantly improve early diagnosis, optimize treatment strategies, and ultimately reduce the global burden of cardiovascular diseases.

Methodology and Results
The project begins with the analysis of a public heart disease dataset [8] , containing clinical features such as age, gender, cholesterol levels, blood pressure, chest pain type, maximum heart rate achieved, and others. The target variable indicates the presence or absence of heart disease. The dataset is loaded and explored to assess its structure, dimensionality, and data types. Summary statistics and visualization techniques are employed to understand the distribution of features, identify correlations, and detect class imbalance in the target variable. A mild imbalance is observed in the target values, prompting the use of oversampling techniques to ensure balanced representation of classes in subsequent analysis.

Metric
age
sex
cp
trestbps
chol
fbs
restecg
thalach
exang
oldpeak
slope
ca
thal
target
count
330
330
330
330
330
330
330
330
330
330
330
330
330
330
mean
54.445455
0.693939
0.927273
131.654545
246.990909
0.157576
0.506061
148.469697
0.351515
1.072121
1.393939
0.784848
2.306061
0.5
std
9.008527
0.461555
1.040596
17.623263
51.9905
0.364896
0.524442
22.991488
0.478168
1.168434
0.615869
1.04274
0.628811
0.500759
min
29
0
0
94
126
0
0
71
0
0
0
0
0
0
25%
48
0
0
120
211
0
0
132
0
0
1
0
2
0
50%
56
1
0.5
130
243
0
0
152
0
0.8
1
0
2
0.5
75%
61
1
2
140
280.25
0
1
165
1
1.8
2
1
3
1
max
77
1
3
200
564
1
2
202
1
6.2
2
4
3
1


Table S3.1: A comprehensive statistical description of the dataset after preprocessing. 

Preprocessing Steps

The preprocessing pipeline addresses critical issues in the raw dataset:
Handling Missing Values: A heatmap confirms that there are no missing values in the dataset, eliminating the need for imputation.
Feature Scaling: Given the diverse ranges of feature values, standardization is applied to normalize the data. This ensures that features with larger numeric ranges do not dominate the machine learning models.
Class Balancing: The RandomOverSampler technique is used to balance the dataset by oversampling the minority class, ensuring equitable model performance across target classes.
Outlier Detection: Boxplots are utilized to visually inspect features for outliers. Minimal outliers are detected, primarily in resting heart rate, which are retained given their potential relevance to heart disease diagnosis.
(a)							(b)

Fig S3.1: Figures highlighting the (a) correlations between features of the data set as well a (b) heat map highlighting the lack of missing values. Overall, the data was very clean and little preprocessing was necessary other than addressing some imbalance in target values.

Supervised Learning Models

Three supervised learning models, including Support Vector Machines (SVM), K-Nearest Neighbors (KNN), and Naive Bayes, are implemented to classify patients based on their likelihood of having heart disease.
Support Vector Machines (SVM): The SVM model optimizes its hyperparameters, including C (regularization parameter), gamma (kernel coefficient), and kernel type (linear or rbf). The hyperparameter tuning is conducted using Optuna with a cross-validated objective function to maximize classification accuracy. SVM demonstrates robust performance with high precision and recall metrics.
K-Nearest Neighbors (KNN): The KNN model is configured with hyperparameters such as the number of neighbors (n_neighbors), the weight function (uniform or distance), and the distance metric (euclidean, manhattan, or minkowski). Optuna is employed to determine the optimal hyperparameter combination, leveraging a stratified k-fold cross-validation to ensure reliable performance evaluation.
Naive Bayes: The Gaussian Naive Bayes classifier is implemented without significant hyperparameter tuning, as the model relies on probabilistic assumptions and is computationally efficient. It serves as a benchmark for comparing the performance of more complex models.

Performance metrics such as accuracy, precision, recall, and F1-score are computed for all models to provide a comprehensive evaluation. SHAP (SHapley Additive exPlanations) values are used to interpret the importance of individual features in model predictions, enhancing transparency.
(a)							(b)



Metric:
Value (%)
Accuracy:
81.81818182
Precision:
78.125
Recall:
83.33333333
F1 Score:
80.64516129


Table S3.2: Metrics describing the performance of the Naive Bayes on the test set.

Fig S3.2: (a) Confusion matrix demonstrating the performance of the SVM on the test set and the (b) SHAP values summary plot highlighting the impact of the features on the KNN model’s predictions. These figures were created for each supervised learning model in order to assess their performance.

Unsupervised Learning Models

To explore hidden patterns and groupings within the data, three unsupervised learning models are implemented:
DBSCAN: The density-based clustering algorithm DBSCAN is tuned using Optuna to identify the optimal eps (maximum distance for clustering) and min_samples (minimum samples for a cluster). The silhouette score evaluates clustering performance, with higher values indicating better-defined clusters.
K-Means Clustering: The K-Means algorithm is tuned for the number of clusters (n_clusters) using Optuna. The silhouette score is calculated to assess cluster cohesion and separation, making K-Means the best-performing unsupervised model in this implementation.
t-SNE with K-Means: t-SNE (t-Distributed Stochastic Neighbor Embedding) is applied for dimensionality reduction, followed by clustering with K-Means. Parameters such as perplexity and n_iter are optimized to balance local and global data structure visualization. This combination enables high-dimensional data to be visualized effectively while grouping similar instances.

Metric
DBSCAN
K-Means
t-SNE
Silhouette
0.143465
0.756119
0.388563


Table S3.3: The silhouette scores of the unsupervised ML models describing their ability to successfully cluster the features. K-Means was by the far best model with a score of 0.756.

Hyperparameter Tuning

The project employs Optuna for systematic hyperparameter optimization. For supervised models, metrics such as cross-validation accuracy guide the selection of optimal parameter configurations. For unsupervised models, silhouette scores are maximized to ensure meaningful clustering. This automated tuning framework enhances model performance while reducing the risk of overfitting.


Fig S3.3: Example of hyperparameter tuning for the KNN model using Optuna. A similar process was applied to each model.

Model Evaluation and Insights

Evaluation metrics are calculated for both supervised and unsupervised models, highlighting their strengths and limitations. Key insights into feature importance, clustering patterns, and model reliability are derived through visualizations and interpretability tools like SHAP and decision plots. The integration of supervised and unsupervised approaches provides a holistic understanding of the dataset, improving diagnostic accuracy and uncovering valuable patient risk profiles. This comprehensive implementation demonstrates the potential of machine learning to advance heart disease diagnosis, combining robust preprocessing, algorithmic diversity, and interpretability to achieve meaningful results.

Discussion
The results of this implementation demonstrate the potential of machine learning (ML) to assist medical services in accurately diagnosing heart disease amongst other medical conditions. The results highlight the potential of machine learning for heart disease diagnosis. Supervised models performed well, with K-Nearest Neighbors (KNN) achieving the highest accuracy 85% and recall 93%, indicating its effectiveness in identifying patients at risk. Support Vector Machines (SVM) also showed strong performance, benefiting from optimized hyperparameters, while Naive Bayes provided a reliable benchmark. Unsupervised models offered mixed results. K-Means clustering achieved a moderate silhouette score of 0.76, suggesting meaningful groupings in the data, whereas DBSCAN struggled due to high dimensionality. t-SNE combined with K-Means provided valuable visual insights but was less effective for clustering. Key implementation steps like oversampling, feature scaling, and hyperparameter tuning were crucial for success. Interpretability tools such as SHAP confirmed that features like chest pain type and cholesterol levels were critical predictors, aligning model outputs with clinical expectations. Limitations include the dataset’s lack of diversity and the absence of longitudinal data, which restrict generalizability and dynamic risk assessment. Improvements could focus on external validation, feature engineering, and hybrid modeling to enhance performance and applicability. In summary, the implementation effectively demonstrates ML’s promise for non-invasive heart disease diagnosis but requires further refinement to maximize clinical impact.

Metric
SVM
KNN
Naive Bayes
Accuracy
0.787879
0.848485
0.818182
Precision
0.710526
0.777778
0.78125
Recall
0.9
0.933333
0.833333
F1 Score
0.794118
0.848485
0.806452


Table S4.1: Metrics describing the performance of the 3 supervised machine learning algorithms. They demonstrated impressive performance overall. 

Conclusion
This project demonstrates the significant potential of machine learning for heart disease diagnosis. By leveraging supervised models such as K-Nearest Neighbors, Support Vector Machines, and Naive Bayes, high accuracy and recall were achieved, with KNN emerging as the most effective model. Unsupervised models provided valuable insights into clustering risk profiles, though results highlight the need for further optimization. Key preprocessing steps, including class balancing, feature scaling, and hyperparameter tuning, were essential for reliable performance. Interpretability tools like SHAP enhanced clinical relevance by identifying critical predictors such as chest pain type and cholesterol levels. While the results are promising, limitations such as dataset generalizability and the challenges of unsupervised clustering underscore the need for external validation and advanced techniques. Future efforts should focus on diverse datasets, temporal analysis, and hybrid modeling to improve robustness and scalability. Overall, this implementation underscores the transformative potential of machine learning in creating accessible, non-invasive, and accurate diagnostic tools for heart disease.














References:
[1] Tao R., Zhang S., Huang X., Tao M., Ma J., Ma S., Zhang C., Zhang T., Tang F., Lu J., Shen C., and Xie X., Magnetocardiography-based ischemic heart disease detection and localization using machine learning methods, IEEE Transactions on Biomedical Engineering. (2019) 66, no. 6, 1658–1667, https://doi.org/10.1109/tbme.2018.2877649, 2-s2.0-85055716372.

[2] Fitriyani N. L., Syafrudin M., Alfian G., and Rhee J., HDPM: an effective heart disease prediction model for a clinical decision support system, IEEE Access. (2020) 8, 133034, https://doi.org/10.1109/access.2020.3010511.

[3] Lloyd-Jones, Donald M., et al. “Lifetime Risk of Developing Coronary Heart Disease.” The Lancet, vol. 352, no. 9123, 1998, pp. 1630–1632. DOI:10.1016/S0140-6736(98)10279-9.

[4] Sadr, H., Salari, A., Ashoobi, M.T. et al. Cardiovascular disease diagnosis: a holistic approach using the integration of machine learning and deep learning models. Eur J Med Res 29, 455 (2024). https://doi.org/10.1186/s40001-024-02044-7

[5] Sun, X., Yin, Y., Yang, Q. et al. Artificial intelligence in cardiovascular diseases: diagnostic and therapeutic perspectives. Eur J Med Res 28, 242 (2023). https://doi.org/10.1186/s40001-023-01065-y

[6] S. Hussain, S. K. Nanda, S. Barigidad, S. Akhtar, M. Suaib and N. K. Ray, "Novel Deep Learning Architecture for Predicting Heart Disease using CNN," 2021 19th OITS International Conference on Information Technology (OCIT), Bhubaneswar, India, 2021, pp. 353-357, doi: 10.1109/OCIT53463.2021.00076.

[7] Tiwari, Achyut et al. “Ensemble framework for cardiovascular disease prediction.” Computers in biology and medicine 146 (2022): 105624 .

[8]Redwan, Sony. Heart Disease Data Set from UCI data repository. Kaggle, 2020, https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data. Accessed 12 Nov. 2024.




