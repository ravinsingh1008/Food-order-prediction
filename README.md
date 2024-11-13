# Predict-Food-Delivery-Time
Food order prediction 
This food order prediction project aims to predict customers’ food orders based on their preferences, location, past orders, time of the day, etc. Learners will build a classification model using supervised learning techniques and Python libraries to make predictions. 

Learning outcomes: 
Understanding of Machine Learning work process: Get to know how machine learning works in the real world 
Data processing: Learn to clean, transform, and prepare data for training, along with handling missing values and encoding categorical variables 
Exploratory Data Analysis (EDA): Understand data visualization, learn the relationships between features, and gain insights on how to select the appropriate machine learning algorithm 
Model selection: Experiment with multiple supervised learning algorithms, evaluating their performance and optimizing their accuracy 
Model evaluation: Knowledge of assessing the performance of the classification model using multiple evaluation metrics 

What it takes to execute this project:
Collect the required dataset containing customer food order information like types of food ordered, time of order, location, etc 
Use Python libraries like Pandas and NumPy to prepare your data, identify patterns, and clean inconsistencies
Split the data into parts, one for training and the other for testing 
Train your model on the training data, allowing it to learn different patterns that influence customers’ food choices
Test the model to check for its performance and accuracy 

Real world applications: 
Restaurants can use it to forecast customer orders and keep their stock filled, have proper staff, and be ready for the food preparation process
Food delivery apps can use this to provide personalized food recommendations to customers based on their search or past orders
Food producers and distributors can use the model to predict food habit changes or demands and plan for better production and distribution strategy 
Marketers can use it to develop targeted marketing campaigns, offering personalized promotions based on customer preferences 

The entire world is transforming digitally and our relationship with technology has grown exponentially over the last few years. We have grown closer to technology, and it has made our life a lot easier by saving time and effort. Today everything is accessible with smartphones — from groceries to cooked food and from medicines to doctors. In this hackathon, we provide you with data that is a by-product as well as a thriving proof of this growing relationship. 

When was the last time you ordered food online? And how long did it take to reach you?

![Image](https://www.machinehack.com/wp-content/uploads/2019/11/courier-delivery-service-illustration-3-02.jpg)

In this hackathon, we are providing you with data from thousands of restaurants in India regarding the time they take to deliver food for online order. As data scientists, our goal is to predict the online order delivery time based on the given factors.

Analytics India Magazine and IMS Proschool bring to you ‘Predicting Predicting Food Delivery Time Hackathon’.

Size of training set: 11,094 records

Size of test set: 2,774 records

FEATURES:

- Restaurant: A unique ID that represents a restaurant.
- Location: The location of the restaurant.
- Cuisines: The cuisines offered by the restaurant.
- Average_Cost: The average cost for one person/order.
- Minimum_Order: The minimum order amount.
- Rating: Customer rating for the restaurant.
- Votes: The total number of customer votes for the restaurant.
- Reviews: The number of customer reviews for the restaurant.
- Delivery_Time: The order delivery time of the restaurant. (Target Classes) 


## Modeling and Predicting
Finally, we are on to building a simple classifier that can predict and evaluate on our sample data.

We will use a simple RandomForest classifier without any parameter tuning. This is a good starting point.

Before we begin to create a model, make sure we have a small dataset to test our model performance. The best approach is to split the training set into a training set and a validation set.

Also, it is important to separate out the independent and dependent variables from all the dataset samples.


What the following module does:

Splits the training data into a training set and a validation set
Separates the dependent and independent features for the training set and validation set
Initializes an Random Forest classifier
Trains the classifier with the training data
Evaluates the score on a validation set 
Predicts the classes for the test set.

#### Training Score is 0.774 and Testing Score is 76.901


Link to the Hackathon: [Predict Food Delivery Time](https://www.machinehack.com/course/predicting-food-delivery-time-hackathon-by-ims-proschool/)
