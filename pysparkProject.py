#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import *
from pyspark.sql.functions import col
from pyspark.sql.functions import corr


# In[3]:


# Create a SparkSession
spark = SparkSession.builder.appName("ReadCSV").getOrCreate()

# Read the CSV file into a DataFrame
df = spark.read.csv("ML_hw_dataset.csv", header=True, inferSchema=True)

df.printSchema()
df.show()


# In[4]:


df.dropna(how='any', thresh=None, subset=None)
df.show()


# In[5]:


# select categorical columns
categorical_cols = [ 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
# index all categorical columns
for column in categorical_cols:
  indexer = StringIndexer(inputCol=column, outputCol=column+"_index")
  df = indexer.fit(df).transform(df)
    
# Drop the original categorical column and rename the encoded column
for column in categorical_cols:
  df = df.drop(column).withColumnRenamed(column+"_index", column)

# Show the result
df.show()
df.printSchema()


# In[6]:


selected_cols=['marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome','age','duration','campaign','pdays','previous','emp_var_rate','cons_price_idx','cons_conf_idx','euribor3m','nr_employed','y']
# Calculate the correlation matrix using the corr() function
corr_matrix = df.select(selected_cols).toPandas().corr()

# Print the correlation matrix
print(corr_matrix)


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt
# Create heatmap with colorbar
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True)
# Show plot
plt.show()


# In[8]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
inputCol=["euribor3m", "nr_employed", "emp_var_rate"]
assembler = VectorAssembler(inputCols =["euribor3m", "nr_employed", "emp_var_rate","duration","pdays","emp_var_rate"], outputCol='features')
output = assembler.transform(df)
finalised_data = output.select('features', 'y')
finalised_data.show()
train, test = finalised_data.randomSplit([0.7, 0.3])
print(str(train.count()),str(test.count()))
lr = LogisticRegression(featuresCol='features',labelCol="y",maxIter=5)
lrn = lr.fit(train)
lrn_summary = lrn.summary
lrn_summary.predictions.show()


# In[9]:


# Make predictions on the test data
predictions = lrn.transform(test)

# Select prediction and label columns for evaluation
predictions = predictions.select(col("prediction"), col("y").alias("label"))

# Evaluate the model's performance
accuracy = predictions.filter(predictions.prediction == predictions.label).count() / float(test.count())

print("Accuracy:", accuracy)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




