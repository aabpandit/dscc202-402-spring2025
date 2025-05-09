# Databricks notebook source
# MAGIC %md
# MAGIC ## DSCC202-402 Data Science at Scale Final Project
# MAGIC ### Tracking Tweet sentiment at scale using a pretrained transformer (classifier)
# MAGIC <p>Consider the following illustration of the end to end system that you will be building.  Each student should do their own work.  The project will demonstrate your understanding of Spark Streaming, the medalion data architecture using Delta Lake, Spark Inference at Scale using an MLflow packaged model as well as Exploritory Data Analysis and System Tracking and Monitoring.</p>
# MAGIC <br><br>
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/images/pipeline.drawio.png">
# MAGIC
# MAGIC <p>
# MAGIC You will be pulling an updated copy of the course GitHub repositiory: <a href="https://github.com/lpalum/dscc202-402-spring2025">The Repo</a>.  
# MAGIC
# MAGIC Once you have updated your fork of the repository you should see the following template project that is resident in the final_project directory.
# MAGIC </p>
# MAGIC
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/images/notebooks.drawio.png">
# MAGIC
# MAGIC <p>
# MAGIC You can then pull your project into the Databrick Workspace using the <a href="https://github.com/apps/databricks">Databricks App on Github</a> or by cloning the repo to your laptop and then uploading the final_project directory and its contents to your workspace using file imports.  Your choice.
# MAGIC
# MAGIC <p>
# MAGIC Work your way through this notebook which will give you the steps required to submit a complete and compliant project.  The following illustration and associated data dictionary specifies the transformations and data that you are to generate for each step in the medallion pipeline.
# MAGIC </p>
# MAGIC <br><br>
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/images/dataframes.drawio.png">
# MAGIC
# MAGIC #### Bronze Data - raw ingest
# MAGIC - date - string in the source json
# MAGIC - user - string in the source json
# MAGIC - text - tweet string in the source json
# MAGIC - sentiment - the given sentiment of the text as determined by an unknown model that is provided in the source json
# MAGIC - source_file - the path of the source json file the this row of data was read from
# MAGIC - processing_time - a timestamp of when you read this row from the source json
# MAGIC
# MAGIC #### Silver Data - Bronze Preprocessing
# MAGIC - timestamp - convert date string in the bronze data to a timestamp
# MAGIC - mention - every @username mentioned in the text string in the bronze data gets a row in this silver data table.
# MAGIC - cleaned_text - the bronze text data with the mentions (@username) removed.
# MAGIC - sentiment - the given sentiment that was associated with the text in the bronze table.
# MAGIC
# MAGIC #### Gold Data - Silver Table Inference
# MAGIC - timestamp - the timestamp from the silver data table rows
# MAGIC - mention - the mention from the silver data table rows
# MAGIC - cleaned_text - the cleaned_text from the silver data table rows
# MAGIC - sentiment - the given sentiment from the silver data table rows
# MAGIC - predicted_score - score out of 100 from the Hugging Face Sentiment Transformer
# MAGIC - predicted_sentiment - string representation of the sentiment
# MAGIC - sentiment_id - 0 for negative and 1 for postive associated with the given sentiment
# MAGIC - predicted_sentiment_id - 0 for negative and 1 for positive assocaited with the Hugging Face Sentiment Transformer
# MAGIC
# MAGIC #### Application Data - Gold Table Aggregation
# MAGIC - min_timestamp - the oldest timestamp on a given mention (@username)
# MAGIC - max_timestamp - the newest timestamp on a given mention (@username)
# MAGIC - mention - the user (@username) that this row pertains to.
# MAGIC - negative - total negative tweets directed at this mention (@username)
# MAGIC - neutral - total neutral tweets directed at this mention (@username)
# MAGIC - positive - total positive tweets directed at this mention (@username)
# MAGIC
# MAGIC When you are designing your approach, one of the main decisions that you will need to make is how you are going to orchestrate the streaming data processing in your pipeline.  There are several valid approaches to triggering your steams and how you will gate the execution of your pipeline.  Think through how you want to proceed and ask questions if you need guidance. The following references may be helpful:
# MAGIC - [Spark Structured Streaming Programming Guide](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)
# MAGIC - [Databricks Autoloader - Cloudfiles](https://docs.databricks.com/en/ingestion/auto-loader/index.html)
# MAGIC - [In class examples - Spark Structured Streaming Performance](https://dbc-f85bdc5b-07db.cloud.databricks.com/editor/notebooks/2638424645880316?o=1093580174577663)
# MAGIC
# MAGIC ### Be sure your project runs end to end when *Run all* is executued on this notebook! (7 points)
# MAGIC
# MAGIC ### This project is worth 25% of your final grade.
# MAGIC - DSCC-202 Students have 55 possible points on this project (see points above and the instructions below)
# MAGIC - DSCC-402 Students have 60 possible points on this project (one extra section to complete)

# COMMAND ----------

# DBTITLE 1,Pull in the Includes & Utiltites
# MAGIC %run ./includes/includes

# COMMAND ----------

# DBTITLE 1,Notebook Control Widgets (maybe helpful)
"""
Adding a widget to the notebook to control the clearing of a previous run.
or stopping the active streams using routines defined in the utilities notebook
"""
dbutils.widgets.removeAll()

dbutils.widgets.dropdown("clear_previous_run", "No", ["No","Yes"])
if (getArgument("clear_previous_run") == "Yes"):
    clear_previous_run()
    print("Cleared all previous data.")

dbutils.widgets.dropdown("stop_streams", "No", ["No","Yes"])
if (getArgument("stop_streams") == "Yes"):
    stop_all_streams()
    print("Stopped all active streams.")

dbutils.widgets.dropdown("optimize_tables", "No", ["No","Yes"])
if (getArgument("optimize_tables") == "Yes"):
    # Suck up those small files that we have been appending.
    # Optimize the tables
    optimize_table(BRONZE_DELTA)
    optimize_table(SILVER_DELTA)
    optimize_table(GOLD_DELTA)
    print("Optimized all of the Delta Tables")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.0 Import your libraries here (2 points)
# MAGIC - Are your shuffle partitions consistent with your cluster and your workload?
# MAGIC - Do you have the necessary libraries to perform the required operations in the pipeline/application?

# COMMAND ----------

from pyspark.sql.types import *
import pandas as pd
from pyspark.sql.functions import input_file_name, current_timestamp, udf, col, regexp_extract, regexp_replace, expr, to_timestamp, when, count, isnan, desc, upper, lower, trim
import mlflow
import mlflow.pyfunc
from transformers import pipeline
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import os
import json
import shutil
from collections import defaultdict


# COMMAND ----------

#enabling AQE to better process stream
spark.conf.set("spark.sql.adaptive.enabled", "true")

#print(spark.conf.get("spark.sql.shuffle.partitions"))
#print(sc.defaultParallelism)

# COMMAND ----------

spark.conf.set("spark.sql.shuffle.partitions", 2 * sc.defaultParallelism)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.0 Define and execute utility functions (3 points)
# MAGIC - Read the source file directory listing
# MAGIC - Count the source files (how many are there?)
# MAGIC - print the contents of one of the files

# COMMAND ----------

#reading in source files
source_files = dbutils.fs.ls(TWEET_SOURCE_PATH)
display(source_files)

# COMMAND ----------

#count of source files
len(source_files)

# COMMAND ----------

#viewing one file
display(spark.read.json(source_files[0].path))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.0 Transform the Raw Data to Bronze Data using a stream  (8 points)
# MAGIC - define the schema for the raw data
# MAGIC - setup a read stream using cloudfiles and the source data format
# MAGIC - setup a write stream using delta lake to append to the bronze delta table
# MAGIC - enforce schema
# MAGIC - allow a new schema to be merged into the bronze delta table
# MAGIC - Use the defined BRONZE_CHECKPOINT and BRONZE_DELTA paths defined in the includes
# MAGIC - name your raw to bronze stream as bronze_stream
# MAGIC - transform the raw data to the bronze data using the data definition at the top of the notebook

# COMMAND ----------

# Clear checkpoint and output to reprocess all files
dbutils.fs.rm(BRONZE_CHECKPOINT, True)
dbutils.fs.rm(BRONZE_DELTA, True)

# COMMAND ----------

#defining schema for raw data
schema = (StructType(
        [StructField("date", StringType(),nullable=True),
        StructField("sentiment", StringType(), nullable=True),
        StructField("text", StringType(), nullable=True),
        StructField("user", StringType(),nullable=True)])
        )

# COMMAND ----------

#read, transform & write stream
read_stream = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.allowOverwrites", "true")
    .option("maxFilesPerTrigger", 10) #q6
    .schema(schema)
    .load(TWEET_SOURCE_PATH)
    .withColumn("source_file", input_file_name())
    .withColumn("processing_time", current_timestamp())
)

bronze_stream = (
    read_stream.writeStream
    .format("delta")
    .outputMode("append")
    .trigger(processingTime="30 seconds")
    .queryName("bronze_stream")  #required for monitoring
    .option("checkpointLocation", BRONZE_CHECKPOINT)
    .option("mergeSchema", "true")
    .start(BRONZE_DELTA)
)


# COMMAND ----------

while bronze_stream.isActive:
    time.sleep(10)
    progress = bronze_stream.lastProgress
    print(f"\n[{datetime.now()}] Progress Snapshot:")
    if progress:
        print(progress)
    else:
        print("No progress yet.")

# COMMAND ----------

bronze_stream.stop()

# COMMAND ----------

#DeltaTable.forPath(spark, BRONZE_DELTA).optimize()

# COMMAND ----------

bronze_table = spark.read.format("delta").load(BRONZE_DELTA)
#display(bronze_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.0 Transform the Bronze Data to Silver Data using a stream (5 points)
# MAGIC - setup a read stream on your bronze delta table
# MAGIC - setup a write stream to append to the silver delta table
# MAGIC - Use the defined SILVER_CHECKPOINT and SILVER_DELTA paths in the includes
# MAGIC - name your bronze to silver stream as silver_stream
# MAGIC - transform the bronze data to the silver data using the data definition at the top of the notebook

# COMMAND ----------

#reading in bronze stream, applying transformating & writing to silver stream
silver_stream = (spark.readStream
                .format("delta")
                .load(BRONZE_DELTA, inferSchema=True)
                #transformations
                .withColumn("date_str", expr("substring(date, 5, length(date))"))
                .withColumn("timestamp", to_timestamp(col("date_str"), "MMM dd HH:mm:ss z yyyy"))
                .withColumn("mention", regexp_extract(col("text"), r"(@\w+)", 0)) #extract first mention
                .withColumn("cleaned_text", regexp_replace(col("text"), r'@\w+', "")) #remove mentions
                .drop("date", "date_str", "text", "source_file", "processing_time")
                .select("timestamp", "mention", "cleaned_text", "sentiment") #silver columms
                #write stream to silver
                .writeStream
                .outputMode("append")
                .format("delta")
                #.trigger(processingTime="30 seconds") 
                .queryName("silver_table")
                .option("checkpointLocation", SILVER_CHECKPOINT)
                .start(SILVER_DELTA)
        )

#silver_stream.awaitTermination()

# COMMAND ----------

silver_stream.stop()

# COMMAND ----------

#DeltaTable.forPath(spark, SILVER_DELTA).optimize()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.0 Transform the Silver Data to Gold Data using a stream (7 points)
# MAGIC - setup a read stream on your silver delta table
# MAGIC - setup a write stream to append to the gold delta table
# MAGIC - Use the defined GOLD_CHECKPOINT and GOLD_DELTA paths defines in the includes
# MAGIC - name your silver to gold stream as gold_stream
# MAGIC - transform the silver data to the gold data using the data definition at the top of the notebook
# MAGIC - Load the pretrained transformer sentiment classifier from the MODEL_NAME at the production level from the MLflow registry
# MAGIC - Use a spark UDF to parallelize the inference across your silver data

# COMMAND ----------

#loading form HF
HF_model = pipeline("sentiment-analysis", model=HF_MODEL_NAME)

def inference(text):
    prediction = HF_model(text)
    label = prediction[0]['label']
    score = prediction[0]['score']*100
    sentiment_id = 1 if label == "POSITIVE" else 0
    return label, score, sentiment_id

#output schema for UDF
schema = StructType([
    StructField("label", StringType(), False),
    StructField("score", DoubleType(), False),
    StructField("id", IntegerType(), False)
])

infer_udf = udf(inference, schema)


# COMMAND ----------

silver = (spark
            .readStream
            .format('delta')
            .load(SILVER_DELTA)
        )

# COMMAND ----------

#parallelizing w UDF inference on silver data
transform_silver = (silver
             .withColumn("prediction", infer_udf(col("cleaned_text")))
             .select(
                 col("timestamp"),
                 col("mention"),
                 col("cleaned_text"),
                 col("sentiment"),
                 col("prediction.score").alias("predicted_score"),
                 col("prediction.label").alias("predicted_sentiment"),
                 when(col("sentiment") == "positive", 1).otherwise(0).alias("sentiment_id"),
                 col("prediction.id").alias("predicted_sentiment_id")
             )
)

# COMMAND ----------

gold_stream = (transform_silver
                .writeStream
                .format("delta")
                .outputMode("append")
                .queryName("gold_stream")
                .trigger(processingTime="30 seconds")
                .option("checkpointLocation", GOLD_CHECKPOINT)
                .start(GOLD_DELTA)
)

#gold_stream.awaitTermination()

# COMMAND ----------

gold_stream.stop()

# COMMAND ----------

#DeltaTable.forPath(spark, GOLD_DELTA).optimize() 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.0 Monitor your Streams (5 points)
# MAGIC - Setup a loop that runs at least every 10 seconds
# MAGIC - Print a timestamp of the monitoring query along with the list of streams, rows processed on each, and the processing time on each
# MAGIC - Run the loop until all of the data is processed (0 rows read on each active stream)
# MAGIC - Plot a line graph that shows the data processed by each stream over time
# MAGIC - Plot a line graph that shows the average processing time on each stream over time

# COMMAND ----------

# Function to get current stats from all active Spark streams
# Data for plotting
stream_data = defaultdict(list)
stream_time = []

# Loop to monitor active streams
while True:
    active_streams = spark.streams.active
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stream_time.append(timestamp)

    print(f"\n[{timestamp}] Monitoring {len(active_streams)} active streams...")

    all_idle = True
    for stream in active_streams:
        progress = stream.lastProgress
        if progress:
            name = stream.name or stream.id
            input_rows = int(progress["numInputRows"])
            proc_time = float(progress["durationMs"]["addBatch"]) / 1000  # convert ms to sec

            stream_data[name + "_rows"].append(input_rows)
            stream_data[name + "_time"].append(proc_time)

            print(f"Stream {name}: Rows={input_rows}, Time={proc_time:.2f}s")

            if input_rows > 0:
                all_idle = False
        else:
            print(f"Stream {stream.id} has no progress yet.")

    if all_idle:
        print("All streams are idle (0 rows processed). Stopping monitor.")
        break

    time.sleep(10)  # Wait 10 seconds before next check

# Plotting rows processed
for key in stream_data:
    if key.endswith("_rows"):
        plt.figure()
        plt.plot(stream_time, stream_data[key], marker='o')
        plt.title(f"{key.replace('_rows', '')} - Rows Processed Over Time")
        plt.xlabel("Time")
        plt.ylabel("Rows Processed")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Plotting processing time
for key in stream_data:
    if key.endswith("_time"):
        plt.figure()
        plt.plot(stream_time, stream_data[key], marker='x')
        plt.title(f"{key.replace('_time', '')} - Processing Time Over Time")
        plt.xlabel("Time")
        plt.ylabel("Processing Time (s)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.0 Bronze Data Exploratory Data Analysis (5 points)
# MAGIC - How many tweets are captured in your Bronze Table?
# MAGIC - Are there any columns that contain Nan or Null values?  If so how many and what will you do in your silver transforms to address this?
# MAGIC - Count the number of tweets by each unique user handle and sort the data by descending count.
# MAGIC - How many tweets have at least one mention (@) how many tweet have no mentions (@)
# MAGIC - Plot a bar chart that shows the top 20 tweeters (users)
# MAGIC

# COMMAND ----------

BRONZE_DF = spark.read.format("delta").load(BRONZE_DELTA)
tweet_count = BRONZE_DF.count()
print(f"Total number of tweets: {tweet_count}")

# COMMAND ----------

null_exprs = []
for c in BRONZE_DF.columns:
    dtype = BRONZE_DF.schema[c].dataType
    if isinstance(dtype, (DoubleType, FloatType)):
        null_exprs.append(count(when(col(c).isNull() | isnan(col(c)), c)).alias(c))
    else:
        null_exprs.append(count(when(col(c).isNull(), c)).alias(c))

null_check = BRONZE_DF.select(null_exprs)
print("Null/NaN counts per column:")
null_check.show()
     

# COMMAND ----------

user_counts = BRONZE_DF.groupBy("user").count().orderBy(desc("count"))
user_counts.show()

# COMMAND ----------

mention_col = "text"
with_mentions = BRONZE_DF.filter(col(mention_col).contains("@")).count()
without_mentions = BRONZE_DF.filter(~col(mention_col).contains("@")).count()
print(f"Tweets with @mention: {with_mentions}")
print(f"Tweets without @mention: {without_mentions}")

# COMMAND ----------

top20_users = user_counts.limit(20).toPandas()

# COMMAND ----------

plt.figure(figsize=(12, 6))
bars = plt.bar(top20_users["user"], top20_users["count"], color="skyblue")
plt.title("Top 20 Users by Tweet Count")
plt.ylabel("Tweet Count")
plt.xlabel("User")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.0 Capture the accuracy metrics from the gold table in MLflow  (4 points)
# MAGIC Store the following in an MLflow experiment run:
# MAGIC - Store the precision, recall, and F1-score as MLflow metrics
# MAGIC - Store an image of the confusion matrix as an MLflow artifact
# MAGIC - Store the model name and the MLflow version that was used as an MLflow parameters
# MAGIC - Store the version of the Delta Table (input-silver) as an MLflow parameter

# COMMAND ----------

gold = spark.read.format("delta").load(GOLD_DELTA)

# COMMAND ----------

# Start an MLflow run
pdf = gold.select(
    "sentiment",
    "predicted_sentiment",
    "predicted_sentiment_id",
    "predicted_score"
).toPandas()

y_true = pdf["sentiment"].str.upper()
y_pred = pdf["predicted_sentiment"].str.upper()

precision = precision_score(y_true, y_pred, average="macro")
recall = recall_score(y_true, y_pred, average="macro")
f1 = f1_score(y_true, y_pred, average="macro")

cm = confusion_matrix(y_true, y_pred, labels=["POSITIVE", "NEGATIVE", "NEU"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["POS", "NEG", "NEU"])

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax)
plt.tight_layout()
conf_matrix_path = "/tmp/confusion_matrix.png"
plt.savefig(conf_matrix_path)
plt.close()

with mlflow.start_run(run_name="gold_metrics_logging"):
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.log_artifact(conf_matrix_path)

    mlflow.log_param("model_name", MODEL_NAME)
    mlflow.log_param("mlflow_version", mlflow.__version__)

    delta_table = DeltaTable.forPath(spark, GOLD_DELTA)
    delta_version = delta_table.history().select("version").orderBy("version", ascending=False).first()["version"]
    mlflow.log_param("delta_version", delta_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9.0 Application Data Processing and Visualization (6 points)
# MAGIC - How many mentions are there in the gold data total?
# MAGIC - Count the number of neutral, positive and negative tweets for each mention in new columns
# MAGIC - Capture the total for each mention in a new column
# MAGIC - Sort the mention count totals in descending order
# MAGIC - Plot a bar chart of the top 20 mentions with positive sentiment (the people who are in favor)
# MAGIC - Plot a bar chart of the top 20 mentions with negative sentiment (the people who are the vilians)
# MAGIC
# MAGIC *note: A mention is a specific twitter user that has been "mentioned" in a tweet with an @user reference.

# COMMAND ----------

gold_df = spark.read.format("delta").load(GOLD_DELTA)

mention_count = gold_df.filter(col("mention").isNotNull()).count()
print(f"Total number of mentions: {mention_count}")

gold_df = gold_df.withColumn("sentiment", trim(lower(col("sentiment"))))

mention_sentiment_counts = (
    gold_df.filter(col("mention").isNotNull())
    .groupBy("mention")
    .agg(
        count(when(col("sentiment") == "neu", True)).alias("neutral"),
        count(when(col("sentiment") == "positive", True)).alias("positive"),
        count(when(col("sentiment") == "negative", True)).alias("negative")
    )
)

mention_sentiment_counts = mention_sentiment_counts.withColumn(
    "total", col("neutral") + col("positive") + col("negative")
)

mention_sorted = mention_sentiment_counts.orderBy(col("total").desc())
mention_sorted.show()

# COMMAND ----------

top20_positive = mention_sorted.orderBy(col("positive").desc()).limit(20).toPandas()
plt.figure(figsize=(12, 6))
plt.bar(top20_positive["mention"], top20_positive["positive"], color='green')
plt.title("Top 20 Mentions with Positive Sentiment")
plt.xlabel("Mention")
plt.ylabel("Positive Tweet Count")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# COMMAND ----------

top20_negative = mention_sorted.orderBy(col("negative").desc()).limit(20).toPandas()
plt.figure(figsize=(12, 6))
plt.bar(top20_negative["mention"], top20_negative["negative"], color='red')
plt.title("Top 20 Mentions with Negative Sentiment")
plt.xlabel("Mention")
plt.ylabel("Negative Tweet Count")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10.0 Clean up and completion of your pipeline (3 points)
# MAGIC - using the utilities what streams are running? If any.
# MAGIC - Stop all active streams
# MAGIC - print out the elapsed time of your notebook. Note: In the includes there is a variable START_TIME that captures the starting time of the notebook.

# COMMAND ----------

active_streams = spark.streams.active
if len(active_streams) > 0:
    print("Active Streams:")
    for stream in active_streams:
        stop_named_stream(spark, stream)

else:
    print("No active streams.")

end_time = time.time() 
elapsed = end_time - START_TIME
print(f"\nTotal Elapsed Time: {elapsed:.2f} seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11.0 How Optimized is your Spark Application (Grad Students Only) (5 points)
# MAGIC Graduate students (registered for the DSCC-402 section of the course) are required to do this section.  This is a written analysis using the Spark UI (link to screen shots) that support your analysis of your pipelines execution and what is driving its performance.
# MAGIC Recall that Spark Optimization has 5 significant dimensions of considertation:
# MAGIC - Spill: write to executor disk due to lack of memory
# MAGIC - Skew: imbalance in partition size
# MAGIC - Shuffle: network io moving data between executors (wide transforms)
# MAGIC - Storage: inefficiency due to disk storage format (small files, location)
# MAGIC - Serialization: distribution of code segments across the cluster
# MAGIC
# MAGIC Comment on each of the dimentions of performance and how your impelementation is or is not being affected.  Use specific information in the Spark UI to support your description.  
# MAGIC
# MAGIC Note: you can take sreenshots of the Spark UI from your project runs in databricks and then link to those pictures by storing them as a publicly accessible file on your cloud drive (google, one drive, etc.)
# MAGIC
# MAGIC References:
# MAGIC - [Spark UI Reference Reference](https://spark.apache.org/docs/latest/web-ui.html#web-ui)
# MAGIC - [Spark UI Simulator](https://www.databricks.training/spark-ui-simulator/index.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ENTER YOUR MARKDOWN HERE