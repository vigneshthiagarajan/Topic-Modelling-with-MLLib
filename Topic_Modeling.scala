// Databricks notebook source
import org.apache.spark.ml.feature.{Tokenizer, CountVectorizer}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.functions._

// COMMAND ----------

val corpus = sc.wholeTextFiles("/FileStore/tables/FileStore/tables/Sherlock.txt").map(_._2).map(_.toLowerCase())
//only worked with whole text files

// COMMAND ----------

//corpus.take(2)

// COMMAND ----------

val corpus_body = corpus
//corpus_body.collect

// COMMAND ----------

val corpus_df = corpus_body.toDF("corpus")

// COMMAND ----------

//display(corpus_df)

// COMMAND ----------

import org.apache.spark.ml.feature.RegexTokenizer
	// Set params for RegexTokenizer
	val tokenizer = new RegexTokenizer()
  .setPattern("[\\W_]+")
  .setMinTokenLength(5) // Filter away tokens with length < 4
  .setInputCol("corpus")
  .setOutputCol("tokens")

	// Tokenize document
	val tokenized_df = tokenizer.transform(corpus_df)

// COMMAND ----------

display(tokenized_df)

// COMMAND ----------

val remover = new StopWordsRemover()
  .setInputCol("tokens")
  .setOutputCol("filtered")


// COMMAND ----------

// Create new DF with Stopwords removed
val filtered_df = remover.transform(tokenized_df)

// COMMAND ----------

//display(filtered_df)

// COMMAND ----------

val cv = new CountVectorizer()
  .setInputCol("filtered")
  .setOutputCol("features")
  .setVocabSize(10000)
  .setMinTF(1)
  .setMinDF(1)
  .setBinary(true)
val cvFitted = cv.fit(filtered_df)
val prepped = cvFitted.transform(filtered_df)


// COMMAND ----------

//display(prepped)

// COMMAND ----------

import org.apache.spark.ml.clustering.LDA
val lda = new LDA().setK(5).setMaxIter(5)
println(lda.explainParams())
val model = lda.fit(prepped)

// COMMAND ----------

val vocabList = cvFitted.vocabulary
val termsIdx2Str = udf { (termIndices: Seq[Int]) => termIndices.map(idx => vocabList(idx)) }

import org.apache.spark.sql.functions._
val topics = model.describeTopics(maxTermsPerTopic = 1)//changed from 5 to 1 to get one term from each topic
  .withColumn("terms", termsIdx2Str(col("termIndices")))
display(topics.select("topic", "terms", "termWeights"))

// COMMAND ----------


