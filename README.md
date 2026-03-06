\# Self-Correcting LLM Analytical Agent



A \*\*Self-Correcting Large Language Model (LLM) Agent\*\* designed for \*\*SQL reasoning, statistical analysis, and hallucination detection\*\*.

The system evaluates LLM-generated outputs using analytical validation and statistical grounding to improve reliability in data analysis tasks.



---



\## Overview



Large Language Models often generate \*\*hallucinated results\*\* when performing analytical reasoning or SQL-based data analysis.

This project introduces a \*\*self-correcting analytical pipeline\*\* that:



\* Generates analytical responses using an LLM

\* Executes SQL queries on structured databases

\* Performs statistical validation

\* Detects hallucinations using evaluation metrics

\* Corrects outputs through analytical feedback loops



The system is designed as a \*\*research-oriented framework\*\* for improving \*\*trustworthy AI in data analysis workflows\*\*.



---



\## Key Features



\* LLM-based analytical reasoning

\* SQL query generation and execution

\* Retrieval-Augmented Generation (RAG)

\* Statistical hypothesis testing

\* Automated hallucination detection

\* Evaluation metrics for analytical correctness

\* Modular research pipeline

\* Interactive interface using Streamlit



---



\## Project Architecture



User Query

↓

LLM Reasoning Agent

↓

SQL Generation

↓

Database Execution

↓

Statistical Analysis Engine

↓

Evaluation Metrics

↓

Hallucination Detection

↓

Corrected Analytical Output



---



\## Tech Stack



\* Python

\* Streamlit

\* Pandas

\* NumPy

\* Scikit-learn

\* SQL

\* Retrieval-Augmented Generation (RAG)



---



\## Project Structure



```

llm-analytical-agent



agent.py

app.py

rag.py

prompting.py

research\_pipeline.py

stats\_engine.py

hypothesis\_engine.py

evaluation\_metrics.py

config.py

run\_app.py



requirements.txt

README.md



data/

&nbsp;  chinook.db

&nbsp;  sakila.db

&nbsp;  sql-murder-mystery.db

```



---



\## Example Use Case



Example analytical question:



\*\*"Is there a statistically significant difference in sales across music genres?"\*\*



Pipeline execution:



1\. LLM generates SQL query

2\. Database executes query

3\. Statistical engine performs hypothesis testing

4\. Evaluation module checks numerical grounding

5\. System flags potential hallucinations



Output includes:



\* Statistical results

\* Analytical explanation

\* Confidence evaluation



---



\## Evaluation Metrics



The system evaluates analytical reliability using metrics such as:



\* \*\*Hallucination Rate (HR)\*\*

\* \*\*Numerical Grounding Score (NGS)\*\*

\* \*\*Analytical Consistency Score (ACS)\*\*

\* \*\*Confidence Calibration Error (CCE)\*\*



These metrics help quantify \*\*trustworthiness of LLM-generated analytical outputs\*\*.



---



\## Installation



Clone the repository:



```

git clone https://github.com/HIIAYUSHI/llm-analytical-agent.git

```



Navigate to the project directory:



```

cd llm-analytical-agent

```



Install dependencies:



```

pip install -r requirements.txt

```



Run the application:



```

streamlit run app.py

```



---



\## Future Improvements



\* Integration with advanced LLMs

\* Enhanced hallucination detection mechanisms

\* Model interpretability modules

\* Cloud deployment

\* Interactive analytics dashboard



---



\## Author



\*\*Ayushi Bisht\*\*



Student – Data Science \& Statistics

Interested in \*\*Machine Learning, AI Systems, and Trustworthy LLMs\*\*



