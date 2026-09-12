<h2>Project Overview</h2> <hr>

This project builds an AI-assisted customer support agent for AppleSupport using real customer support conversations from the Customer Support on Twitter dataset.

For each incoming customer message, the system:

Classifies the message into one of 12 support intents.
Retrieves the 3 most similar historical AppleSupport conversations.
Generates a customer-facing reply using the predicted intent and retrieved evidence.
Decides whether the request should be handled automatically or escalated.

The system is designed to keep its intermediate decisions visible, making the classification, retrieval, response generation, and escalation behavior easier to inspect and evaluate.

<h2>System Pipeline</h2> <hr>
Customer message
        ↓
Intent Classification
TF-IDF + Logistic Regression
        ↓
Historical Evidence Retrieval
TF-IDF + Cosine Similarity
        ↓
Top 3 AppleSupport conversations
        ↓
Response Generation
openai/gpt-oss-20b through Groq
        ↓
Escalation Decision
Classifier confidence + retrieval similarity
<h2>Dataset</h2> <hr>

The project uses the Customer Support on Twitter dataset and focuses on AppleSupport conversations.

The dataset contains approximately 106K direct customer-to-AppleSupport response pairs used to build the historical support data.

For intent classification, 200 direct AppleSupport customer-response pairs were randomly sampled using random_state=42 and manually labeled. Messages that were too vague to label reliably were excluded, resulting in 181 usable labeled examples.

The labeled data covers 12 support intents and is used for classifier development and 5-fold stratified cross-validation.

The historical AppleSupport conversations are stored in DATA/processed/apple_history.csv.

The raw dataset is stored in DATA/raw/twcs.csv.

<h2>Intent Taxonomy</h2> <hr>

The system uses 12 intent categories derived from recurring AppleSupport customer issues.

App Issue — Apps crashing, not opening, or malfunctioning.
iOS Update Issue — iOS update problems or update-related bugs.
Device Issue — General iPhone/iPad hardware or device problems.
Battery Issue — Battery drain, battery health, charging, or Low Power Mode.
Screen/UI Issue — Screen, display, keyboard, autocorrect, or visual/UI glitches.
Apple Music Issue — Apple Music-specific problems.
App Store/Purchase Issue — App Store, iTunes, purchases, subscriptions, or payments.
Account/Security Issue — Apple ID, password, hacked account, phishing, or security issues.
iCloud/Data Issue — iCloud, backup, restoring, transferring photos, or data issues.
Connectivity Issue — Wi-Fi, Bluetooth, cellular, or network connectivity.
Support/Repair Issue — Apple Store, repair, appointments, or telephone support.
Product Order Issue — Product purchases, availability, pre-orders, or delivery estimates.
<h2>Results</h2> <hr>

The intent classifier was evaluated using 5-fold stratified cross-validation on 181 manually labeled AppleSupport examples.

Metric	Result
Accuracy	38.68% ± 7.47%
Macro F1	23.94% ± 4.18%
Weighted F1	35.32% ± 6.82%

Historical retrieval:

Mean best-match similarity: 0.3636
Median best-match similarity: 0.3166
Top 3 historical AppleSupport conversations retrieved for each query

Reply evaluation was performed on 30 sampled examples. 23 completed the full generation and judging process and were used for the reported scores.

Aspect	Score / 5
Groundedness	4.87
Helpfulness	3.70
Professionalism	4.30
Overall quality	3.87

The LLM judge was also compared with manual human scoring on the same 23 replies using Spearman correlation, exact agreement, and agreement within ±1 point.

<h2>Repository Structure</h2> <hr>
Hiver-SDE-Assignment/
├── DATA/
│   ├── raw/
│   │   └── twcs.csv
│   └── processed/
│       ├── apple_history.csv
│       ├── apple_labeled_clean.csv
│       ├── apple_labeled_sample.csv
│       └── apple_support_sample.csv
│
├── evaluation/
│   ├── baseline_majority.py
│   ├── baseline_tfidf.py
│   ├── calculate_agreement.py
│   ├── cross_validate_classifier.py
│   ├── evaluate_classifier.py
│   ├── evaluate_replies_groq.py
│   ├── evaluate_retrieval.py
│   ├── human_judge.py
│   └── local_evaluation.py
│
├── models/
│   ├── intent_classifier.joblib
│   └── intent_vectorizer.joblib
│
├── results/
│   ├── human_llm_agreement.csv
│   ├── reply_evaluation_results.csv
│   └── retrieval_evaluation_clean.csv
│
├── scripts/
│   ├── analyze_apple_intents.py
│   ├── analyze_dataset.py
│   ├── clean_apple_labels.py
│   ├── inspect_brand_pairs.py
│   ├── label_apple_sample.py
│   ├── prepare_apple_history.py
│   └── summarize_apple_sample.py
│
├── src/
│   ├── apple_support_agent.py
│   ├── intent_classifier.py
│   └── retrieve_history.py
│
├── README.md
└── requirements.txt
<h2>Setup</h2> <hr>

<strong>Python version</strong>

Python 3.10+ is recommended.

<strong>Create and activate a virtual environment</strong>

Windows:

python -m venv .venv
.venv\Scripts\activate

macOS/Linux:

python3 -m venv .venv
source .venv/bin/activate

<strong>Install dependencies</strong>

pip install -r requirements.txt

The processed datasets and trained intent model are already included in the repository, so no preprocessing or model training is required to run the main agent.

<h2>API Key</h2> <hr>

The response-generation and LLM-judge components use the Groq API with the open-weight openai/gpt-oss-20b model.

Set your own Groq API key as an environment variable before running the components that require the API.

<strong>Windows PowerShell</strong>

$env:GROQ_API_KEY="your_api_key"

<strong>macOS/Linux</strong>

export GROQ_API_KEY="your_api_key"

The API key is not included in the repository.

<h2>Run the Agent</h2> <hr>

From the repository root, run:

python src/apple_support_agent.py

Enter a customer message when prompted.

The agent displays:

Predicted intent
Classifier confidence
Top 3 historical AppleSupport conversations and their similarity scores
Generated customer-facing reply
Escalation decision
Reason for the escalation or automated handling

Type quit to exit the agent.

<h2>Reproduce the Evaluation</h2> <hr>

<strong>Intent classification</strong>

python evaluation/local_evaluation.py

<strong>Majority-class baseline</strong>

python evaluation/baseline_majority.py

<strong>TF-IDF + Logistic Regression baseline</strong>

python evaluation/baseline_tfidf.py

<strong>Historical retrieval evaluation</strong>

python evaluation/evaluate_retrieval.py

<strong>Reply generation and LLM-judge evaluation</strong>

python evaluation/evaluate_replies_groq.py

The reply evaluation requires the GROQ_API_KEY environment variable.

<strong>Human–LLM judge agreement</strong>

python evaluation/calculate_agreement.py

The evaluation scripts report the corresponding classification, retrieval, reply-quality, and human–LLM agreement metrics described in the report.

<h2>Baselines</h2> <hr>

Two simple baselines were used to evaluate the intent classification component.

<strong>Majority Class</strong>

Every message is assigned the most common intent, ios_update_issue.

Accuracy: 35.14%

Macro F1: 4.73%

<strong>TF-IDF + Logistic Regression</strong>

The customer message is converted into TF-IDF word and word-pair features and classified using Logistic Regression with balanced class weights.

Mean 5-fold Accuracy: 38.68% ± 7.47%

Mean Macro F1: 23.94% ± 4.18%

Mean Weighted F1: 35.32% ± 6.82%

The classifier improves over the majority baseline, but the improvement is modest because the manually labeled dataset is small and imbalanced.

<h2>Escalation Decision</h2> <hr>

The escalation policy combines classifier confidence with the highest historical retrieval similarity.

The system escalates when confidence and historical evidence are insufficient for reliable automated handling. Strong historical evidence can allow automated handling even when classifier confidence is lower.

Each escalation decision includes a reason, making the decision inspectable.

<h2>Limitations</h2> <hr>

The main limitations identified during evaluation are:

The labeled dataset is small and imbalanced, with some intents having only a few examples.
TF-IDF classification struggles with semantically similar intent categories.
TF-IDF retrieval can be unreliable for short or generic customer messages.
High lexical similarity does not always indicate useful resolution evidence.
Generated replies can occasionally introduce troubleshooting information not supported by the retrieved historical responses.
The LLM-judge evaluation was based on 23 successfully completed examples.
Human–LLM agreement was weaker for helpfulness and overall quality.
The current system does not use full conversation-level context or perform real support actions.
<h2>Next Steps</h2> <hr>

With one more week, I would prioritize:

Expand the labeled dataset, particularly minority intents and difficult boundary cases.
Evaluate semantic embedding-based retrieval against the current TF-IDF approach.
Add retrieval-quality gating so weak historical evidence leads to escalation.
Add stronger verification of generated troubleshooting against retrieved evidence.
Build a larger held-out end-to-end evaluation set.
Tune escalation thresholds using validation data and explicit risk trade-offs.

Priority: better data, better retrieval, safer generation, and more reliable escalation rather than adding unnecessary system complexity.