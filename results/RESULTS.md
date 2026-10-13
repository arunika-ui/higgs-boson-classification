| Model | Test ROC AUC | Test AMS | Events selected | Train time |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.8168 | 2.006 | 34.0% | 3 s |
| Neural Network (MLP) | 0.9107 | 3.472 | 17.5% | 40 s |
| XGBoost **(best)** | 0.9131 | 3.593 | 15.5% | 10 s |

_Test set: 450,000 held-out events (the challenge's private leaderboard split)._
