# Architecture Diagram

The Loan Decision Support Platform is structured as a 10-phase integrated enterprise application.

```mermaid
graph TD
    subgraph Data Layer
        A[Historical Loan Data] --> B(Data Processor & Scaler)
        A --> C(Isolation Forest Anomaly Detection)
        A --> D(K-Means Clustering)
    end

    subgraph Core Predictive Layer
        B --> E(Random Forest Classifier)
        E --> F[Loan Prediction Core]
        E --> G(SHAP Explainability Engine)
    end

    subgraph Analytical Layer
        F --> H[Risk Analytics Scoring]
        F --> I[Financial Health Engine]
        C --> J[Fraud Risk Assessment]
        D --> K[Customer Segmentation]
    end

    subgraph AI Intelligence Layer
        G --> L[Decision Intelligence]
        H --> M[AI Loan Optimization Engine]
        I --> M
        J --> M
    end

    subgraph Business Layer
        H --> N[Business Intelligence Dashboard]
        I --> N
        J --> N
        K --> N
        A --> N
    end

    subgraph Output Layer
        M --> O[What-If Simulator]
        L --> P[PDF Report Generator]
        M --> P
    end

    %% User Interaction Flow
    User((Loan Officer)) --> F
    User --> O
    User --> N
```

## Module Definitions
1. **Business Intelligence Dashboard**: Macro-level portfolio aggregation and executive KPIs.
2. **Loan Prediction**: The primary entry point for individual applicant evaluation.
3. **Dataset Insights**: Raw exploration of the underlying training data distributions.
4. **Risk Analytics**: Credit and demographic risk stratification.
5. **Financial Health**: Debt-to-Income and affordability analysis.
6. **What-If Simulator**: Real-time manipulation of features to see outcome shifts.
7. **Decision Intelligence**: Transparent AI reasoning mapping exact SHAP value impacts.
8. **Customer Segmentation**: Unsupervised clustering assigning applicants to 5 distinct personas.
9. **AI Loan Optimization**: Algorithmic goal-seeking to rescue rejected applicants with restructured terms.
10. **Fraud Risk Assessment**: Dual-layer heuristic and anomaly detection for risk mitigation.
