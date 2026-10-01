# MLOps Technical Interview & Portfolio Pitch Guide

This guide prepares you to speak with hiring managers, senior ML engineers, and recruiters about this project with confidence and technical depth.

---

## 1. The 60-Second Elevator Pitch
> *"I built a production-grade customer churn prediction microservice designed to bridge the gap between exploratory Jupyter notebooks and reliable cloud deployments. The pipeline uses LightGBM and Scikit-Learn ColumnTransformers to eliminate train-serve skew, logs experiments and model lineage to MLflow, and is served via an asynchronous FastAPI service with strict Pydantic v2 validation and LRU caching. To demonstrate real-world engineering discipline, I containerized the service using a multi-stage Docker build optimized for 512MB RAM free-tier instances, cutting image size from 900MB to 265MB with sub-3ms inference latency. Merges are automated via a sub-2-minute GitHub Actions CI/CD pipeline that runs automated pytest checks and deploys straight to Render."*

---

## 2. Common MLOps Technical Interview Questions & Model Answers

### Q1: "Why did you choose LightGBM / shallow XGBoost over a Deep Neural Network?"
**Strong Answer:**
> *"For tabular customer churn data with under 50,000 rows, gradient-boosted decision trees consistently outperform deep neural networks in sample efficiency and calibration. More importantly, from an MLOps perspective, LightGBM produces a serialized model artifact under 200 KB that evaluates in under 3 milliseconds on a single CPU core. A PyTorch or TensorFlow neural network would require heavy runtime wheels, hundreds of megabytes of memory, and unnecessary GPU dependencies—violating our 512MB free-tier constraints without providing higher ROC-AUC."*

### Q2: "How did you solve 'Train-Serve Skew' in your project?"
**Strong Answer:**
> *"Train-serve skew occurs when data transformations applied during training diverge from those applied during real-time inference. Instead of running ad-hoc pandas transformations in a training script and rewriting parsing logic in the FastAPI handler, I created a shared `src/preprocessing.py` module and encapsulated the feature engineering and Scikit-Learn `ColumnTransformer` inside a unified Scikit-Learn `Pipeline`. When the model is saved to `models/model.pkl`, the preprocessor is baked into the pipeline object. In production, the API passes raw dictionaries into `pipeline.predict_proba()`, guaranteeing identical transformations with zero duplicated code."*

### Q3: "How did you optimize your Docker container to stay under 512MB RAM and 400MB image size?"
**Strong Answer:**
> *"I used three key techniques:*
> 1. *Multi-stage builds: In the first builder stage, compilers (`gcc`, `g++`) generate binary wheels. In the second runner stage, we use `python:3.10-slim` and copy only the pre-compiled wheels, discarding the heavy build toolchain.*
> 2. *Dependency splitting: I decoupled production runtime dependencies (`requirements.txt`) from training/dev packages (`mlflow`, `matplotlib`, `pytest`). Eliminating visualization libraries alone saved ~350MB of disk bloat and 150MB of RAM allocation.*
> 3. *Single-worker Uvicorn: By running a single asynchronous event-loop worker with a non-root user, the idle memory footprint stays under 90MB, leaving 80% of memory available for incoming request spikes."*

### Q4: "How does your system handle incoming bad or malicious inputs?"
**Strong Answer:**
> *"I leverage Pydantic v2's Rust-backed validation core at the FastAPI router level. Every feature has strict type definitions and boundary checks (`tenure` between 0 and 120, `monthly_charges` non-negative, and `contract_type` restricted via `Literal` types). Any invalid payload is rejected in under 0.1 milliseconds with an HTTP 400 Bad Request before touching model memory or executing Python logic, preventing unexpected runtime exceptions and Denial-of-Service vulnerabilities."*

### Q5: "What is your strategy for monitoring model performance and data drift in production?"
**Strong Answer:**
> *"Because ground-truth labels (whether a customer actually churned) often take 30 to 90 days to realize, we monitor for covariate shift (input data drift) using Evidently AI. In `app/main.py`, incoming prediction requests and extracted features are appended to `monitoring/prediction_logs.csv`. Our drift detection module (`src/drift_detection.py`) runs two-sample Kolmogorov-Smirnov tests on numerical distributions and Chi-Square tests on categorical distributions against baseline training splits. If drift exceeds a 30% threshold, it flags the need for model retraining."*

### Q6: "How would you scale this service to handle 10,000 requests per second?"
**Strong Answer:**
> *"To scale horizontally to 10k RPS:*
> 1. *Deploy on Kubernetes (EKS / GKE) using a Horizontal Pod Autoscaler (HPA) targeting 70% CPU utilization.*
> 2. *Place an NGINX or AWS Application Load Balancer (ALB) in front with TLS termination and round-robin routing.*
> 3. *Replace the single-node CSV prediction logger with asynchronous Pub/Sub or Kafka event streaming to decouple logging I/O from the request-response critical path.*
> 4. *Scale the in-memory LRU cache into an external Redis Cluster to share cached predictions across multiple container replicas."*
