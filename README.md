🎮 Tic Tac Toe AI Predictor

🧠 Machine Learning for Game State Classification

📌 About the Project

This project was developed as part of an Artificial Intelligence discipline at PUCRS, aiming to explore Machine Learning techniques applied to the analysis of Tic Tac Toe game states.

Instead of building an AI that plays the game, this system acts as a game state evaluator, capable of analyzing any 3x3 board configuration and correctly classifying its current state.

The project covers the full Machine Learning pipeline — from dataset construction and preprocessing to model training, evaluation, and comparison — culminating in an interactive terminal-based application.

🎯 Objective

The main goal of this project is to develop and evaluate Machine Learning models capable of classifying the state of a Tic Tac Toe game based on a given board.

The AI predicts one of the following classes:

▶️ Game in progress

❌ X wins

⭕ O wins

🤝 Draw


⚙️ Project Pipeline

The project follows a complete Machine Learning workflow:

🧩 1. Data Processing
Load and analyze the original dataset
Convert categorical values (x, o, b) into numerical format
Reclassify game states
Generate additional valid board states
Clean and validate data

⚖️ 2. Data Balancing
Ensure equal class distribution
Apply oversampling techniques

🧠 3. Feature Engineering
Count number of X and O moves
Compute difference between players
Enrich dataset with additional features

✂️ 4. Data Splitting
Training set
Validation set
Test set

🤖 5. Model Training

Train and tune multiple classification algorithms.

📊 6. Evaluation

Models are evaluated using:

Accuracy
Precision
Recall
F1-score

🏆 7. Model Comparison

Compare performance across models to identify the best solution.

🤖 Algorithms Used

The project implements and compares multiple approaches:

🔹 K-Nearest Neighbors (KNN)
🌳 Random Forest
🧠 Multi-Layer Perceptron (MLP)
⚡ Perceptron
🌲 Decision Trees (via Random Forest variations)

Each model was individually trained, tuned, and evaluated.

🎮 Interactive Application

The project includes a terminal-based interface where:

A human player competes against a random opponent
After each move, the AI evaluates the board state
The system provides real-time feedback:
Whether the game is ongoing
If a player has won
If the game ended in a draw

This allows practical validation of the trained models.

🚀 Technologies Used
Python
Pandas
NumPy
Scikit-learn
Joblib
Jupyter

📈 Results

The models were evaluated using standard classification metrics, enabling comparison in terms of:

Generalization capability
Class sensitivity
Robustness with balanced datasets

The goal is to identify which algorithm performs best for this classification problem.

💡 Key Learnings

During development, the following concepts were explored:

Data preprocessing and balancing
Feature engineering
Hyperparameter tuning
Model evaluation
Overfitting vs generalization
Practical application of AI in games

👨‍💻 Author

Developed by Giancarlo Brandalise, Kauã Souza, Leonardo Duarte, Lucas Vinhatti and Renato Forte.

⭐ Highlight

This project demonstrates a complete Machine Learning pipeline, from dataset construction to real-world application integration, with a strong focus on algorithm comparison.
