# AI Agent Twins for Real-Time Decision Systems

## Overview
This project explores the design and implementation of AI Agent Twins operating within real-time environments. The system simulates autonomous agents making decisions under dynamic conditions, with a focus on reinforcement learning, latency-sensitive processing, and model performance trade-offs.

The goal of this project is to evaluate how different AI approaches perform in real-time, high-frequency decision-making scenarios, where both accuracy and responsiveness are critical.

---

## Key Features

- Dual-agent ("twin") architecture for comparative evaluation  
- Real-time decision loop with continuous state updates  
- Reinforcement learning-based policy optimisation  
- Support for multiple model configurations (RL / ML / hybrid approaches)  
- Performance benchmarking across latency, accuracy, and decision quality  
- Modular architecture for experimenting with different strategies  

---

## Core Concepts

### Agent Twins
Two independent agents operate in parallel under identical conditions, allowing direct comparison of:
- Decision-making strategies  
- Model performance  
- Adaptability to dynamic environments  

### Real-Time Environment
The system processes live or simulated streaming data, requiring:
- Low-latency inference  
- Continuous state updates  
- Efficient execution pipelines  

### Reinforcement Learning
Agents learn optimal strategies through:
- Custom reward functions  
- Environment interaction loops  
- Policy optimisation techniques  

---

## Architecture

Environment → State Processing → Agent Policies → Decision Execution → Feedback Loop

Components:
- Environment Module – simulates or processes real-time data  
- Agent Module – implements decision-making logic (RL / ML models)  
- Evaluation Module – tracks performance metrics  
- Control Loop – manages continuous execution and updates  

---

## Tech Stack

- Language: Python  
- Frameworks/Libraries:  
  - PyTorch  
  - NumPy / Pandas  
  - Custom RL implementations  
- Tools:  
  - Docker (for reproducibility)  
  - Git (version control)  

---

## Evaluation Metrics

The system evaluates agents across:

- Latency – response time per decision  
- Accuracy – correctness of predictions/actions  
- Stability – consistency over time  
- Reward Performance – cumulative reward in RL setting  

---

## Experimental Focus

- Comparing different RL policies in real-time environments  
- Analysing trade-offs between speed and decision quality  
- Evaluating model performance under high-frequency input streams  

---

## Project Structure

/src            Core logic and agent implementations  
/models         Model definitions and training logic  
/environment    Simulation or real-time data handling  
/evaluation     Metrics and benchmarking tools  
/scripts        Utility and execution scripts  

---

## Getting Started

Clone the repository:
git clone https://github.com/adriann1z/AI-ML-Agent-Twins-DISS.git
cd AI-ML-Agent-Twins-DISS

Install dependencies:
pip install -r requirements.txt

Run the system:
python main.py

---

## Applications

This project is relevant for:

- Algorithmic trading systems  
- Autonomous agents and robotics  
- Real-time decision systems  
- AI research and experimentation  

---

## Future Improvements

- Integration with live data streams (e.g. financial APIs)  
- Scaling to multi-agent environments  
- Advanced RL techniques (e.g. PPO, DQN variants)  
- GPU acceleration for faster inference  

---

## Author

Adrian Wrezel  
Computer Science Graduate – AI & Machine Learning  
GitHub: https://github.com/adriann1z  

---

## Notes

This project was developed as part of a dissertation exploring real-time AI systems and agent-based learning architectures.
