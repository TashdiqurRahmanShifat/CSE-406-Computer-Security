# CSE-406 Computer Security

![Security](https://img.shields.io/badge/Course-CSE--406-blue)
![Python](https://img.shields.io/badge/Language-Python-green)
![License](https://img.shields.io/badge/License-Educational-yellow)

## 📚 Course Overview

This repository contains assignments and projects for the **CSE-406 Computer Security** course. The projects cover fundamental security concepts including cryptography implementations, side-channel attacks, and network security vulnerabilities.

## 🔧 Prerequisites

- Python 3.7+
- Required Python packages:
  - `BitVector` (for cryptography)
  - `Flask` (for web applications)
  - `torch` (for machine learning)
  - `numpy`, `matplotlib` (for data analysis)
  - `scapy` (for network operations)
  - `sqlalchemy` (for database operations)
  - `sklearn` (for machine learning utilities)

## 📁 Project Structure

```
CSE-406-Computer-Security/
├── README.md
├── Cryptography/                          # Assignment 1: Cryptographic Algorithms
│   ├── 2005102_aes.py                    # AES encryption/decryption implementation
│   ├── 2005102_bitvector_demo.py         # BitVector operations and S-box implementation
│   ├── 2005102_client.py                 # Client-side secure communication
│   ├── 2005102_ecc.py                    # Elliptic Curve Cryptography implementation
│   ├── 2005102_server.py                 # Server-side secure communication
│   └── CSE406 - Assignment 1.pdf         # Assignment specifications
├── Side Channel Attack/                   # Assignment 2: Website Fingerprinting Attack
│   ├── app.py                            # Flask web application for attack simulation
│   ├── collect.py                        # Data collection script for traces
│   ├── database.py                       # Database management for collected data
│   ├── train.py                          # Neural network training for classification
│   ├── model_metadata.json               # Model configuration and metadata
│   ├── specification.pdf                 # Project requirements
│   ├── 2005102_report.pdf               # Technical report
│   ├── README.md                         # Project-specific documentation
│   ├── DriveLink.txt                     # External resources link
│   ├── Bonus/                            # Additional bonus implementation
│   │   ├── train.py                      # Enhanced training script
│   │   ├── model_metadata.json           # Bonus model configuration
│   │   └── Result.txt                    # Performance results
│   └── static/                           # Web interface files
│       ├── index.html                    # Main web interface
│       ├── index.js                      # Frontend JavaScript
│       ├── warmup.js                     # Browser warm-up script
│       └── worker.js                     # Web worker for background tasks
└── IPv6 Flooding Attack Project/         # Assignment 3: Network Security Attack
    ├── attack.py                         # IPv6 Router Advertisement flooding script
    ├── Attack_Design.pdf                 # Attack methodology documentation
    ├── IPv6_attack_report.pdf            # Comprehensive attack analysis
    └── Commands for defense.txt          # Defensive countermeasures
```

## 🚀 Projects Description

### 1. Cryptography Implementation
**Objective**: Implement fundamental cryptographic algorithms from scratch.

**Key Features**:
- **AES Encryption/Decryption**: Complete implementation with key expansion and all transformation rounds
- **Elliptic Curve Cryptography**: Point operations and key generation
- **Secure Client-Server Communication**: Demonstrating practical cryptographic protocols
- **BitVector Operations**: Low-level bit manipulation for cryptographic functions

**Technologies**: Python, BitVector library

### 2. Side Channel Attack - Website Fingerprinting
**Objective**: Develop a machine learning-based attack to identify websites through timing side-channels.

**Key Features**:
- **Data Collection**: Web-based interface for gathering timing traces
- **Machine Learning Pipeline**: Neural network classifiers for website identification
- **Real-time Analysis**: Flask web application for interactive attack simulation
- **Performance Evaluation**: Comprehensive metrics and visualizations

**Technologies**: Python, Flask, PyTorch, NumPy, Matplotlib, SQLAlchemy

**Attack Methodology**:
1. Collect timing traces from web browsing patterns
2. Extract features from timing data
3. Train neural network classifiers
4. Evaluate attack success rates

### 3. IPv6 Flooding Attack
**Objective**: Demonstrate network-level security vulnerabilities in IPv6 autoconfiguration.

**Key Features**:
- **Router Advertisement Flooding**: Exploit IPv6 neighbor discovery protocol
- **Realistic Traffic Generation**: Create believable spoofed packets
- **Attack Logging**: Database tracking of attack attempts
- **Defense Analysis**: Documentation of countermeasures

**Technologies**: Python, Scapy, SQLite

## 🔧 Installation & Setup

### General Requirements
```bash
pip install BitVector flask torch numpy matplotlib scapy sqlalchemy scikit-learn
```

### Project-Specific Setup

#### Cryptography Project
```bash
cd Cryptography/
python 2005102_server.py  # Start server
python 2005102_client.py  # Run client (in separate terminal)
```

#### Side Channel Attack Project
```bash
cd "Side Channel Attack"/
python app.py  # Start Flask application
# Access http://localhost:5000 in browser
```

#### IPv6 Attack Project
```bash
cd "IPv6 Flooding Attack Project"/
# Requires administrative privileges
sudo python attack.py <network_interface> --username <user> --count <packetcount>
```