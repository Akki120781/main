# 📊 Stock Market Comparison Analysis

This project focuses on analyzing and predicting stock market trends using real-time data. Stock prices are fetched directly using the `yfinance` Python module, which allows access to historical and live data for multiple companies.

---

## ✅ Key Features

- 📥 Real-time data fetching via **yfinance**
- 📈 Visualization of daily, monthly, and yearly stock trends
- 🤖 Machine Learning-based predictions using:
  - Linear Regression
  - LSTM
  - RNN
- 🥇 **Accuracy comparison**: Linear Regression outperformed LSTM and RNN
- 🧠 Interactive **GUI-based dashboard** to explore stock trends across companies

This project integrates **data science**, **machine learning**, **finance**, and **Python GUI development** into one complete stock analysis tool.

---

## 🛠️ Run Locally

### 🔽 Clone the Project

```bash
  git clone https://github.com/Akki120781/main.git
```

Go to the project directory

```bash
  cd <my-project-directory>
```

Install dependencies

```bash
  python -m venv venv
  venv\Scripts\activate
  pip install yfinace
  pip install sklearn
  pip install tensorflow
  pip install matplotlib, numpy, pandas, seaborn
```

Start the server

```bash
  streamlit run stock_analysis.py
```



