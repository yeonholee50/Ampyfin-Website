# 🌐 **AmpyFin Trading Bot Website & API**  
Welcome to the **AmpyFin Frontend & API** repository! This project is the frontend and API integration for **AmpyFin**, an advanced AI-powered trading bot designed for the NASDAQ-100.  

This repository focuses on building the **user interface** and **API endpoints** that communicate with the core trading bot, providing a seamless experience for users and contributors.

---

## **🚀 Introduction**  
**AmpyFin** empowers traders with a fully autonomous trading solution, using diverse strategies to optimize performance.  
The backend logic and machine learning models are hosted separately in the core **AmpyFin** repository:  
👉 [AmpyFin Core Repository (ML & Trading Logic)](https://github.com/yeonholee50/AmpyFin)

---

## **🌟 Features**  

- **Frontend**: React-based user interface to monitor and manage AmpyFin's performance.  
- **API Endpoints**: Backend API to access real-time trading data, rankings, and portfolio performance.  
- **Data Storage**: Integration with MongoDB to securely store and retrieve data.  
- **Dynamic Algorithm Ranking**: Display the current rankings of trading algorithms.  
- **Portfolio Overview**: View AmpyFin's overall trading performance and holdings.  

---

## **🛠️ Installation**  

### 1. **Clone the Repository**  
```bash
git clone https://github.com/yeonholee50/AmpyFin-Website.git
cd AmpyFin-Website
```

### 2. **Install Dependencies**  
```bash
pip install -r requirements.txt
```

### 3. **Set Up MongoDB**  
- Sign up for a MongoDB cluster (e.g., via [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)).  
- Create a database for AmpyFin data storage.  
- Update the `config.py` file with your MongoDB credentials.

---

## **⚙️ Configuration**  

Create a `config.py` file based on `config_template.py` and add your API keys and MongoDB credentials:  

```python
API_KEY = "your_alpaca_api_key"
SECRET_API = "your_alpaca_secret_api_key"
MONGO_DB_USER = "your_mongo_user"
MONGO_DB_PASS = "your_mongo_password"
```

---

## **🖥️ Running the Project**  

To start the frontend and API server:  

### Start the API Server  
```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

### Start the Frontend (React)  
```bash
cd ampyfin-react
npm install
npm run build
serve -s build
```

The frontend will be accessible at `http://localhost:3000`, and the API server will run on `http://127.0.0.1:8000`.

---

## **📡 API Endpoints**  

### Main API Endpoint  
`https://ampyfin-api-app.onrender.com/`  
The main entry point for AmpyFin's trading bot API.

### **Rankings Endpoint**  
`GET /rankings`  
Returns the current rankings of trading algorithms.

### **Holdings Endpoint**  
`GET /holdings`  
Provides the current holdings of the trading bot.

### **Portfolio & Major ETFs Endpoint**  
`GET /portfolio_percentage`  
Returns the total profit percentage of the bot since going live, along with benchmark data for QQQ and SPY ETFs.

### **Ticker Sentiment Endpoint**  
`GET /ticker/{ticker}`  
Provides the trading bot's sentiment on a specific ticker. Replace `{ticker}` with the desired stock symbol.

---

## **📂 File Structure**  

### `app.py`  
- **Objective**: Manages backend API endpoints for AmpyFin.  
- **Features**:  
  - Serves data to the frontend.  
  - Handles requests for rankings, holdings, and portfolio performance.

### `ampyfin-react/`  
- **Objective**: React-based frontend interface for AmpyFin.  
- **Features**:  
  - Displays real-time rankings and portfolio data.  
  - Provides user-friendly access to API data.

### Helper Files  
- **`config.py`**: Stores API keys and database credentials. Please make before using  
- **`helper_files.client_helper.py`**: Common utility functions used for operations.

---

## **📈 Usage**  

1. **Start the Backend API**:  
```bash
python apo_server.py
```

2. **Start the Frontend (React)**:  
```bash
npm start
```

---

## **📝 Notes**  

- The bot is limited to 250 API calls per day (Polygon API free tier).  
- Contributions are welcome! Feel free to open a pull request or submit issues for bugs and feature requests.

---

## **📜 License**  
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

**Happy Trading!** 🚀