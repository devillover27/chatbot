# Portfolio RAG Chatbot

A comprehensive Portfolio Analysis Chatbot built with a **FastAPI** backend and a **React Native (Expo)** frontend. This application uses Retrieval-Augmented Generation (RAG) to provide intelligent insights into financial portfolios based on historical trades and current holdings.

## 🚀 Features

- **RAG-Powered Chat**: Intelligent responses based on your specific portfolio data (`trades.csv` and `holdings.csv`).
- **Real-time Streaming**: Chat responses stream in real-time for a smooth experience.
- **Deep Navy UI**: A modern, premium dark-themed interface.
- **Data Insights**: Analyze trading patterns and current asset distribution.

---

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI (Python)
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **LLM**: Anthropic Claude API (via `google-genai` or similar integration)
- **Data Processing**: Pandas, NumPy
- **Server**: Uvicorn

### Frontend
- **Framework**: React Native with Expo
- **Routing**: Expo Router
- **Icons**: Expo Vector Icons
- **Styling**: Native Components with custom premium aesthetics

---

## 📦 Installation & Setup

### 1. Prerequisites
- Python 3.10+
- Node.js (LTS)
- npm or yarn
- Expo Go app (for mobile testing) or Android/iOS Emulator

### 2. Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure Environment Variables:
   Create a `.env` file in the `backend/` folder (standard templates):
   ```env
   ANTHROPIC_API_KEY=your_api_key_here
   ```

### 3. Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd portfolio-chatbot
   ```
2. Install dependencies:
   ```bash
   npm install
   ```

---

## 🏃 Driving the Application

### Start the Backend
From the `backend/` directory:
```bash
python main.py
```
The server will start at `http://localhost:8000`.

### Start the Frontend
From the `portfolio-chatbot/` directory:
```bash
npx expo start
```
- Press **'a'** for Android emulator.
- Press **'i'** for iOS simulator.
- Press **'w'** for web browser.
- Scan the QR code with the **Expo Go** app to run on a physical device.

---

## 📂 Project Structure

```text
.
├── backend/                # FastAPI Python Backend
│   ├── data/               # CSV Data Files
│   ├── models/             # Pydantic Schemas
│   ├── rag/                # RAG Logic (Embedder, Loader, Retriever)
│   ├── routes/             # API Endpoints (Chat, Search, Summary)
│   └── main.py             # Entry Point
├── portfolio-chatbot/      # React Native Expo Frontend
│   ├── src/                # Source Code
│   │   ├── api/            # API Client
│   │   ├── components/     # UI Components
│   │   └── app/            # Expo Router Pages
│   └── package.json
├── trades.csv              # Raw Trade Data
└── holdings.csv            # Raw Holdings Data
```

---

## 📄 License
This project is licensed under the MIT License.
