# 📊 PortfolioIQ — RAG Chatbot

A full-stack AI-powered portfolio chatbot that uses **Retrieval-Augmented Generation (RAG)** to answer questions about your stock trades and holdings. Built with a **FastAPI** Python backend and a **React Native (Expo)** mobile frontend.

---

## 🗂️ Project Structure

```
chatbot/
├── backend/                  # Python FastAPI backend
│   ├── main.py               # App entry point
│   ├── requirements.txt      # Python dependencies
│   ├── .env                  # Environment variables (API keys)
│   ├── data/
│   │   ├── holdings.csv      # Your holdings data
│   │   └── trades.csv        # Your trades data
│   ├── models/               # Pydantic schemas
│   ├── rag/                  # RAG pipeline (loader, embedder, retriever)
│   └── routes/               # API routes (chat, search, summary)
├── portfolio-chatbot/        # React Native Expo frontend
│   ├── app/                  # Expo Router screens
│   ├── src/                  # Components, API clients, hooks
│   ├── package.json
│   └── app.json
├── holdings.csv              # Root-level CSV (reference)
├── trades.csv                # Root-level CSV (reference)
└── README.md
```

---

## ⚙️ Prerequisites

Make sure you have the following installed before starting:

| Tool | Version | Download |
|------|---------|----------|
| Python | 3.10+ | https://www.python.org/downloads/ |
| Node.js | 18+ | https://nodejs.org/ |
| npm | 9+ | Comes with Node.js |
| Expo CLI | Latest | `npm install -g expo-cli` |
| Git | Any | https://git-scm.com/ |

---

## 🔧 Backend Setup (FastAPI + Python)

### Step 1 — Navigate to the backend folder

```bash
cd backend
```

### Step 2 — Create a Python virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs the following packages:

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework for building the REST API |
| `uvicorn` | ASGI server to run FastAPI |
| `python-dotenv` | Load environment variables from `.env` |
| `pandas` | CSV data loading and manipulation |
| `numpy` | Numerical computations |
| `google-genai` | Google Gemini API for LLM responses |
| `sentence-transformers` | Local embedding model (`all-MiniLM-L6-v2`) |
| `scikit-learn` | Cosine similarity for RAG retrieval |
| `pydantic` | Data validation and schemas |
| `sse-starlette` | Server-Sent Events for streaming responses |

### Step 4 — Configure environment variables

Create or edit `backend/.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
TOP_K=8
EMBEDDING_MODEL=all-MiniLM-L6-v2
PORT=8000
```

> 🔑 Get your Gemini API key from: https://aistudio.google.com/app/apikey

### Step 5 — Run the backend server

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **Base URL:** `http://localhost:8000`
- **Swagger Docs:** `http://localhost:8000/docs`
- **API Endpoints:**
  - `POST /api/chat` — Streaming chat with RAG
  - `GET  /api/summary` — Portfolio summary
  - `GET  /api/search` — Semantic search over data

---

## 📱 Frontend Setup (React Native + Expo)

### Step 1 — Navigate to the frontend folder

```bash
cd portfolio-chatbot
```

### Step 2 — Install Node.js dependencies

```bash
npm install
```

This installs the following key packages:

| Package | Purpose |
|---------|---------|
| `expo` | Core Expo SDK |
| `expo-router` | File-based navigation |
| `react-native` | Cross-platform mobile framework |
| `@expo/vector-icons` | Icon library |
| `@react-native-async-storage/async-storage` | Local chat history persistence |
| `react-native-markdown-display` | Render markdown in chat responses |
| `react-native-safe-area-context` | Safe area handling |
| `react-native-screens` | Native screen management |
| `react-native-svg` | SVG support for charts |
| `react-native-web` | Web support for Expo |

### Step 3 — Configure the API URL

Open `src/api/client.ts` and make sure the `BASE_URL` points to your backend:

```ts
// For local development (Android Emulator)
const BASE_URL = 'http://10.0.2.2:8000';

// For local development (Physical Device / Web)
const BASE_URL = 'http://192.168.x.x:8000';  // Your machine's local IP

// For production
const BASE_URL = 'https://your-deployed-backend.com';
```

### Step 4 — Start the Expo development server

```bash
npm start
```

Then choose your target platform:

| Command | Platform |
|---------|---------|
| Press `a` | Android (emulator or device) |
| Press `i` | iOS simulator (macOS only) |
| Press `w` | Web browser |
| Scan QR code | Expo Go app on physical device |

Or run directly:
```bash
npm run android    # Android
npm run ios        # iOS
npm run web        # Web browser
```

---

## 🚀 Running the Full App (Both Together)

Open **two terminal windows** and run:

**Terminal 1 — Backend:**
```bash
cd backend
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
python main.py
```

**Terminal 2 — Frontend:**
```bash
cd portfolio-chatbot
npm start
```

---

## 📁 Data Files

The chatbot reads from two CSV files in `backend/data/`:

| File | Description |
|------|-------------|
| `holdings.csv` | Current stock holdings (symbol, quantity, price, etc.) |
| `trades.csv` | Historical trade data (buy/sell transactions) |

You can replace these with your own CSV files. The RAG pipeline will automatically load and index them on startup.

---

## 🌐 API Reference

### POST `/api/chat`
Streams a chat response using RAG over your portfolio data.

```json
{
  "message": "What is my best performing stock?",
  "history": []
}
```

### GET `/api/summary`
Returns a structured summary of your portfolio.

### GET `/api/search?q=AAPL`
Performs semantic search over your portfolio data.

---

## 🔒 Security Notes

> ⚠️ **Important:** Never commit real API keys to a public repository.
> - Add `.env` to your `.gitignore`
> - Use environment variables in production
> - Rotate your API keys if they were accidentally exposed

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|---------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` in the activated venv |
| `CORS error` in frontend | Make sure backend is running and BASE_URL is correct |
| Expo app can't connect | Use your machine's local IP (not `localhost`) for physical devices |
| `venv\Scripts\activate` fails | Run PowerShell as Administrator or use `Set-ExecutionPolicy RemoteSigned` |
| Port 8000 already in use | Change `PORT=8001` in `.env` and update frontend BASE_URL |

---

## 📄 License

This project is for personal and educational use.

---

## 🙏 Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/) — Modern Python web framework
- [Expo](https://expo.dev/) — React Native toolchain
- [Google Gemini](https://ai.google.dev/) — LLM for intelligent responses
- [Sentence Transformers](https://www.sbert.net/) — Local embedding models
- [HuggingFace](https://huggingface.co/) — Model hosting
